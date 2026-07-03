# autoresearch — pink IK pick-place SM

LLM 이 스스로 실험을 반복하는 자율 연구 루프 (karpathy `autoresearch` 이식).

**목표: full-DR(bell) 범위에 스폰된 단일 큐브를 집어 그릇에 안착시키는 성공률(success_rate)을 1.0 으로.**
1차 목표는 pick(집기) 전 구간 성공, 최종 목표는 pick+place(그릇 안착) 전 구간 성공이다 —
메트릭은 처음부터 place 기준(success_rate)이고, pick 만 된 에피소드는 ever_rate 로 구분된다.

**DR 범위 내 모든 배치는 human teleoperation 으로 pick-place 가능함이 이미 확인됐다.**
"이 셀은 물리적으로 불가능" 이라는 결론은 금지 — 실패는 전부 planner(이 노드)의 몫이다.

## 실행 환경 (setup 완료 상태)

- worktree `~/Workspaces/SO101-Sim2Real-autoresearch`, 브랜치 `autoresearch/jul3`. 여기서만 작업.
- docker 이미지(so101-pink-ik·so101-isaac-sim)는 메인 repo 와 공유(빌드 불요).
  compose 는 worktree 의 `scripts/`·`src/` 를 마운트하므로 노드 수정 → 재빌드 없이 즉시 반영.
- GPU 1장 공유 — bridge 는 동시에 1개만. 메인 repo 쪽 isaac-sim 컨테이너가 떠 있으면 실험 불가.
- 결과 확인 노트북: `autoresearch/analysis.ipynb` (results.tsv 시각화).

## 규칙

**편집 가능한 파일은 단 하나: `scripts/datagen/pink_ik_bridge_node.py`.**
grasp 전략·phase 구성·waypoint·IK 파라미터·타이밍 등 노드 내부는 전부 자유다.
top-down grasp 일 필요도 없다 — 중간 과정(접근 방향, 세그먼트 구성, 재시도 로직 등)은
성공률만 올린다면 어떻게 바꿔도 된다.

**금지:**
- harness(`autoresearch/run_experiment.sh`)·bridge(`scripts/inference/run_cube_desk_ros_bridge.py`)·
  env config(`src/sim_to_real/**`)·docker 파일 수정. 판정 기준(cube-in-bowl)은 고정 메트릭이다.
- 새 패키지 설치, 의존성 추가.
- `--self-check` 를 통과 못 하게 만드는 변경 (harness 1단계 게이트).
  self-check 케이스가 새 전략과 안 맞으면 self-check 코드도 노드 안이므로 함께 갱신해도 된다 —
  단 "실패해야 할 것을 통과시키는" 방향의 훼손은 금지.

## 1 실험 절차

1. 현재 git 상태 확인 (branch `autoresearch/jul3`).
2. `scripts/datagen/pink_ik_bridge_node.py` 에 실험 아이디어 1개를 구현.
3. `git add -A && git commit -m "<아이디어 요약>"`
4. 실행: `bash autoresearch/run_experiment.sh > autoresearch/run.log 2>&1` (tee 금지 — 출력 홍수 방지)
5. 결과: `grep "^success_rate:\|^ever_rate:\|^harness_status:" autoresearch/run.log`
6. `harness_status: ok` 가 아니면 크래시다. 진단 순서:
   - `selfcheck_failed` → `autoresearch/selfcheck.log` tail
   - `bridge_crashed`/`timeout` → `autoresearch/bridge.log` tail (인프라 문제일 수 있음 — 노드 탓 전에 확인)
   - 노드 자체 에러 → `autoresearch/pink.log` 에서 해당 에피소드 구간
7. `autoresearch/results.tsv` 에 기록 (아래 포맷, **커밋하지 말 것** — untracked 유지).
8. **success_rate 가 이전 최고보다 오르면 keep**(커밋 유지·branch 전진).
   동률이면 ever_rate 가 오르면 keep, 아니면 discard.
   **내려가면 discard: `git reset --hard HEAD~1`.**

## results.tsv 포맷

탭 구분 5열 (쉼표 금지 — description 에 쉼표 들어감):

```
commit	success_rate	ever_rate	status	description
a1b2c3d	0.40	0.55	keep	baseline
b2c3d4e	0.55	0.70	keep	approach 세그먼트 leg_sec 1.5로 단축
c3d4e5f	0.35	0.50	discard	grasp z 5mm 하향
d4e5f6g	0.00	0.00	crash	self-check 실패 (문법 에러)
```

crash 는 0.00/0.00 으로 기록.

## 실패 분석 도구 (읽기 전용)

- `autoresearch/pink.log`: 에피소드별 `=== episode N ===` 마커 + `cube(tf)=[x y z] ψ=..°` (스폰 위치·yaw),
  세그먼트별 TCP 추적 로그. **어느 셀(bell 어디)에서 실패하는지 여기서 특정하라.**
- `autoresearch/bridge.log`: `[eval] ep N/20: ... (ever k)` — place 실패 vs pick 실패 구분.
  에피소드 seed = `SEED+ep` 라 같은 실험 조건에서 레이아웃은 매번 동일(재현 가능).
- 오프라인 반복(무료·수초): `docker compose --env-file .env -f docker/docker-compose.yaml run --rm pink-ik python3 /workspace/scripts/datagen/pink_ik_bridge_node.py --self-check`
- 실패 위치를 노드의 `--sweep`(오프라인 kinematic map)으로 좁히는 것도 가능.

## 실험 루프

LOOP (사용자가 /loop 로 반복 호출):

1. results.tsv 와 직전 실험의 로그를 읽고 다음 아이디어 1개를 정한다.
   - 우선순위: 크래시 수정 > 실패 에피소드 공통 패턴(bell 특정 영역·yaw·near-bowl 등) 공략 > 파라미터 탐색.
2. 위 "1 실험 절차" 를 수행한다.
3. keep/discard 판단 후 tsv 기록.

**첫 실험은 반드시 무수정 baseline** (현재 코드 그대로 실행해 기준점 기록).

**단순성 기준**: 같은 성능이면 단순한 코드가 이긴다. 삭제로 얻은 개선이 최고의 개선이다.

**멈추지 말 것**: 실험 하나가 끝나면 결과를 기록하고 다음 아이디어로 넘어간다.
아이디어가 고갈되면 pink.log 의 실패 에피소드를 하나씩 뜯어보라 — 실패는 항상 남아 있다.

## 비용 감각

1 실험 ≈ bridge 부팅(3~5분) + 20 ep × ~25s ≈ **12~18분**. 시간당 3~4 실험.
빠른 반복이 필요한 가설(기하·IK 수렴)은 먼저 `--self-check`/`--sweep` 오프라인으로 거른 뒤
비싼 eval 에 올려라.
