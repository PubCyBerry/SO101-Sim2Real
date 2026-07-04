# Autoresearch 결과 요약 — pink IK pick-place SM (branch `autoresearch/jul3`)

**목표**: `autoresearch/program.md` 루프로 `pink_ik_bridge_node.py` 단일 파일을 반복 개선,
N_EP=20 DR-v0 eval `success_rate` 1.0 달성.
**최종**: **best = 0.75** (commit `bec5b14`). 1.0 미달. 아래는 왜 0.75가 robust ceiling인지의 규명.

편집 대상은 `scripts/datagen/pink_ik_bridge_node.py` 단독. harness·bridge·env config·판정기준
(cube-in-bowl) 고정. 실험 로그 = `autoresearch/results.tsv`.

---

## 1. 실패 taxonomy (baseline `bec5b14`, fails = ep2/4/5/9/13)

| 부류 | 에피소드 | 원인 |
|---|---|---|
| **Mode-B (lateral)** | ep2, ep13 | gripper가 큐브서 **옆으로** 어긋남(pink↔sim FK link mismatch). top-down·droop 무관 |
| **far-reach droop** | ep4, ep5, ep9 | bell **밑동**(low-y, 옆쪽) 큐브. 팔 extension서 elevation 관절이 sag → gripper가 큐브 **위+뒤 ~20mm**서 헛닫힘 |
| **marginal flip** | ep14, ep15, ep17 | baseline PASS지만 물리 noise로 run마다 ±1ep PASS/FAIL |

**핵심 진단(`autoresearch/pink.log` = harness가 pink stdout 캡처)**: 全 에피소드가 select_grasp에서
g_err 0.7~2mm로 grasp를 **정확히 찾음**(dead-zone 0건). 실패는 planning이 아니라 **물리**(droop·FK).

---

## 2. 이번 세션 실험 (exp39–49)

| exp | 아이디어 | 결과 | 왜 |
|---|---|---|---|
| 39 | visual servo (predictive) | 0.75 | Mode-B 고치나 far-marginal 파손 |
| 40 | dev-gated reactive servo | 0.70 | servo 3변종 全 ≤0.75(Mode-B↔marginal 상쇄) |
| 41 | droop FF windup clamp 25→45° | 0.70 | **far-edge = Jacobian-limited droop 확정**: 팔 extension서 elevation FF가 TCP z를 못 내림(ff 45°여도 z 0.081 stuck). clamp 무효 + marginal 파손 |
| **42** | **sink-stall reactive tilt** | 0.75 | **★11실험+FF 다 실패한 far-edge(ep4)를 tilt로 물리 fix한 첫 기전.** 단 stall이 edge+marginal 둘다 발동 → marginal 상쇄 |
| 43 | +조기발동+짧은접근 | 0.65 | short-approach가 near-marginal descent 급격→4개 파손 |
| 44 | 조기발동만 | 0.70 | 조기발동이 marginal 파손 |
| 45 | outward-gated tilt | 0.65 | outward false-catch가 edge-tilt 차단 |
| **46** | **cube-z-follow slip-tilt** | 0.70 | **★ground-truth 파지신호(lift 중 cube가 gripper 따라 오르나)로 edge↔marginal 직접 구별 — discriminator 검증됨**(정상 큐브 무발동·Mode-B slip 정확검출). 단 ①edge는 outward 경유로 slip-tilt 지각 ②느린 marginal false-positive |
| 47 | (user) 헛닫힘 wide-open 재파지 | 0.65 | 오프셋이 **수직**(위+뒤)이라 수평 개구확대 무효 + marginal bulldoze |
| 48 | (user unlock) margin-select shallow | 0.75 | **★joint-limit margin이 실패와 完 무상관 확정**(fail margin 25-32°가 PASS와 겹침) → shallow 선택 불가 |
| 49 | (user unlock) universal radial-shallow | (self-check fail) | radial α-tilt가 bell-end(α20° unplannable)·near-arm(α12° unplannable) 못 잡고 g_err 11-17mm 부정확 → universal 불가 |

---

## 3. 결론 — 0.75 robust ceiling

**~19실험(prior exp28-38 + exp39-49) 5+ mechanism family 全이 "일부 fix ↔ 동수 marginal 파손"으로 0.75.**

근본 원인: **실패 큐브(edge/Mode-B)와 marginal(baseline pass)이 노드가 관측 가능한 signal 상 구별 불가.**
- 위치·yaw·stall 깊이·gripper feature(g)·**joint-limit margin** 全 overlap (exp48 map으로 margin 무상관 확정)
- 유일한 clean signal = **cube-z-follow(ground-truth 파지)**(exp46) 이나 lift-후 timing이라 edge 재파지 예산 부족 + 느린 marginal false-positive
- geometry(shallow) 회피도 radial α-tilt로는 bell-base side-edge 도달 불가(exp49 self-check)

**미완 유망 리드(future work)**:
1. **cube-z-follow probe-lift**: grasp 직후 probe-lift(2cm)+cube 확인으로 slip을 outward 전에 조기검출 (exp46 timing fix)
2. **lateral-tilt approach**: radial이 아니라 tangential(큐브 옆) tilt — bell-base side-edge 도달용 (exp49가 radial의 한계 규명)
3. **out-of-node**: URDF↔USD FK 캘리브레이션 or bridge EE tf (node 밖, 현 제약선 금지)

상세 실험 로그 = `autoresearch/results.tsv`. 진단 도구 = `autoresearch/pink.log`(에피소드별
`[grasp]`/`[sink#]`/tilt 로그). 메모리 = `autoresearch-pink-droop-findings`.
