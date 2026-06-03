# 에이전트 도구 셋업 — 스킬 · MCP 서버 · 검증 라이브러리

> 대상: 이 repo에서 **Claude Code + Codex** 가 공용으로 쓰는 보조 도구.
> Agent skill, MCP 출처(GitHub): ovrtx · PhysX/ovphysx · kit-usd-agents (아래 §출처). 본 문서는 조사·판정·설치·활용을 한 곳에 정리.
> 관련: [`SIM2REAL_MASTERPLAN.md`](SIM2REAL_MASTERPLAN.md) · [`../TASKS.md`](../TASKS.md)

## 0. 한눈에 — 무엇을 깔까

| 도구 | 정체 | 적합도 | 판정 | API 키 | 지금 설치 가능? |
|---|---|---|---|---|---|
| ovrtx 스킬 (큐레이션 10개) | Agent SKILL.md (마크다운) | USD 저작·SemanticsAPI(=Phase C segmentation) | **설치 T1** | 불요 | ✅ 지금 |
| ovphysx | USD 물리 시뮬 **라이브러리** (pip) | Phase A 물리 게이트 고속 자기검증 | **설치 T2** | 불요 | ✅ 지금 (pre-release 주의) |
| USD Code MCP (9903) | RAG 문서검색 서버 | USD API ↔ de-leisaac 재작성 | **설치 T1** | **필수** | ⛔ 키 대기 |
| Isaac Sim MCP (9904) | RAG 문서검색 서버 | Isaac Sim/Lab API ↔ 재구현 | **설치 T1** | **필수** | ⛔ 키 대기 |
| Kit MCP (9902) | RAG 문서검색 서버 | Kit settings/extension 보조 | **설치 T2** | **필수** | ⛔ 키 대기 |
| OmniUI MCP (9901) | RAG 문서검색 서버 | omni.ui GUI (우린 headless) | **스킵** | 필수 | — |
| ovrtx 비-USD 스킬 ~23개 | SKILL.md | ovrtx 렌더러 전용(우리 스택 아님) | **스킵(노이즈)** | — | — |

> **정정**: `ref_repos/PhysX/ovphysx` 는 "Agent skills" 가 아니라 **라이브러리**다 (SKILL.md 0개 확인). ovrtx 스킬은 **ovrtx 렌더러 라이브러리** 전용 — USD 부분만 차용하고 렌더러 호출부는 Isaac Sim API로 치환해야 한다.

---

## 출처 (GitHub)

스킬 10개는 이미 `.claude/skills`·`.agents/skills` 에 vendored.

| 도구 | GitHub / 패키지 |
|---|---|
| ovrtx 스킬 | https://github.com/NVIDIA-Omniverse/ovrtx (`skills/<name>/SKILL.md`, snippet 소스 `tests/docs/`) |
| ovphysx | https://github.com/NVIDIA-Omniverse/PhysX (`ovphysx/`) · PyPI https://pypi.org/project/ovphysx · 문서 https://nvidia-omniverse.github.io/PhysX/ovphysx/latest/ |
| MCP 4종 | https://github.com/NVIDIA-Omniverse/kit-usd-agents (`source/mcp/<name>/`) |

---

## 1. MCP 서버는 전부 API 키가 필요한가? → **그렇다**

4개 모두 **RAG 문서검색 서버**다 (USD/Isaac을 실행하지 않음 — 지식 검색만). 구조: `MCP → 임베더 + 리랭커 → Atlas DB`.

| 모드 | NVIDIA_API_KEY | NGC_API_KEY | GPU | 비고 |
|---|---|---|---|---|
| 클라우드 (권장) | **필수** | 불요 | 불요 | 임베딩·rerank·LLM 전부 NVIDIA 클라우드 |
| 로컬 NIM | **필수** | 필수 | **2장** | 임베더/리랭커만 로컬, NVIDIA 키는 여전히 필요 |

`.env.example` 에서 `NVIDIA_API_KEY` 는 무조건 **Required**. → **키 없이 동작하는 MCP 모드는 없다.** 키 발급(`build.nvidia.com/settings/api-keys`)이 풀릴 때까지 MCP 3종은 보류, 스킬·ovphysx 먼저 진행.

---

## 2. 설치 대상 최종 목록 (Tier 2까지)

### 스킬 (ovrtx 큐레이션 10개 — 키 불요)
```
semantic-labels      ★ Phase C 배경 오버레이(USD SemanticsAPI → segmentation/SemanticIdMap)
loading-usd            USD 씬 로드·composition·inline USDA
writing-transforms     prim transform(이동/회전/스케일)
writing-attributes     attribute 쓰기(머티리얼/색/메시)
reading-attributes     attribute 읽기(transform/mesh 샘플)
stage-queries          prim 탐색·attribute 스키마 조회
binding-materials      머티리얼 바인딩
cloning-prims          USD subtree 복제
reading-render-output  렌더 픽셀 readback(Phase C 검증)
camera-outputs-rt2     카메라 AOV/render var(depth/normal/segmentation)
```

### MCP 서버 (키 필요 — 발급 후)
```
USD Code MCP   (port 9903)   T1
Isaac Sim MCP  (port 9904)   T1
Kit MCP        (port 9902)   T2
```

### 라이브러리 (키 불요)
```
ovphysx   (pip / uv) — Phase A 물리 게이트 검증 전용
```

---

## 3. 스킬 설치 — Claude + Codex (지금 가능)

스킬은 마크다운이라 설치=복사. 두 에이전트 디렉토리에 둔다.

| 에이전트 | 스킬 경로 | 자동 사용? |
|---|---|---|
| Claude Code | `.claude/skills/<name>/SKILL.md` | ✅ description 트리거로 자동 호출 |
| Codex | `.agents/skills/<name>/SKILL.md` | Codex 스킬 로딩 방식 **확인 필요**(ovrtx repo가 이 규약 사용) |

### 설치 (Git Bash, repo 루트에서)

> ✅ 이미 실행 완료 — 스킬 vendored(`.claude/skills`·`.agents/skills`). `ref_repos/` 삭제 후 재취득: `git clone https://github.com/NVIDIA-Omniverse/ovrtx` 후 동일 복사.

```bash
SKILLS="semantic-labels loading-usd writing-transforms writing-attributes \
reading-attributes stage-queries binding-materials cloning-prims \
reading-render-output camera-outputs-rt2"

for s in $SKILLS; do
  mkdir -p ".claude/skills/$s" ".agents/skills/$s"
  cp "ref_repos/ovrtx/skills/$s/SKILL.md" ".claude/skills/$s/SKILL.md"
  cp "ref_repos/ovrtx/skills/$s/SKILL.md" ".agents/skills/$s/SKILL.md"
done

# ⚠️ 노이즈 정리: ref_repos 의 자동스캔되는 스킬 디렉토리 제거
#    (현재 ref_repos/ovrtx/.claude/skills 때문에 33개 전체가 세션에 로드됨)
rm -rf ref_repos/ovrtx/.claude ref_repos/ovrtx/.agents
```

### 캐비엇 (중요)
- **렌더러 API 치환**: 스킬 본문에 ovrtx `renderer.open_usd_from_string()` 등 호출이 섞여 있다. **USD/SemanticsAPI 패턴만 차용**하고 실제 호출은 Isaac Sim/usd-core API로 바꾼다. 각 복사본 frontmatter `description` 끝에 한 줄 주석 권장: `(이 프로젝트에선 USD 부분만 참고, ovrtx 렌더러 호출은 Isaac Sim/Replicator로 치환)`.
- **Source 링크**: SKILL.md 의 `> Source: tests/docs/...` 는 ovrtx repo 기준 상대경로다. `ref_repos/` 삭제 후엔 [ovrtx GitHub](https://github.com/NVIDIA-Omniverse/ovrtx) 의 `tests/docs/...` 로 참조. (본문 표·설명만으로도 충분히 유용.)
- 이미 있는 프로젝트 스킬 `usd-scene-builder` 와 상호보완(그쪽은 우리 펜 씬 전용, ovrtx는 일반 USD API).

---

## 4. MCP 서버 설치 — Claude + Codex (키 발급 후)

### 사전 요건 (호스트 빌드 도구)
| 항목 | 용도 | 전역설치 회피 |
|---|---|---|
| `NVIDIA_API_KEY` (nvapi) | 클라우드 임베딩/rerank/LLM | build.nvidia.com — **현재 차단** |
| Docker | 이미지 빌드/실행 | 보유 |
| Git LFS | FAISS 인덱스 pull(없으면 first-call 무음 실패) | `build-wheels.sh` 가 자동 pull, 바이너리만 PATH |
| Poetry + py3.11~3.13 | 휠 빌드(특히 isaacsim_mcp) | `uvx poetry` / `pipx` 로 전역 회피 |

> `docker-compose.local.yaml` 은 **로컬 NIM(2 GPU + NGC)** 용이라 안 씀. **클라우드 모드 = 이미지별 `docker run` + NVIDIA 키만**.

### 빌드·실행 (Windows 워크스테이션, CPU·인터넷만)
```bash
git clone https://github.com/NVIDIA-Omniverse/kit-usd-agents.git   # ref_repos 삭제됐으므로 재클론
cd kit-usd-agents/source/mcp
cp .env.example .env          # NVIDIA_API_KEY=nvapi-... (NGC 빈칸)

# T1
( cd usd_code_mcp  && ./build-docker.sh )
docker run -d --name usd-code-mcp  -p 9903:9903 --env-file ../.env --restart unless-stopped usd-code-mcp:latest
( cd isaacsim_mcp  && ./build-docker.sh )
docker run -d --name isaacsim-mcp  -p 9904:9904 --env-file ../.env --restart unless-stopped isaacsim-mcp:latest
# T2 (선택)
( cd kit_mcp       && ./build-docker.sh )
docker run -d --name kit-mcp       -p 9902:9902 --env-file ../.env --restart unless-stopped kit-mcp:latest

python usd_code_mcp/check_mcp_health.py   # 검증(POST /mcp initialize)
```

### 등록 — Claude Code (repo 스코프 `.mcp.json`)
repo 루트 `.mcp.json`:
```json
{ "mcpServers": {
  "usd-code-mcp":  { "type": "http", "url": "http://localhost:9903/mcp" },
  "isaac-sim-mcp": { "type": "http", "url": "http://localhost:9904/mcp" },
  "kit-mcp":       { "type": "http", "url": "http://localhost:9902/mcp" }
}}
```
또는 `claude mcp add usd-code-mcp -t http http://localhost:9903/mcp` (전역은 `--scope user`).

### 등록 — Codex
`~/.codex/config.toml` (또는 프로젝트 config) 에 `[mcp_servers.usd-code-mcp]` 등 추가.
⚠️ **Codex 의 streamable-HTTP MCP 지원 여부 버전 확인.** stdio만 지원하면 http→stdio 브릿지로 래핑:
```toml
[mcp_servers.usd-code-mcp]
command = "npx"
args = ["-y", "mcp-remote", "http://localhost:9903/mcp"]
```

### 실행 위치
오케스트레이터 호스트(**Windows**)에 띄우고 양 에이전트가 localhost로 접속. 서버측 워커가 USD/Isaac 지식이 필요하면 LAN IP(`http://<win-ip>:9903/mcp`)로 가리키거나 서버에도 동일 기동.

---

## 5. ovphysx 설치 — 검증 전용 (지금 가능)

USD 물리 시뮬 라이브러리. **Phase A 물리 게이트를 Isaac Sim 부팅 없이 고속 검증**(scene 로드→step→rigid-body pose 텐서 read).

출처: PyPI https://pypi.org/project/ovphysx · GitHub https://github.com/NVIDIA-Omniverse/PhysX (`ovphysx/`) · 문서 https://nvidia-omniverse.github.io/PhysX/ovphysx/latest/

### 설치 (uv 격리, pre-release 핀)
`pyproject.toml` 에 별도 그룹(예 `validation`) — ABI 핀 충돌 회피 위해 isaac 그룹과 분리:
```toml
[dependency-groups]
validation = ["ovphysx"]   # pre-release: 동작 확인 후 정확 버전 핀
```
`uv sync --group validation` (또는 별 venv).

### 사용 — 게이트 스크립트 예 (`scripts/validate_scene_physics.py`)
```python
from ovphysx import PhysX
from ovphysx.types import TensorType
import numpy as np

physx = PhysX(device="cpu")
physx.add_usd("assets/scenes/pen_desk/scene.usd"); physx.wait_all()
for _ in range(120):                      # 2초 settle
    physx.step(1/60, 0.0)
b = physx.create_tensor_binding(pattern="/World/.../PenWhite", tensor_type=TensorType.RIGID_BODY_POSE)
pose = np.zeros(b.shape, np.float32); b.read(pose)
# z 관통(<0) / 영역 이탈 / settle 후 큰 이동(바운스) 판정 → exit code
physx.release()
```

### 캐비엇
- **pre-release** (API 변동) → 핵심 의존 금지, 검증 보조로만.
- **Isaac Sim 과 동일 프로세스 금지** (자체 OV USD 런타임 번들 — schema 경로 충돌). 독립 스크립트로만.
- numpy 1.26 등 ABI 핀과 충돌 가능 → 별 그룹/별 venv.

---

## 6. 단계별 활용

| 단계 | 도구 | 용도 |
|---|---|---|
| 0d de-leisaac | USD Code MCP · ovrtx(stage-queries/writing-attributes) | UsdPhysics·subasset API 확인하며 재작성 |
| A 물리 정합 | **ovphysx** · USD Code MCP | 관통/바운스 게이트 고속검증, friction/contactOffset 스키마 |
| B RL env | Isaac Sim MCP | ManagerBasedRLEnv·rsl_rl·settings 패턴 길잡이 |
| C 오버레이 | ovrtx **semantic-labels** · reading-render-output · Isaac Sim MCP | SemanticsAPI 라벨→segmentation, Replicator 합성, 픽셀 readback |

---

## 7. 리스크

- MCP는 **검색기** — 코드 실행/USD 편집 못 함. 항상 우리 게이트(env_smoke, validator)로 재검증.
- 자율 루프 폭주 호출 → NVIDIA 클라우드 **rate limit(429)**.
- **Codex HTTP MCP 미확정** → 브릿지 필요 가능.
- **ovphysx pre-release** → 보조 한정.
- MCP 코퍼스가 Isaac Sim 5.1·Lab 2.3.2 정확 일치 보장 안 됨 → 공식 문서/실제 import 교차검증.

---

## 8. 현재 상태 / 다음

- ✅ **지금**: 스킬 10개 복사(§3) + ref_repos 노이즈 정리 + ovphysx 그룹(§5). 키 불요.
- ⛔ **차단**: MCP 3종 — `NVIDIA_API_KEY` 발급 대기. 발급되면 §4 실행.
- 플랜 반영: 마스터플랜 §12 + TASKS Phase A 게이트에 활용 지점 명시(자동 사용 안 되는 ovphysx 위주).
