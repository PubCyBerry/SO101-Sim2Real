# USD 포맷 가이드 (Windows)

## 1. USD 포맷 종류 및 차이점

| | `.usda` | `.usdc` | `.usdz` | `.usd` |
|---|---|---|---|---|
| **형식** | 텍스트 (ASCII) | 바이너리 | ZIP 묶음 | 텍스트 또는 바이너리 |
| **가독성** | 사람이 읽을 수 있음 | 불가 | 불가 | 확장자만으론 알 수 없음 |
| **파일 크기** | 큼 | 작음 | 작음 (압축) | 가변 |
| **로드 속도** | 느림 | 빠름 | 보통 | 가변 |
| **직접 편집** | 텍스트 에디터로 가능 | 불가 | 불가 | 가변 |
| **외부 참조** | 가능 | 가능 | ❌ 자기완결형 | 가능 |
| **주 용도** | 디버깅, 개발, 버전관리 | 프로덕션, 대용량 씬 | 배포, Apple AR Quick Look | 범용 |

### 포맷 관계

```
.usda  ──┐
          ├── usdcat으로 상호 변환 가능
.usdc  ──┘

.usdz  = .usdc (또는 .usda) + 텍스처 등 의존 파일을 ZIP으로 묶은 것
.usd   = .usda 또는 .usdc 중 하나 (파일 열어봐야 알 수 있음)
```

- `.usdz` 패키징은 `usdcat` 대신 `usdzip` 도구를 사용한다.

---

## 2. 본 프로젝트의 USD 의존성

`scripts/author_pick_pen_scene.py` 의 `.usda` → `.usdc` 변환은 `pyproject.toml` 공용 의존성 `usd-core>=26.5` 의 Python API 로 처리한다. CLI(`usdcat`) 가 PATH 에 없어도 `uv run python scripts/author_pick_pen_scene.py` 가 동작한다.

```python
from pxr import Sdf

layer = Sdf.Layer.FindOrOpen("input.usda")
layer.Export("output.usd", args={"format": "usdc"})  # binary PXR-USDC
```

씬을 GUI 로 열어보거나 사후 변환·flatten 이 필요할 때만 아래 NVIDIA 바이너리(`usdview`, `usdcat`) 설치를 검토하면 된다.

---

## 3. NVIDIA OpenUSD CLI 설치 (선택)

### 방법 1: NVIDIA 사전 빌드 바이너리 (권장)

`usdcat`, `usdview` 등 전체 툴셋이 포함된다.

1. [OpenUSD Developer Resources](https://developer.nvidia.com/usd) 접속
2. **For Windows** 링크에서 ZIP 다운로드
3. 원하는 경로에 압축 해제 (예: `C:\OpenUSD\`)

### 방법 2: pip 만 (Python API)

Python 3.9 이상, 3.13 미만 필요. 본 레포는 이 경로를 `uv` 로 자동 적용한다.

```powershell
uv pip install usd-core
```

> **주의:** `usd-core`는 Python API만 제공하며 `usdcat` 같은 CLI 도구는 포함되지 않는다. 본 프로젝트의 author 스크립트는 API 만 사용하므로 CLI 없이도 동작한다.

### 방법 3: PATH 영구 등록

```powershell
[Environment]::SetEnvironmentVariable(
  "PATH",
  $env:PATH + ";C:\OpenUSD\bin",
  "User"
)
[Environment]::SetEnvironmentVariable(
  "PYTHONPATH",
  "C:\OpenUSD\lib\python",
  "User"
)
```

PowerShell 재시작 후 `usdcat` 명령을 어디서든 사용할 수 있다.

---

## 4. usdc ↔ usda 변환

### Python API (`usd-core`, 본 레포 기본 경로)

```python
from pxr import Sdf

# binary(.usdc) → text(.usda)
Sdf.Layer.FindOrOpen("input.usdc").Export("output.usda", args={"format": "usda"})

# text(.usda) → binary(.usd as usdc)
Sdf.Layer.FindOrOpen("input.usda").Export("output.usd", args={"format": "usdc"})
```

`args={"format": "usdc"|"usda"}` 는 출력 확장자에 상관없이 포맷을 강제한다 — `usdcat --usdFormat` 옵션과 같다.

### usdcat 사용 (선택)

```powershell
# 기본 변환
C:\OpenUSD\scripts\usdcat.bat input.usdc -o output.usda

# 여러 레이어를 하나로 flatten해서 변환
C:\OpenUSD\scripts\usdcat.bat scene.usdc -o scene_flat.usda --flatten
```

### usdcat 주요 옵션

| 옵션 | 설명 |
|---|---|
| `-o output.usda` | 출력 파일 지정 |
| `--usdFormat usda\|usdc` | `.usd` 확장자 출력 시 포맷 명시 |
| `-f` / `--flatten` | 여러 레이어를 하나로 병합해서 출력 |
| `-l` / `--loadOnly` | 파일 로드 가능 여부만 검사 |

---

## 5. 시각화

### usdview (바이너리에 포함)

```powershell
# 절대 경로 권장 (상대 경로는 파일을 못 찾을 수 있음)
C:\OpenUSD\scripts\usdview_gui.bat D:\Workspaces\USDView\output.usda

# 또는 $PWD 활용
C:\OpenUSD\scripts\usdview_gui.bat "$PWD\output.usda"
```

### 카메라 조작

| 동작 | 방법 |
|---|---|
| 회전 | Alt + 좌클릭 드래그 |
| 패닝 | Alt + 중클릭 드래그 |
| 줌 | Alt + 우클릭 드래그 또는 스크롤 |
| 전체 화면에 맞추기 | `F` |
| 선택 오브젝트로 포커스 | 오브젝트 클릭 후 `F` |

카메라 프리셋은 메뉴 `View → Cameras` 또는 뷰포트 상단 드롭다운에서 변경한다 (Front / Back / Left / Right / Top / Bottom / Perspective).

### usdview 주요 기능

- **3D 뷰포트** — 마우스로 회전/줌/패닝
- **Stage 트리** — 계층 구조 탐색 (Prim, Attribute 확인)
- **Timeline** — 애니메이션 재생
- **Python 콘솔 내장** — 런타임에 USD API 직접 실행 가능

### 대안 도구

| 도구 | 특징 | 접근 방법 |
|---|---|---|
| **NVIDIA Omniverse** | 고품질 렌더링, 협업 | [omniverse.nvidia.com](https://www.nvidia.com/en-us/omniverse/) 무료 설치 |
| **Blender 4.x+** | USD import/export 내장 | File → Import → Universal Scene Description |
| **VSCode usdc-viewer** | 가벼운 에디터 내 미리보기 | VS Code 확장 마켓플레이스 |
