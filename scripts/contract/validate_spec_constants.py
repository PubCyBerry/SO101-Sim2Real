#!/usr/bin/env python3
"""명세서(`docs/spec/`) 상수 대장 ↔ 코드 상수 대조 — 드리프트 감지기.

값의 단일 소스는 **코드**이고 명세서는 사본이다. 사본이 갈라지는 것만 잡는다.

대상 = `docs/spec/*.md` 안의 3-열 표 행 중 아래 형태:

    | `SYMBOL` | `<python repr>` | `<경로>::SYMBOL` |

각 행마다 앵커 파일을 **AST 로 파싱**해(import 하지 않는다) 최상위 리터럴 대입을 찾고,
그 `repr()` 이 문서에 적힌 값과 같은지 본다. 따라서 의존성 0 — Isaac Sim·GPU·numpy 불요이며
`isaaclab` 을 import 하는 모듈도 검사할 수 있다.

한계: **Python 리터럴 상수만** 다룬다. dataclass 호출·comprehension 파생값
(`CUBE_SPECS`, `CUBE_SIZES` 등)은 대장에 싣지 않고 해당 절에 서술한다.

실행:

    python3 scripts/contract/validate_spec_constants.py            # 대조
    python3 scripts/contract/validate_spec_constants.py --self-test  # 검증기 자체 점검

exit 0 = 전부 일치, 1 = 불일치 또는 표를 못 읽음.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_DIR = REPO_ROOT / "docs" / "spec"

# | `SYMBOL` | `value` | `path::SYMBOL` |
ROW = re.compile(
    r"^\|\s*`(?P<name>[A-Za-z_][A-Za-z0-9_]*)`\s*"
    r"\|\s*`(?P<value>.+?)`\s*"
    r"\|\s*`(?P<path>[\w./-]+\.py)::(?P<anchor>[A-Za-z_][A-Za-z0-9_]*)`\s*\|\s*$"
)


def literal_constants(py_path: Path) -> dict[str, object]:
    """최상위 `NAME = <literal>` 대입만 수집 (import 하지 않음)."""
    out: dict[str, object] = {}
    for node in ast.parse(py_path.read_text(encoding="utf-8")).body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        target = node.targets[0] if isinstance(node, ast.Assign) else node.target
        if not isinstance(target, ast.Name) or node.value is None:
            continue
        try:
            out[target.id] = ast.literal_eval(node.value)
        except (ValueError, SyntaxError, TypeError):
            pass  # 비-리터럴(호출·comprehension)은 대상 밖
    return out


def check(spec_dir: Path, repo_root: Path) -> tuple[int, list[str]]:
    """(검사한 행 수, 오류 메시지 목록)."""
    cache: dict[Path, dict[str, object]] = {}
    errors: list[str] = []
    checked = 0

    for md in sorted(spec_dir.glob("*.md")):
        for lineno, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
            m = ROW.match(line)
            if not m:
                continue
            checked += 1
            # --spec-dir 로 repo 밖(임시 사본)을 검사할 수도 있으므로 relative_to 는 방어적으로.
            try:
                shown = md.relative_to(repo_root)
            except ValueError:
                shown = md
            where = f"{shown}:{lineno} `{m['name']}`"

            if m["name"] != m["anchor"]:
                errors.append(f"{where}: 심볼명 불일치 (앵커는 `{m['anchor']}`)")
                continue

            py = repo_root / m["path"]
            if not py.is_file():
                errors.append(f"{where}: 앵커 파일 없음 — {m['path']}")
                continue

            if py not in cache:
                cache[py] = literal_constants(py)
            if m["anchor"] not in cache[py]:
                errors.append(f"{where}: 코드에 리터럴 상수가 없음 — {m['path']}")
                continue

            actual = repr(cache[py][m["anchor"]])
            if actual != m["value"]:
                errors.append(f"{where}: 문서 `{m['value']}` != 코드 `{actual}`")

    return checked, errors


ANCHOR = re.compile(r"`([A-Za-z_0-9/.-]+\.(?:py|sh|yaml|yml|toml|env))::[A-Za-z_]")
_SKIP_DIRS = {".git", "__pycache__", ".claude", "ref_repos", ".venv", "node_modules"}


def check_anchors(spec_dir: Path, repo_root: Path) -> tuple[int, list[str]]:
    """문서 전체의 `경로::심볼` 앵커가 실재 파일을 가리키는지 확인.

    본문 산문은 절 머리에서 전체 경로를 밝힌 뒤 파일명만 쓰기도 한다. 그래서 repo 상대경로가
    아니면 **경로 접미사가 일치하는 파일**로 해석하고, 후보가 0개(rename/삭제)거나 2개 이상
    (모호)이면 오류로 본다.
    """
    index: dict[str, list[Path]] = {}
    for p in repo_root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(repo_root)  # skip 판정은 **repo 상대경로** 기준
        if any(part in _SKIP_DIRS for part in rel.parts):
            continue
        index.setdefault(p.name, []).append(rel)

    anchors: set[str] = set()
    for md in [*sorted(spec_dir.glob("*.md")), spec_dir.parent / "SPEC.md"]:
        if md.is_file():
            anchors.update(ANCHOR.findall(md.read_text(encoding="utf-8")))

    errors = []
    for a in sorted(anchors):
        if (repo_root / a).is_file():
            continue
        cands = [c for c in index.get(Path(a).name, []) if str(c).endswith(a)]
        if not cands:
            errors.append(f"앵커 경로 해석 실패 — `{a}` (rename/삭제?)")
        elif len(cands) > 1:
            errors.append(f"앵커 경로 모호 — `{a}` → {[str(c) for c in cands]}")
    return len(anchors), errors


def _self_test() -> int:
    """임시 픽스처로 일치/불일치/누락 검출을 확인한다."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "pkg").mkdir()
        # isaaclab 처럼 import 불가한 모듈이어도 AST 는 통과해야 한다.
        (root / "pkg" / "mod.py").write_text(
            "import isaaclab_does_not_exist\nGOOD = (0.1, 0.2)\nBAD = 3\nCALL = dict(a=1)\n",
            encoding="utf-8",
        )
        spec = root / "spec"
        spec.mkdir()
        (spec / "x.md").write_text(
            "| 심볼 | 값 | 앵커 |\n|---|---|---|\n"
            "| `GOOD` | `(0.1, 0.2)` | `pkg/mod.py::GOOD` |\n"        # 일치
            "| `BAD` | `4` | `pkg/mod.py::BAD` |\n"                   # 값 불일치
            "| `MISSING` | `1` | `pkg/mod.py::MISSING` |\n"           # 코드에 없음
            "| `CALL` | `{'a': 1}` | `pkg/mod.py::CALL` |\n"          # 비-리터럴
            "| `GONE` | `1` | `pkg/nope.py::GONE` |\n",               # 파일 없음
            encoding="utf-8",
        )
        checked, errors = check(spec, root)
        n_anchors, anchor_errors = check_anchors(spec, root)

    problems = []
    if checked != 5:
        problems.append(f"행 인식 {checked} != 5")
    if len(errors) != 4:
        problems.append(f"오류 검출 {len(errors)} != 4: {errors}")
    if not any("!= 코드" in e for e in errors):
        problems.append("값 불일치 미검출")
    if not any("리터럴 상수가 없음" in e for e in errors):
        problems.append("누락/비-리터럴 미검출")
    if not any("앵커 파일 없음" in e for e in errors):
        problems.append("파일 없음 미검출")
    # 앵커는 **유일 경로** 단위 — 픽스처는 pkg/mod.py, pkg/nope.py 2개이고 후자만 실패한다.
    if n_anchors != 2:
        problems.append(f"앵커 인식 {n_anchors} != 2")
    if len(anchor_errors) != 1 or "해석 실패" not in anchor_errors[0]:
        problems.append(f"앵커 오류 검출 이상: {anchor_errors}")

    if problems:
        print("FAIL self-test:", "; ".join(problems), file=sys.stderr)
        return 1
    print("PASS self-test: 값 유효 1 + 값 오류 4 + 앵커 오류 1")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec-dir", type=Path, default=SPEC_DIR)
    parser.add_argument("--self-test", action="store_true", help="검증기 자체 점검")
    args = parser.parse_args()

    if args.self_test:
        return _self_test()

    if not args.spec_dir.is_dir():
        print(f"FAIL: 명세 디렉터리 없음 — {args.spec_dir}", file=sys.stderr)
        return 1

    checked, errors = check(args.spec_dir, REPO_ROOT)
    n_anchors, anchor_errors = check_anchors(args.spec_dir, REPO_ROOT)

    # 파서가 조용히 0건 통과하는 실패모드 차단.
    if checked == 0:
        print(f"FAIL: {args.spec_dir} 에서 상수 표 행을 하나도 못 읽음", file=sys.stderr)
        return 1

    for e in errors + anchor_errors:
        print(f"  {e}", file=sys.stderr)
    if errors or anchor_errors:
        print(
            f"FAIL: 상수 {len(errors)}/{checked} · 앵커 {len(anchor_errors)}/{n_anchors} 불일치",
            file=sys.stderr,
        )
        return 1

    print(f"OK: 상수 {checked}/{checked} 일치 · 앵커 {n_anchors}/{n_anchors} 해석")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
