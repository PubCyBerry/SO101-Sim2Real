#!/usr/bin/env python3
"""진입 스크립트의 **미정의 전역 이름**을 정적으로 잡는다 — import 없이, GPU 없이.

## 왜 필요한가 (실측 근거, 2026-08-04)

`_pre_back` 을 `curobo_batch_planner.py` 에서 `so101_contract.grasp_manifold` 로 올릴 때
**호출부를 안 고쳤다.** 그런데:

* `python -m py_compile` 통과 — 문법은 맞다
* import 도 성공 — 그 이름은 **함수 안**에서만 참조된다
* self-check 통과 — 그 경로를 안 탄다

그래서 **GPU 로 SM 을 실제로 돌린 뒤에야** 터졌다::

    [sm] planner ERROR: NameError: name '_pre_back' is not defined
    [sm] plan FAIL (all 4 envs)

planner 는 이 예외를 잡아 `ok=False` 로 응답하므로 프로세스가 죽지도 않는다 — 전 env 계획
실패가 조용히 "성공률 0%" 로 보인다. 리팩토링으로 심볼을 옮길 때마다 재발할 수 있는 부류다.

## 무엇을 검사하나

각 파일을 AST 로 읽어 **Load 문맥의 `Name`** 중 그 파일 안에서 정의·import·바인딩되지 않은
것을 찾는다. 런타임 import 가 필요 없어 호스트에서 즉시 돈다.

한계: `globals()` 조작·`exec`·조건부 import 는 못 본다. 그 대신 **거짓 양성이 없어야** 하므로
바인딩 형태(할당·함수/클래스 정의·import alias·인자·except as·global/nonlocal·comprehension·
walrus·with as·for target)를 전부 수집한다.

사용:
    python scripts/contract/validate_no_undefined_names.py            # 기본 대상
    python scripts/contract/validate_no_undefined_names.py a.py b.py  # 지정
    python scripts/contract/validate_no_undefined_names.py --self-check
"""

from __future__ import annotations

import argparse
import ast
import builtins
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: 심볼 이동 리팩토링이 잦고, 실패가 조용한(예외를 삼키는) 진입점들.
DEFAULT_TARGETS = (
    "scripts/cuRobo/curobo_batch_planner.py",
    "scripts/cuRobo/pickplace_sm.py",
    "scripts/datagen/generate_mimic_dataset.py",
    "scripts/datagen/annotate_mimic_demos.py",
    "src/sim_to_real/datagen/skillgen_planner.py",
    "src/sim_to_real/tasks/pick_cube/mimic_env.py",
    "src/sim_to_real/tasks/pick_cube/mimic_env_cfg.py",
    "src/so101_contract/grasp_manifold.py",
    "src/so101_contract/curobo_frames.py",
)


def bound_names(tree: ast.AST) -> set[str]:
    """모듈 안에서 이름이 **묶이는** 모든 형태를 수집한다."""
    names = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)          # 할당·for target·comprehension·walrus·with as 포함
        elif isinstance(node, ast.alias):
            names.add((node.asname or node.name).split(".")[0])
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
    return names


def undefined_names(source: str) -> list[str]:
    """Load 문맥에서 쓰이는데 어디서도 안 묶인 이름(정렬)."""
    tree = ast.parse(source)
    used = {n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return sorted(used - bound_names(tree))


def check(paths: list[Path]) -> int:
    failed = 0
    for path in paths:
        if not path.exists():
            print(f"[skip] {path} — 없음")
            continue
        missing = undefined_names(path.read_text(encoding="utf-8"))
        rel = path.relative_to(_REPO_ROOT) if path.is_relative_to(_REPO_ROOT) else path
        if missing:
            failed += 1
            print(f"[FAIL] {rel}: {', '.join(missing)}")
        else:
            print(f"[ok]   {rel}")
    return failed


def self_check() -> int:
    """양성(잡아야 한다) 1 · 음성(잡으면 안 된다) 5."""
    positive = "def f():\n    return _moved_symbol(1)\n"
    assert undefined_names(positive) == ["_moved_symbol"], "이동한 심볼을 못 잡는다"

    negatives = {
        "import 별칭": "import numpy as np\ndef f():\n    return np.zeros(3)\n",
        "from import": "from math import cos\ndef f():\n    return cos(0)\n",
        "except as": "def f():\n    try:\n        pass\n    except ValueError as exc:\n        return exc\n",
        "comprehension·walrus": "def f(xs):\n    return [(y := x * 2) for x in xs] + [y]\n",
        "with as·global": "g = 1\ndef f(p):\n    global g\n    with open(p) as fh:\n        return fh, g\n",
    }
    for label, code in negatives.items():
        found = undefined_names(code)
        assert not found, f"거짓 양성({label}): {found}"

    print(f"[self-check] PASS — 양성 1 · 음성 {len(negatives)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paths", nargs="*", help="검사할 .py (생략 시 기본 대상)")
    parser.add_argument("--self-check", action="store_true", help="검사기 자체 검증")
    args = parser.parse_args()

    if args.self_check:
        return self_check()

    targets = ([Path(p).resolve() for p in args.paths] if args.paths
               else [_REPO_ROOT / p for p in DEFAULT_TARGETS])
    failed = check(targets)
    print(f"\n{'FAIL' if failed else 'PASS'} — {failed}/{len(targets)} 파일에 미정의 이름")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
