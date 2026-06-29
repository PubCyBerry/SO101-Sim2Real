"""SO-101 follower USD 의 팔 관절 가동범위(joint position limit)를 조정.

배경: sim USD 의 revolute joint limit 이 실기기 가동범위보다 좁으면, 실기기에서 녹화한
데이터를 sim 으로 replay 할 때 그 관절이 물리 articulation 에서 clamp 돼 덜 뻗는다
(좌표계는 정합이어도 apex 포즈가 짧아짐). 실 calibration(so101_robot.json) 기준
elbow_flex·wrist_flex 는 100° 이상 도달 가능한데 USD 는 각각 90°·95° 로 잘려 있었다.

기본값 = ``leader_calibration.SO101_FOLLOWER_USD_JOINT_LIMITS`` 단일 소스의 **6관절 전체
limit**. 인자 없이 실행하면 모든 관절을 그 테이블 값으로 동기화한다(테이블=USD 보장).
즉 가동범위를 바꾸려면 **그 테이블만 고치고 이 스크립트를 재실행**하면 된다 — 테이블이
유일 정의(leader→sim remap·replay limit-check 와 공유). joint prim 은 이름으로 traverse
매칭(경로 무관). isaacsim 불필요(usd-core 만).

실행:
    .venv/bin/python scripts/assets/set_arm_joint_limits.py   # 테이블 6관절 일괄 적용
    # 옵션: --no-backup  --usd <경로>
    # 일회성 override: --set elbow_flex:-100:100 --set wrist_flex::105  (lower 생략=유지)
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from pxr import Usd, UsdPhysics

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))  # editable 설치 없이도 so101_contract import
from so101_contract.leader_calibration import (  # noqa: E402
    SO101_FOLLOWER_USD_JOINT_LIMITS,
)

ROBOT_USD = _REPO_ROOT / "assets" / "robots" / "so101_follower.usd"

# 기본 limit = leader_calibration 단일 소스의 6관절 USD limit 전체(degree, (lower, upper)).
# 인자 없이 실행하면 모든 관절 limit 을 이 테이블 값으로 USD 에 일괄 적용 → 테이블=USD 동기 보장.
# 값 중복 0(테이블이 유일 정의). 관절별 한 번에 바꾸려면 --set 으로 override.
DEFAULT_LIMITS: dict[str, tuple[float | None, float | None]] = dict(
    SO101_FOLLOWER_USD_JOINT_LIMITS
)


def _parse_set(items: list[str]) -> dict[str, tuple[float | None, float | None]]:
    """'joint:lower:upper' 문자열 리스트 → limit dict. lower/upper 빈칸=유지."""
    out: dict[str, tuple[float | None, float | None]] = {}
    for raw in items:
        parts = raw.split(":")
        if len(parts) != 3:
            raise SystemExit(f"--set 형식은 'joint:lower:upper' (예 elbow_flex::100): {raw!r}")
        name, lo, hi = parts
        out[name] = (float(lo) if lo else None, float(hi) if hi else None)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-backup", action="store_true", help="원본 .bak 백업 생략")
    parser.add_argument("--usd", default=str(ROBOT_USD), help="대상 로봇 USD 경로")
    parser.add_argument("--set", action="append", default=[], metavar="J:LO:UP",
                        help="관절 limit override (기본=elbow_flex/wrist_flex upper 100)")
    args = parser.parse_args()

    limits = _parse_set(args.set) if args.set else dict(DEFAULT_LIMITS)

    usd_path = Path(args.usd)
    if not usd_path.exists():
        raise FileNotFoundError(usd_path)

    if not args.no_backup:
        bak = usd_path.with_suffix(usd_path.suffix + ".preJointLimit.bak")
        if not bak.exists():
            shutil.copy2(usd_path, bak)
            print(f"[backup] {usd_path.name} -> {bak.name}")
        else:
            print(f"[backup] 이미 존재: {bak.name} (덮어쓰지 않음)")

    stage = Usd.Stage.Open(str(usd_path))
    # 이름 -> joint prim (RevoluteJoint).
    joints = {
        p.GetName(): p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)
    }

    changed = 0
    for name, (lo, hi) in limits.items():
        prim = joints.get(name)
        if prim is None:
            print(f"[MISS] RevoluteJoint 없음: {name}")
            continue
        lo_attr = prim.GetAttribute("physics:lowerLimit")
        hi_attr = prim.GetAttribute("physics:upperLimit")
        prev = (round(lo_attr.Get(), 2), round(hi_attr.Get(), 2))
        if lo is not None:
            lo_attr.Set(float(lo))
        if hi is not None:
            hi_attr.Set(float(hi))
        now = (round(lo_attr.Get(), 2), round(hi_attr.Get(), 2))
        print(f"[ok] {name}: limit {prev} -> {now} °")
        changed += 1

    if changed == 0:
        print("[done] 변경 없음")
        return
    stage.GetRootLayer().Save()
    print(f"[done] {changed} joint 저장: {usd_path}")

    verify = Usd.Stage.Open(str(usd_path))
    vjoints = {p.GetName(): p for p in verify.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)}
    for name in limits:
        p = vjoints.get(name)
        if p:
            print(f"[verify] {name}: "
                  f"[{p.GetAttribute('physics:lowerLimit').Get():.1f}, "
                  f"{p.GetAttribute('physics:upperLimit').Get():.1f}]°")


if __name__ == "__main__":
    main()
