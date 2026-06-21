#!/usr/bin/env python
"""무부하/cube payload 공통 canonical dynamics excitation plan 생성."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from so101_parity.dynamics import build_excitation_plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--home",
        nargs=6,
        type=float,
        default=[0.0, -1.3, 1.2, -0.3490658504, -1.5707963268, 40.0],
    )
    parser.add_argument("--condition", choices=("no_load", "cube_payload"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = build_excitation_plan(
        np.asarray(args.home, dtype=np.float64),
        condition=args.condition,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    {
                        "schema": "so101-dynamics-plan-v1",
                        "step": row.step,
                        "time_s": row.time_s,
                        "phase": row.phase,
                        "condition": row.condition,
                        "canonical_target": list(row.target),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
    duration_s = len(rows) / 30.0
    print(
        json.dumps(
            {
                "status": "passed",
                "condition": args.condition,
                "steps": len(rows),
                "duration_s": duration_s,
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
