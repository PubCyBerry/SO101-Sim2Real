#!/usr/bin/env python3
"""크기 DR sweep 결과 JSON 들 → 크기 × yaw 조건 성공률 표 (stdlib 만)."""
import json
import pathlib
import re
import sys

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/cube_size_dr_sweep")
rows = []
for p in sorted(OUT.glob("sweep_*mm_*.json")):
    m = re.match(r"sweep_(\d+)mm_(.+)\.json", p.name)
    if not m:
        continue
    mm, tag = int(m.group(1)), m.group(2)
    d = json.loads(p.read_text(encoding="utf-8"))
    cells = d["cells"]
    n = sum(c["n"] for c in cells)
    placed = sum(c["n_placed"] for c in cells)
    planned = sum(c["n_planned"] for c in cells)
    log = p.with_suffix(".log")
    timeout = log.exists() and "planner TIMEOUT" in log.read_text(encoding="utf-8", errors="ignore")
    fails = [(c["x"], c["y"], c["kind"], c["n"] - c["n_placed"]) for c in cells if c["n_placed"] < c["n"]]
    rows.append((mm, tag, placed, planned, n, len(cells), timeout, fails))

print(f"{'cube':>6} {'yaw':>9} {'placed':>12} {'planned':>12} {'cells':>6}  note")
for mm, tag, placed, planned, n, ncell, timeout, fails in sorted(rows):
    pct = 100.0 * placed / max(n, 1)
    note = "⚠ planner TIMEOUT 구간 있음" if timeout else ("" if placed == n else f"fail cells={len(fails)}")
    print(f"{mm:>4}mm {tag:>9} {placed:>5}/{n:<6}({pct:5.1f}%) {planned:>5}/{n:<6}      {ncell:>4}  {note}")
    for x, y, kind, k in fails[:10]:
        print(f"         └ fail ({x:+.3f},{y:+.3f}) {kind} ×{k}")
if not rows:
    print("(결과 없음)")
