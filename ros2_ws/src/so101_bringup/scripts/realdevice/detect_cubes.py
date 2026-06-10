#!/usr/bin/env python3
"""top 이미지에서 회색 큐브 검출 → 픽셀 중심 + 주석 이미지. usage: detect_cubes.py [img.jpg]"""
import cv2, numpy as np, sys
f = sys.argv[1] if len(sys.argv) > 1 else '/mnt/c/Users/taehunkim/AppData/Local/Temp/fd_top.jpg'
img = cv2.imread(f)
if img is None:
    print('cannot read', f); raise SystemExit
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hh, ss, vv = cv2.split(hsv)
# 회색 큐브: 낮은 채도 + 중간~높은 밝기 (그릇/깃털/그리퍼/나무 배제)
mask = ((ss < 55) & (vv > 105) & (vv < 215)).astype(np.uint8) * 255
mask[:80, :] = 0      # 상단 트레이/데스크
mask[440:, :] = 0     # 하단 데스크
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
out = img.copy(); cubes = []
for c in cnts:
    a = cv2.contourArea(c)
    if a < 400 or a > 4000:
        continue
    x, y, w, h = cv2.boundingRect(c)
    ar = w / float(h)
    if ar < 0.55 or ar > 1.8:
        continue
    cx, cy = x + w // 2, y + h // 2
    cubes.append((cx, cy, int(a)))
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.circle(out, (cx, cy), 3, (0, 255, 0), -1)
    cv2.putText(out, '%d,%d' % (cx, cy), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
cv2.imwrite(f.replace('.jpg', '_det.jpg'), out)
print('detected %d cubes:' % len(cubes), sorted(cubes))
