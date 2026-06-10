#!/usr/bin/env python3
"""이미지에서 보라 그리퍼 검출 → tip(최상단 극점)+centroid 픽셀. usage: detect_gripper.py [img.jpg]"""
import cv2, numpy as np, sys
f = sys.argv[1] if len(sys.argv) > 1 else '/mnt/c/Users/taehunkim/AppData/Local/Temp/fd_top.jpg'
img = cv2.imread(f)
if img is None:
    print('cannot read', f); raise SystemExit
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, (115, 60, 40), (165, 255, 255))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not cnts:
    print('no gripper'); raise SystemExit
c = max(cnts, key=cv2.contourArea)
area = cv2.contourArea(c)
pts = c.reshape(-1, 2)
tip = pts[np.argmin(pts[:, 1])]          # 화면 위쪽(=전방) 극점
M = cv2.moments(c); ccx, ccy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
out = img.copy()
cv2.drawContours(out, [c], -1, (0, 255, 0), 2)
cv2.circle(out, tuple(tip), 6, (0, 0, 255), -1)
cv2.circle(out, (ccx, ccy), 6, (255, 0, 0), -1)
cv2.imwrite(f.replace('.jpg', '_grip.jpg'), out)
print('area=%d  tip=(%d,%d)  centroid=(%d,%d)' % (area, tip[0], tip[1], ccx, ccy))
