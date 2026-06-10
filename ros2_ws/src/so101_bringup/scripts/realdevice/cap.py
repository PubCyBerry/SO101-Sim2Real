#!/usr/bin/env python3
"""3캠 단발 캡처 → Windows %TEMP%/{tag}_{name}.jpg. (env.sh source 한 셸에서)"""
import rclpy, time, cv2, sys
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
tag = sys.argv[1] if len(sys.argv) > 1 else 'x'
OUT = '/mnt/c/Users/taehunkim/AppData/Local/Temp'
rclpy.init(); n = rclpy.create_node('cap'); b = CvBridge(); s = {}
def mk(nm):
    def cb(m):
        if nm not in s:
            s[nm] = 1
            cv2.imwrite('%s/%s_%s.jpg' % (OUT, tag, nm), b.imgmsg_to_cv2(m, 'bgr8'))
    return cb
for nm in ['top', 'wrist', 'front']:
    n.create_subscription(Image, '/camera/' + nm + '/image_raw', mk(nm), 10)
t0 = time.time()
while time.time() - t0 < 8 and len(s) < 3:
    rclpy.spin_once(n, timeout_sec=0.1)
print('captured', sorted(s.keys()))
