import time
import mujoco
import mujoco.viewer
from so101_mujoco_utils import set_initial_pose, send_position_command, move_to_pose, hold_position
from so101_forward_kinematics import get_forward_kinematics
import numpy as np

# 이 폴더엔 model/ 이 없어 run_mujoco_simulation.py 와 동일하게 ref_repos scene.xml 사용
m = mujoco.MjModel.from_xml_path(r'C:\Users\taehunkim\Workspace\SO101-LeRobot-VLA\ref_repos\SO-ARM100\Simulation\SO101\scene.xml')
d = mujoco.MjData(m)

def show_cylinder(viewer, position, rotation, radius=0.0245, halfheight=0.05, rgba=[1, 0, 0, 1]):
    # Add a cylinder aligned with z-axis
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[0],
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,   # cylinder type
        size=[radius, halfheight, 0],                  # [radius, half-height, ignored]
        pos=position,                         # center position
        mat=rotation.flatten(),              # orientation matrix (identity = z-up)
        rgba=rgba                           # color
    )
    viewer.user_scn.ngeom = 1
    viewer.sync()
    return

test_configuration = {
    'shoulder_pan': -45.0,   # in radians for mujoco!
    'shoulder_lift': 45.0,
    'elbow_flex': -45.00,
    'wrist_flex': 90.0,
    'wrist_roll': 0.0,
    'gripper': 10
}

set_initial_pose(d, test_configuration)
send_position_command(d, test_configuration)

object_position, object_orientation = get_forward_kinematics(test_configuration)

with mujoco.viewer.launch_passive(m, d) as viewer:
    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()

    # Add a cylinder as a site for visualization
    show_cylinder(viewer, object_position, object_orientation)

    # Hold Starting Position for 2 seconds
    hold_position(m, d, viewer, 20.0)
