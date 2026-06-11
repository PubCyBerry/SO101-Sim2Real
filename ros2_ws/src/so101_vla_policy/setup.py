import os
from glob import glob

from setuptools import find_packages, setup

package_name = "so101_vla_policy"

setup(
    name=package_name,
    version="0.1.0",
    # vendor/(vendored mini-lerobot)는 패키지로 설치하지 않는다 — 런타임 PYTHONPATH 로만 사용.
    packages=find_packages(exclude=["test", "vendor", "vendor.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="taehun.kim",
    maintainer_email="taehun.kim@konantech.com",
    description="SO-101 VLA inference ROS 2 node (sim VLA eval over policy-server gRPC)",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "vla_policy_node = so101_vla_policy.vla_policy_node:main",
            "joint_command_to_trajectory = so101_vla_policy.joint_command_to_trajectory:main",
        ],
    },
)
