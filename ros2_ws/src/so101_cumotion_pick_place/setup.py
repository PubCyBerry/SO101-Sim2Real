import os
from glob import glob

from setuptools import find_packages, setup

package_name = "so101_cumotion_pick_place"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
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
    description="SO-101 cuMotion + ROS 2 pick-and-place state machine (PATH E)",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "pick_place_sm = so101_cumotion_pick_place.pick_place_sm:main",
        ],
    },
)
