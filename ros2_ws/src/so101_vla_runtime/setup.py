import os
from glob import glob

from setuptools import find_packages, setup

package_name = "so101_vla_runtime"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="taehun.kim",
    maintainer_email="taehun.kim@konantech.com",
    description="Deterministic canonical SO-101 VLA ROS 2 runtime",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "vla_server = so101_vla_runtime.server_node:main",
            "mock_client = so101_vla_runtime.mock_client:main",
            "integration_probe = so101_vla_runtime.integration_probe:main",
        ],
    },
)
