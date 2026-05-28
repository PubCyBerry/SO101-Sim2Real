"""로봇 USD의 재료 색상 목록을 출력하는 headless 스크립트."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(["--headless"])
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, Gf  # noqa: E402

USD_PATH = "assets/robots/so101_follower.usd"

stage = Usd.Stage.Open(USD_PATH)
print(f"\n{'='*60}")
print(f"재료 목록: {USD_PATH}")
print(f"{'='*60}")
for prim in stage.Traverse():
    if prim.GetTypeName() == "Shader":
        dc = prim.GetAttribute("inputs:diffuseColor")
        roughness = prim.GetAttribute("inputs:roughness")
        metallic = prim.GetAttribute("inputs:metallic")
        if dc.IsValid() and dc.Get() is not None:
            print(
                f"  {prim.GetPath()}\n"
                f"    diffuseColor={tuple(round(v,3) for v in dc.Get())}"
                f"  roughness={roughness.Get():.2f}"
                f"  metallic={metallic.Get():.2f}"
            )

simulation_app.close()
