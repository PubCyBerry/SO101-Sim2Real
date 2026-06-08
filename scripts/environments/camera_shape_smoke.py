"""TA.3 camera shape/FOV smoke: top/wrist/front RGB frame shape + intrinsics 점검.

검증 목표:
  - top_camera, wrist_camera, front_camera 각각 RGB 렌더 프레임 shape (num_envs, 480, 640, 3|4)
  - 각 카메라의 intrinsic matrix, world-frame 포즈를 JSON으로 출력

Usage:
    uv run python scripts/environments/camera_shape_smoke.py \\
        --task SimToReal-SO101-PickPen-v0 \\
        --num_envs 1 --device cuda:0
"""

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import json
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher

_EXPECTED_H = 480
_EXPECTED_W = 640
_EXPECTED_C = 3  # rgb (alpha 제외)
_CAMERA_NAMES = ["top_camera", "wrist_camera", "front_camera"]
_WARM_UP_STEPS = 5  # 렌더 파이프라인 안정화용

parser = argparse.ArgumentParser(description="TA.3 camera shape/FOV smoke")
parser.add_argument("--task", default="SimToReal-SO101-PickPen-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--save-dir", type=Path, default=None, help="Optional directory to save RGB PNG previews")
# --device, --headless, --enable_cameras 는 AppLauncher 에서 등록
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True  # TiledCamera 렌더 파이프라인 활성화

launcher = AppLauncher(args)
simulation_app = launcher.app

# SimApp 기동 후에만 import 가능
import gymnasium as gym       # noqa: E402
import torch                  # noqa: E402

import sim_to_real            # noqa: E402  # SimToReal-SO101-PickPen-v0 등록

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from sim_to_real.tasks.pick_pen.pick_pen_env_cfg import add_pick_pen_cameras  # noqa: E402


def _safe_tolist(tensor) -> list | None:
    try:
        return tensor.tolist()
    except Exception:
        return None


def _save_rgb_preview(rgb: torch.Tensor, path: Path) -> None:
    from PIL import Image

    array = rgb[0].detach().cpu().numpy()
    if array.shape[-1] == 4:
        array = array[..., :3]
    Image.fromarray(array).save(path)


def main() -> None:
    device: str = args.device
    env = None
    try:
        env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
        env_cfg.episode_length_s = 10.0
        # 기본 씬에는 카메라가 없다 — 여기서 주입해야 TiledCamera 가 등록된다.
        add_pick_pen_cameras(env_cfg.scene)
        env = gym.make(args.task, cfg=env_cfg)

        env.reset()

        # 렌더 파이프라인 워밍업
        zero_action = torch.zeros(args.num_envs, 6, device=device)
        for _ in range(_WARM_UP_STEPS):
            env.step(zero_action)

        passed = True
        cameras: dict[str, dict] = {}
        saved_images: dict[str, str] = {}
        if args.save_dir is not None:
            args.save_dir.mkdir(parents=True, exist_ok=True)

        for name in _CAMERA_NAMES:
            cam = env.unwrapped.scene[name]
            rgb: torch.Tensor = cam.data.output["rgb"]  # (N, H, W, C)
            shape = list(rgb.shape)

            # shape 검증: (N, 480, 640, 3 or 4)
            shape_ok = (
                len(shape) == 4
                and shape[0] == args.num_envs
                and shape[1] == _EXPECTED_H
                and shape[2] == _EXPECTED_W
                and shape[3] in (_EXPECTED_C, 4)  # alpha 포함 4ch 허용
            )
            if not shape_ok:
                passed = False
            if args.save_dir is not None and len(shape) == 4 and shape[0] > 0:
                image_path = args.save_dir / f"{name}.png"
                _save_rgb_preview(rgb, image_path)
                saved_images[name] = str(image_path)

            # intrinsic matrix — (N, 3, 3)
            intrinsics = None
            try:
                K = cam.data.intrinsic_matrices  # (N, 3, 3)
                intrinsics = _safe_tolist(K[0])
            except Exception:
                pass

            # world-frame 포즈
            pos_w = None
            try:
                pos_w = _safe_tolist(cam.data.pos_w[0])
            except Exception:
                pass

            rot_w = None
            try:
                # Isaac Lab convention: quat_w_world 또는 quat_w_ros
                if hasattr(cam.data, "quat_w_world"):
                    rot_w = _safe_tolist(cam.data.quat_w_world[0])
                elif hasattr(cam.data, "quat_w_ros"):
                    rot_w = _safe_tolist(cam.data.quat_w_ros[0])
            except Exception:
                pass

            # horizontal FOV: 2 * atan(W / (2 * fx)) (radians → degrees)
            fov_h_deg = None
            try:
                if intrinsics is not None:
                    import math
                    fx = intrinsics[0][0]
                    fov_h_deg = round(math.degrees(2.0 * math.atan(_EXPECTED_W / (2.0 * fx))), 2)
            except Exception:
                pass

            cameras[name] = {
                "rgb_shape": shape,
                "expected_shape": [args.num_envs, _EXPECTED_H, _EXPECTED_W, _EXPECTED_C],
                "shape_ok": shape_ok,
                "dtype": str(rgb.dtype),
                "intrinsics_3x3": intrinsics,
                "fov_horizontal_deg": fov_h_deg,
                "pos_w": pos_w,
                "rot_w_wxyz": rot_w,
            }

        env.close()

        result = {
            "task_id": "TA.3",
            "task": args.task,
            "status": "passed" if passed else "failed",
            "num_envs": args.num_envs,
            "warm_up_steps": _WARM_UP_STEPS,
            "cameras": cameras,
            "saved_images": saved_images,
            "notes": [
                "카메라는 add_pick_pen_cameras() 로 주입 — 기본 씬은 카메라 없이 --enable_cameras 불필요",
                "wrist_camera는 gripper 링크 자식 prim 으로 부착되어 gripper 움직임을 따라감(prim_path=.../Robot/gripper/WristCamera)",
                "top_camera는 world 절대 포즈로 고정",
                "포즈/FOV는 observation.images.{top,wrist,front} 데이터셋 프레임과 docs/pics 물리 배치 사진을 기준으로 1차 정합",
            ],
        }
        print(json.dumps(result, indent=2))
        sys.stdout.flush()
        if not passed:
            sys.exit(1)

    except Exception as exc:
        tb = traceback.format_exc()
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        result = {
            "task_id": "TA.3",
            "task": args.task,
            "status": "failed",
            "error": str(exc),
            "traceback": tb,
        }
        print(json.dumps(result, indent=2))
        sys.stdout.flush()
        sys.exit(1)

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
