"""Pen Pick-and-Place task configuration."""

from __future__ import annotations

from typing import Any

import torch
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from leisaac.utils.domain_randomization import (
    domain_randomization,
    randomize_camera_uniform,
)
from leisaac.utils.general_assets import parse_usd_and_create_subassets

from sim_to_real.assets.scenes.pen_desk import PEN_DESK_CFG, PEN_DESK_USD_PATH
from leisaac.tasks.template import (
    SingleArmObservationsCfg,
    SingleArmTaskEnvCfg,
    SingleArmTaskSceneCfg,
    SingleArmTerminationsCfg,
)
from sim_to_real.utils.constant import PEN_CUP_NAME, PEN_NAMES
from sim_to_real.utils.domain_randomization import (
    randomize_object_in_ellipse,
    randomize_object_on_arc,
)

from . import mdp


# World-frame (x, y) of the pen cup at scene authoring time. The cup is a
# rigid body so this is only the initial pose; pen_in_cup checks against this
# seed location for now. PEN_CUP_LOCAL=(0, 0.40) + SCENE_OFFSET=(2.2, -0.57)
# = (2.2, -0.17). Cup is the apex of the forward-facing ±30° arc, sitting
# ≈ 0.44 m forward of the robot (SO-101 reach perimeter).
PEN_CUP_CENTER_XY: tuple[float, float] = (2.2, -0.17)



@configclass
class PickPenSceneCfg(SingleArmTaskSceneCfg):
    """Scene configuration for the pick orange task."""

    scene: AssetBaseCfg = PEN_DESK_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")



@configclass
class PickPenObservationsCfg(SingleArmObservationsCfg):
    """Adds per-pen grasped / in-cup signals to the shared single-arm obs."""

    @configclass
    class SubtaskCfg(ObsGroup):
        pick_white = ObsTerm(func=mdp.pen_grasped, params={"object_cfg": SceneEntityCfg("PenWhite")})
        place_white = ObsTerm(
            func=mdp.pen_in_cup,
            params={"object_cfg": SceneEntityCfg("PenWhite"), "cup_center_xy": PEN_CUP_CENTER_XY},
        )
        pick_gray = ObsTerm(func=mdp.pen_grasped, params={"object_cfg": SceneEntityCfg("PenGray")})
        place_gray = ObsTerm(
            func=mdp.pen_in_cup,
            params={"object_cfg": SceneEntityCfg("PenGray"), "cup_center_xy": PEN_CUP_CENTER_XY},
        )
        pick_black = ObsTerm(func=mdp.pen_grasped, params={"object_cfg": SceneEntityCfg("PenBlack")})
        place_black = ObsTerm(
            func=mdp.pen_in_cup,
            params={"object_cfg": SceneEntityCfg("PenBlack"), "cup_center_xy": PEN_CUP_CENTER_XY},
        )
        pick_blue = ObsTerm(func=mdp.pen_grasped, params={"object_cfg": SceneEntityCfg("PenBlue")})
        place_blue = ObsTerm(
            func=mdp.pen_in_cup,
            params={"object_cfg": SceneEntityCfg("PenBlue"), "cup_center_xy": PEN_CUP_CENTER_XY},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class PickPenTerminationsCfg(SingleArmTerminationsCfg):
    success = DoneTerm(
        func=mdp.task_done,
        params={
            "pens_cfg": [SceneEntityCfg(name) for name in PEN_NAMES],
            "cup_center_xy": PEN_CUP_CENTER_XY,
        },
    )


@configclass
class PickPenEnvCfg(SingleArmTaskEnvCfg):
    """Pen Pick-and-Place environment config (manager-based RL)."""

    scene: PickPenSceneCfg = PickPenSceneCfg(num_envs=1, env_spacing=1.5)
    observations: PickPenObservationsCfg = PickPenObservationsCfg()
    terminations: PickPenTerminationsCfg = PickPenTerminationsCfg()

    task_description: str = "Pick the scattered pens off the desk and drop them into the pen cup."

    def __post_init__(self) -> None:
        super().__post_init__()
        parse_usd_and_create_subassets(PEN_DESK_USD_PATH, self, specific_name_list=[*PEN_NAMES, PEN_CUP_NAME])

        domain_randomization(
            self,
            random_options=[
                # 펜은 docs/pics/펜통_펜_배치_3.jpg 의 "초록 타원" 영역 안에서 흩어진다.
                # 각 펜의 jitter 는 ±5 cm × ±2 cm (가로로 약간 긴) 작은 타원으로 제한해
                # 4 개 default 가 그린 영역 4 개 구역에 분산된 상태에서 펜끼리 capsule
                # collider 가 겹치지 않게 한다. yaw 는 ±10° — 4 개 펜의 author yaw 차이
                # (각각 25°/-30°/60°/-10°) 를 망가뜨리지 않을 정도만 흔든다.
                *[
                    randomize_object_in_ellipse(
                        name,
                        x_radius=0.05,
                        y_radius=0.02,
                        yaw_range_deg=(-10.0, 10.0),
                    )
                    for name in PEN_NAMES
                ],
                # 펜컵은 docs/pics/펜통_펜_배치_3.jpg 의 "주황 호" 를 따라 sampling.
                # default 위치 scene-local (0, 0.40) 이 호의 정점, robot scene-local
                # y=-0.04 에서 0.44 m 떨어진 SO-101 reach 가장자리. ±30° 회전 시
                # 양 끝 (±0.22, 0.34) — 매트 안 + reach 한계 둘 다 만족. 펜 영역
                # (y ≤ 0.28) 과 펜컵 영역 (y ≥ 0.34) 이 y 방향으로 분리되어
                # 펜이 펜컵 안에 spawn 되는 케이스가 원천 차단된다.
                randomize_object_on_arc(
                    PEN_CUP_NAME,
                    radius=0.44,
                    angle_range_deg=(-20.0, 20.0),
                ),
                randomize_camera_uniform(
                    "front",
                    pose_range={
                        "x": (-0.025, 0.025),
                        "y": (-0.025, 0.025),
                        "z": (-0.025, 0.025),
                        "roll": (-2.5 * torch.pi / 180, 2.5 * torch.pi / 180),
                        "pitch": (-2.5 * torch.pi / 180, 2.5 * torch.pi / 180),
                        "yaw": (-2.5 * torch.pi / 180, 2.5 * torch.pi / 180),
                    },
                    convention="ros",
                ),
            ],
        )

    # ------------------------------------------------------------------
    # Recorder export helpers — keep the old API used by the converter.
    # ------------------------------------------------------------------

    def use_teleop_device(self, teleop_device: str) -> None:  # noqa: D401 - inherits docstring
        super().use_teleop_device(teleop_device)

    def preprocess_device_action(self, action: dict[str, Any], teleop_device) -> torch.Tensor:
        return super().preprocess_device_action(action, teleop_device)

    def build_lerobot_frame(self, episode_data: EpisodeData, dataset_cfg: LeRobotDatasetCfg) -> dict:
        return super().build_lerobot_frame(episode_data, dataset_cfg)


# Backwards-compatible alias for code paths that still import the old name.
SO101PickPenTaskEnvCfg = PickPenEnvCfg
