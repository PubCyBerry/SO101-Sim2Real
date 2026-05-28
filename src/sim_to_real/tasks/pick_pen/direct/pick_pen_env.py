"""Direct environment for the pen Pick-and-Place task (mirrors leisaac pick_orange direct)."""

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from leisaac.tasks.template import SingleArmTaskDirectEnv, SingleArmTaskDirectEnvCfg
from leisaac.utils.domain_randomization import (
    domain_randomization,
    randomize_camera_uniform,
    randomize_object_uniform,
)
from leisaac.utils.general_assets import parse_usd_and_create_subassets

from sim_to_real.assets.scenes import PEN_DESK_USD_PATH
from sim_to_real.utils.constant import PEN_NAMES

from .. import mdp
from ..pick_pen_env_cfg import PEN_CUP_CENTER_XY, PickPenSceneCfg


@configclass
class PickPenEnvCfg(SingleArmTaskDirectEnvCfg):
    """Direct env configuration for the pen Pick-and-Place task."""

    scene: PickPenSceneCfg = PickPenSceneCfg(num_envs=1, env_spacing=1.5)

    task_description: str = "Pick the scattered pens off the desk and drop them into the pen cup."

    def __post_init__(self) -> None:
        super().__post_init__()

        parse_usd_and_create_subassets(PEN_DESK_USD_PATH, self, specific_name_list=PEN_NAMES)

        domain_randomization(
            self,
            random_options=[
                *[
                    randomize_object_uniform(
                        name, pose_range={"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (0.0, 0.0)}
                    )
                    for name in PEN_NAMES
                ],
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


class PickPenEnv(SingleArmTaskDirectEnv):
    """Direct env for the pen Pick-and-Place task."""

    cfg: PickPenEnvCfg

    def _get_observations(self) -> dict:
        obs = super()._get_observations()
        obs["subtask_terms"] = {
            "pick_white": mdp.pen_grasped(self, object_cfg=SceneEntityCfg("PenWhite")),
            "place_white": mdp.pen_in_cup(self, object_cfg=SceneEntityCfg("PenWhite"), cup_center_xy=PEN_CUP_CENTER_XY),
            "pick_gray": mdp.pen_grasped(self, object_cfg=SceneEntityCfg("PenGray")),
            "place_gray": mdp.pen_in_cup(self, object_cfg=SceneEntityCfg("PenGray"), cup_center_xy=PEN_CUP_CENTER_XY),
            "pick_black": mdp.pen_grasped(self, object_cfg=SceneEntityCfg("PenBlack")),
            "place_black": mdp.pen_in_cup(self, object_cfg=SceneEntityCfg("PenBlack"), cup_center_xy=PEN_CUP_CENTER_XY),
            "pick_blue": mdp.pen_grasped(self, object_cfg=SceneEntityCfg("PenBlue")),
            "place_blue": mdp.pen_in_cup(self, object_cfg=SceneEntityCfg("PenBlue"), cup_center_xy=PEN_CUP_CENTER_XY),
        }
        return obs

    def _check_success(self) -> torch.Tensor:
        return mdp.task_done(
            env=self,
            pens_cfg=[SceneEntityCfg(name) for name in PEN_NAMES],
            cup_center_xy=PEN_CUP_CENTER_XY,
        )
