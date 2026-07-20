"""SkillGen generated HDF5에 canonical 변환용 실제 joint/image를 추가 기록한다."""

from __future__ import annotations

import torch
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from so101_contract.feature_codec import SO101_JOINT_ORDER


def _joint_ids(env) -> list[int]:
    robot = env.scene["robot"]
    ids, resolved = robot.find_joints(list(SO101_JOINT_ORDER), preserve_order=True)
    if resolved != list(SO101_JOINT_ORDER):
        raise RuntimeError(f"canonical SO-101 joint order mismatch: {resolved}")
    return ids


class PostStepSO101JointDataRecorder(RecorderTerm):
    """물리가 실행된 뒤 실제 state와 slew-limited applied target을 함께 기록한다."""

    def record_post_step(self):
        env = self._env
        robot = env.scene["robot"]
        ids = _joint_ids(env)
        arm_term = env.action_manager.get_term("arm_action")
        gripper_term = env.action_manager.get_term("gripper_action")
        target = torch.cat(
            [arm_term.applied_joint_targets, gripper_term.applied_joint_targets], dim=-1
        )
        return "joint_data", {"state": robot.data.joint_pos[:, ids].clone(), "target": target.clone()}


class PostStepCameraImagesRecorder(RecorderTerm):
    """top/wrist/front uint8 RGB를 generated action과 같은 frame에 기록한다."""

    def record_post_step(self):
        images = self._env.obs_buf.get("images")
        if not isinstance(images, dict):
            raise RuntimeError("camera image observation group is missing; pass --enable_cameras")
        return "images", {name: image.clone() for name, image in images.items()}


class PostStepTimeRecorder(RecorderTerm):
    """30 FPS monotonic timestamp와 frame index를 기록한다."""

    def record_post_step(self):
        frame = self._env.episode_length_buf.clone()
        return "timing", {"frame_index": frame, "timestamp": frame.to(torch.float64) / 30.0}


@configclass
class PostStepSO101JointDataRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PostStepSO101JointDataRecorder


@configclass
class PostStepCameraImagesRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PostStepCameraImagesRecorder


@configclass
class PostStepTimeRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PostStepTimeRecorder


@configclass
class SO101SkillGenJointRecorderManagerCfg(ActionStateRecorderManagerCfg):
    record_post_step_so101_joint_data = PostStepSO101JointDataRecorderCfg()
    record_post_step_timing = PostStepTimeRecorderCfg()


@configclass
class SO101SkillGenRecorderManagerCfg(SO101SkillGenJointRecorderManagerCfg):
    record_post_step_camera_images = PostStepCameraImagesRecorderCfg()
