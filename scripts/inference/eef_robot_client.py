#!/usr/bin/env python3
"""Windows SO-101용 LeRobot async EEF-relative policy client.

Stock ``RobotClient``의 gRPC/camera/control loop를 재사용하면서 경계 I/O만 바꾼다.

- real follower joint observation → calibration → canonical FK → absolute EEF 10D
- server absolute EEF chunk → bounded sequential IK → real follower joint chunk
- overlap은 EEF vector를 평균하지 않고 IK 후 ``latest_only``로 병합
- IK chunk 한 step이라도 실패하면 chunk 전체 폐기, hold/replan
- offline sweep 승인 전에는 ``--real_hardware_ik_validated=true`` 없이는 command 금지
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from queue import Queue
import threading
import time
from typing import Any, Callable

import draccus
import numpy as np
import torch

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.helpers import (
    RemotePolicyConfig,
    TimedAction,
    map_robot_keys_to_lerobot_features,
)
from lerobot.async_inference.robot_client import RobotClient
from lerobot.utils.import_utils import register_third_party_plugins

from so101_contract.eef_ik import IKConfig
from so101_contract.eef_policy_io import (
    EEFPlatformAdapterConfig,
    SO101EEFPolicyIO,
)
from so101_contract.feature_codec import JOINT_FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class EEFRobotClientConfig(RobotClientConfig):
    """Stock client config + 명시적인 real IK deployment gate."""

    eef_urdf_path: str = field(
        default=str(ROOT / "assets" / "robots" / "urdf" / "so_arm101.urdf"),
        metadata={"help": "SO-101 canonical URDF path"},
    )
    eef_robot_yaml_path: str = field(
        default=str(ROOT / "assets" / "robots" / "so101.yml"),
        metadata={"help": "tcp_grasp transform을 포함한 robot YAML path"},
    )
    real_hardware_ik_validated: bool = field(
        default=False,
        metadata={"help": "offline/physical sweep 승인 후에만 true"},
    )
    ik_failure_limit: int = field(default=3, metadata={"help": "연속 IK 실패 episode 중단 한계"})
    ik_position_weight: float = 1.0
    ik_orientation_weight: float = 0.15
    ik_damping: float = 1e-3
    ik_max_iterations: int = 80
    ik_position_tolerance_m: float = 5e-4
    ik_orientation_tolerance_rad: float = 1e-2
    arm_target_max_velocity: float = 5.0
    gripper_target_max_velocity: float = 2.5
    eef_metrics_log: str = field(
        default="",
        metadata={"help": "motor-off target와 measured EEF residual JSONL path"},
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.aggregate_fn_name != "latest_only":
            raise ValueError(
                "EEF-relative client requires --aggregate_fn_name=latest_only; "
                "Rot6D/EEF vector averaging is forbidden"
            )
        if self.ik_failure_limit <= 0:
            raise ValueError("ik_failure_limit must be positive")


class EEFRobotClient(RobotClient):
    """schema v2 representation-aware real client(4 mode 공통).

    EEF mode는 FK/IK adapter를 쓰고, joint mode는 canonical joint feature 경계만 쓴다.
    두 경우 모두 router가 platform 명령을 만들고 경계 변환은 정확히 1회다.
    """

    def __init__(self, config: EEFRobotClientConfig) -> None:
        # schema v2 전용 startup plan. v1 deployment validator는 migration 전용이며
        # runtime fallback으로 호출하지 않는다. kinematics hash는 v2 manifest의
        # kinematics 절과 실제 URDF/robot YAML을 직접 대조해 검증한다.
        # representation env는 **선택적 assertion**이다(비어 있으면 manifest 수용).
        self._startup_plan = plan_inference_startup(
            config.pretrained_name_or_path,
            mode=os.getenv("ACTION_REPRESENTATION_MODE") or None,
            pose_format=os.getenv("ACTION_REPRESENTATION_POSE_FORMAT") or None,
            policy_type=config.policy_type,
            revision=os.getenv("POLICY_REVISION") or None,
            local_files_only=os.getenv("HF_HUB_OFFLINE", "0") == "1",
            urdf_path=config.eef_urdf_path,
            robot_yaml_path=config.eef_robot_yaml_path,
        )
        self._checkpoint_contract = self._startup_plan.contract
        self._requires_ik = self._startup_plan.requires_ik
        super().__init__(config)
        self.config = config
        self._joint_state_lock = threading.Lock()
        self._latest_real_joint_state: np.ndarray | None = None
        self._ik_consecutive_failures = 0
        self._eef_targets_by_timestep: dict[int, np.ndarray] = {}
        self._last_executed_eef_target: np.ndarray | None = None
        self._eef_metrics = {
            "chunks_received": 0,
            "dry_run_chunks": 0,
            "ik_failures": 0,
            "replans": 0,
            "aborts": 0,
            "commands_sent": 0,
            "residual_samples": 0,
            "position_residual_max_m": 0.0,
            "orientation_residual_max_rad": 0.0,
        }
        self._eef_metrics_fh = None
        if config.eef_metrics_log:
            metrics_path = Path(config.eef_metrics_log)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            self._eef_metrics_fh = metrics_path.open("w", buffering=1, encoding="utf-8")
        if tuple(self.robot.action_features) != tuple(JOINT_FEATURE_NAMES):
            raise ValueError(
                "real robot action feature order mismatch: "
                f"{tuple(self.robot.action_features)} != {tuple(JOINT_FEATURE_NAMES)}"
            )

        adapter_config = EEFPlatformAdapterConfig(
            max_arm_step_rad=config.arm_target_max_velocity / config.fps,
            max_gripper_step_rad=config.gripper_target_max_velocity / config.fps,
            real_hardware_ik_validated=config.real_hardware_ik_validated,
        )
        # joint mode는 URDF/YAML을 읽지 않고 IK adapter도 만들지 않는다.
        self._eef_adapter = (
            SO101EEFPolicyIO.from_files(
                config.eef_urdf_path,
                config.eef_robot_yaml_path,
                ik_config=IKConfig(
                    position_weight=config.ik_position_weight,
                    orientation_weight=config.ik_orientation_weight,
                    damping=config.ik_damping,
                    max_iterations=config.ik_max_iterations,
                    position_tolerance_m=config.ik_position_tolerance_m,
                    orientation_tolerance_rad=config.ik_orientation_tolerance_rad,
                ),
                config=adapter_config,
            )
            if self._requires_ik
            else None
        )
        # EEF route만 platform adapter의 sequential IK를 정확히 1회 호출한다.
        self._router = build_router(
            self._startup_plan,
            policy_io=self._eef_adapter,
            adapter_config=adapter_config,
        )

        # Camera schema는 stock robot mapping을 유지하고 state만 canonical EEF 10D로 교체한다.
        lerobot_features = map_robot_keys_to_lerobot_features(self.robot)
        lerobot_features = dict(lerobot_features)
        # manifest가 진실이다. pose format(10D/8D/7D)에 따라 dim/names가 달라진다.
        lerobot_features["observation.state"] = self._startup_plan.state_feature()
        self.policy_config = RemotePolicyConfig(
            config.policy_type,
            config.pretrained_name_or_path,
            lerobot_features,
            config.actions_per_chunk,
            config.policy_device,
        )
        self.logger.info(f"[action-representation] {self._checkpoint_contract.summary()}")
        # mode-aware startup metadata. joint manifest는 kinematics=None이 정상이므로
        # 절대 index하지 않는다(공용 helper가 담당).
        self.logger.info(f"[startup] {format_startup_log(self._startup_plan)}")
        if self._requires_ik and not config.real_hardware_ik_validated:
            self.logger.warning(
                "real_hardware_ik_validated=false: observation/inference는 수행하지만 "
                "IK target은 dry-run log로만 남기고 실기기 command queue에는 넣지 않습니다."
            )

    def _write_metric(self, event: str, **values: Any) -> None:
        if self._eef_metrics_fh is None:
            return
        payload = {
            "event": event,
            "time": time.time(),
            **self._eef_metrics,
            **values,
        }
        import json

        self._eef_metrics_fh.write(json.dumps(payload) + "\n")

    @staticmethod
    def _joint_values(raw_observation: dict[str, Any]) -> np.ndarray:
        try:
            values = np.asarray(
                [raw_observation[name] for name in JOINT_FEATURE_NAMES],
                dtype=np.float32,
            )
        except KeyError as exc:
            raise KeyError(
                f"real observation missing canonical follower joint feature: {exc}"
            ) from exc
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise ValueError(f"real follower observation must be finite (6,), got {values}")
        return values

    def control_loop_observation(self, task: str, verbose: bool = False):
        """Stock sender가 읽을 raw observation을 잠시 EEF feature schema로 투영."""
        original_get_observation = self.robot.get_observation

        def get_eef_observation() -> dict[str, Any]:
            raw = dict(original_get_observation())
            real_joint_state = self._joint_values(raw)
            with self._joint_state_lock:
                self._latest_real_joint_state = real_joint_state.copy()
            if self._requires_ik:
                # EEF: FK → manifest pose format(10D/8D/7D)
                eef_state = observation_to_manifest_format(
                    self._startup_plan,
                    self._eef_adapter.observation_from_real(real_joint_state),
                )
            else:
                # joint: 실 follower → canonical arm radian + gripper feature(경계 변환 1회)
                eef_state = real_follower_state_to_feature(real_joint_state)
            if self._requires_ik and self._last_executed_eef_target is not None:
                target = self._last_executed_eef_target
                # pose format 중립 residual(rot6d/wxyz/rpy 공통, gripper 미포함)
                position_residual, orientation_residual = eef_pose_residual(
                    self._startup_plan,
                    eef_state,
                    target,
                )
                self._eef_metrics["residual_samples"] += 1
                self._eef_metrics["position_residual_max_m"] = max(
                    self._eef_metrics["position_residual_max_m"],
                    position_residual,
                )
                self._eef_metrics["orientation_residual_max_rad"] = max(
                    self._eef_metrics["orientation_residual_max_rad"],
                    orientation_residual,
                )
                self._write_metric(
                    "measured_residual",
                    position_residual_m=position_residual,
                    orientation_residual_rad=orientation_residual,
                    measured_eef=eef_state.tolist(),
                    target_eef=target.tolist(),
                )
            for name in JOINT_FEATURE_NAMES:
                raw.pop(name, None)
            # observation feature 이름/차원도 manifest에서 온다(rot6d 10D 고정 아님).
            raw.update(
                {
                    name: float(eef_state[index])
                    for index, name in enumerate(self._startup_plan.state_names)
                }
            )
            return raw

        self.robot.get_observation = get_eef_observation
        try:
            return super().control_loop_observation(task, verbose)
        finally:
            self.robot.get_observation = original_get_observation

    def _discard_failed_chunk(self, reason: str) -> None:
        self._eef_metrics["ik_failures"] += 1
        self._eef_metrics["replans"] += 1
        self._ik_consecutive_failures += 1
        with self.action_queue_lock:
            self.action_queue = Queue()
        self.logger.error(
            "EEF IK chunk 폐기·hold·replan: "
            f"reason={reason}, consecutive="
            f"{self._ik_consecutive_failures}/{self.config.ik_failure_limit}"
        )
        if self._ik_consecutive_failures >= self.config.ik_failure_limit:
            self._eef_metrics["aborts"] += 1
            self.logger.error("EEF IK 연속 실패 한계 초과: real episode 중단")
            self.shutdown_event.set()
        self._write_metric("ik_failure", reason=reason)

    def _aggregate_action_queues(
        self,
        incoming_actions: list[TimedAction],
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """수신 EEF chunk를 먼저 joint chunk로 바꾼 뒤 latest_only 병합."""
        del aggregate_fn
        if not incoming_actions:
            return
        with self._joint_state_lock:
            current = (
                None
                if self._latest_real_joint_state is None
                else self._latest_real_joint_state.copy()
            )
        if current is None:
            self._discard_failed_chunk("no_measured_real_joint_state")
            return

        ordered = sorted(incoming_actions, key=lambda action: int(action.get_timestep()))
        with self.latest_action_lock:
            latest_action = self.latest_action
        ordered = [action for action in ordered if action.get_timestep() > latest_action]
        if not ordered:
            return
        eef_chunk = np.stack(
            [
                action.get_action().detach().cpu().numpy().astype(np.float32)
                for action in ordered
            ]
        )
        self._eef_metrics["chunks_received"] += 1
        # dry-run gate는 **IK를 쓰는 EEF mode 전용**이다. joint mode는 IK가 없으므로 막지 않는다.
        if self._requires_ik and not self.config.real_hardware_ik_validated:
            conversion = self._router.route(eef_chunk, current, platform="real_dry_run")
            if not conversion.success or conversion.platform_actions is None:
                self._discard_failed_chunk(f"dry_run:{conversion.reason}")
                return
            self._eef_metrics["dry_run_chunks"] += 1
            self._write_metric(
                "dry_run_chunk",
                timesteps=[int(action.get_timestep()) for action in ordered],
                absolute_eef_targets=eef_chunk.tolist(),
                real_joint_targets=conversion.platform_actions.tolist(),
            )
            with self.action_queue_lock:
                self.action_queue = Queue()
            return

        conversion = self._router.route(eef_chunk, current, platform="real")
        if not conversion.success or conversion.platform_actions is None:
            self._discard_failed_chunk(conversion.reason)
            return

        self._ik_consecutive_failures = 0
        self._eef_targets_by_timestep.update(
            {
                int(timed_action.get_timestep()): np.asarray(eef_target, dtype=np.float32)
                for timed_action, eef_target in zip(ordered, eef_chunk, strict=True)
            }
        )
        joint_actions = [
            TimedAction(
                timestamp=timed_action.get_timestamp(),
                timestep=timed_action.get_timestep(),
                action=torch.as_tensor(joints, dtype=torch.float32),
            )
            for timed_action, joints in zip(ordered, conversion.platform_actions, strict=True)
        ]
        # EEF에서 평균하지 않는다. IK 뒤 joint target도 v1은 새 chunk가 overlap을 교체한다.
        super()._aggregate_action_queues(joint_actions, aggregate_fn=lambda _old, new: new)

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any]:
        performed = super().control_loop_action(verbose)
        with self.latest_action_lock:
            timestep = int(self.latest_action)
        target = self._eef_targets_by_timestep.pop(timestep, None)
        if target is not None:
            self._last_executed_eef_target = target
        self._eef_metrics["commands_sent"] += 1
        self._write_metric("command_sent", timestep=timestep)
        return performed

    def stop(self) -> None:
        self._write_metric("final")
        if self._eef_metrics_fh is not None:
            self._eef_metrics_fh.close()
            self._eef_metrics_fh = None
        super().stop()


@draccus.wrap()
def eef_async_client(cfg: EEFRobotClientConfig) -> None:
    client = EEFRobotClient(cfg)
    if not client.start():
        return
    action_receiver_thread = threading.Thread(
        target=client.receive_actions,
        daemon=True,
    )
    action_receiver_thread.start()
    try:
        client.control_loop(task=cfg.task)
    finally:
        client.stop()
        action_receiver_thread.join()


if __name__ == "__main__":
    register_third_party_plugins()
    eef_async_client()
