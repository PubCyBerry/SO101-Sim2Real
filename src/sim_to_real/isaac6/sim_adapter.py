"""Isaac Lab 3 ManagerBasedRLEnv ↔ canonical runtime adapter."""

from __future__ import annotations

import numpy as np
import torch

from so101_parity.calibration import CalibrationBundle
from so101_parity.runtime import CanonicalObservation


class Isaac6ParityAdapter:
    domain = "sim"

    def __init__(
        self,
        env,
        calibration: CalibrationBundle,
        *,
        render_on_capture: bool = False,
    ) -> None:
        self.env = env
        self.calibration = calibration
        self.scene = env.unwrapped.scene
        self.robot = self.scene["robot"]
        self.sim = env.unwrapped.sim
        self.device = env.unwrapped.device
        self.render_on_capture = bool(render_on_capture)
        if self.render_on_capture:
            # Camera construction marks RTX sensors globally active, causing
            # ManagerBasedRLEnv.step() to enter the render path every 30 Hz
            # tick. Chunk inference only consumes an RGB snapshot at request
            # boundaries, so render explicitly in capture() and keep physics
            # ticks free of Kit/RTX work in between.
            env.unwrapped.render_enabled = False
            self.sim.set_setting("/isaaclab/render/rtx_sensors", False)
        self._last_native = self.robot.data.joint_pos.torch[0].detach().clone()

    def read_state(self) -> np.ndarray:
        native = self.robot.data.joint_pos.torch[0].detach().cpu().numpy()
        return self.calibration.sim_to_canonical(native, clamp=True)

    def capture(self, state: np.ndarray | None = None) -> CanonicalObservation:
        if state is None:
            state = self.read_state()
        if self.render_on_capture:
            self.sim.render()
        images = {}
        for contract_name, scene_name in (
            ("top", "top_camera"),
            ("wrist", "wrist_camera"),
            ("front", "front_camera"),
        ):
            rgb_value = self.scene[scene_name].data.output["rgb"]
            rgb = (rgb_value.torch if hasattr(rgb_value, "torch") else rgb_value)[0]
            images[contract_name] = np.ascontiguousarray(
                rgb.detach().cpu().numpy(),
                dtype=np.uint8,
            )
        return CanonicalObservation(state=state, images=images)

    def canonical_to_native(self, target: np.ndarray) -> np.ndarray:
        return self.calibration.canonical_to_sim(target, clamp=True)

    def advance(self, native_command: np.ndarray) -> None:
        native = np.asarray(native_command, dtype=np.float32)
        if native.shape != (6,):
            raise ValueError(f"sim native command shape은 (6,)이어야 한다: {native.shape}")
        self._last_native = torch.as_tensor(
            native,
            device=self.device,
            dtype=torch.float32,
        )
        self.env.step(self._last_native.unsqueeze(0))

    def safe_stop(self, reason: str) -> None:
        del reason
        # Simulation은 torque 개념이 없으므로 마지막 absolute target을 유지한다.
        self.env.step(self._last_native.unsqueeze(0))
