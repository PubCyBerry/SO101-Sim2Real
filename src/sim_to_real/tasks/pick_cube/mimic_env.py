"""SO-101 pick-place **Isaac Lab Mimic / SkillGen** env 어댑터.

`isaaclab.envs.ManagerBasedRLMimicEnv` 가 요구하는 8개 훅을 구현한다. mimic 데이터 생성은
"source 데모를 object-centric 구간으로 쪼개고, 새 물체 배치에 맞춰 각 구간을 SE(3) 변환해
이어붙인다"가 전부라, env 는 **pose ↔ action 변환**과 **subtask 경계 신호**만 제공하면 된다.

## 프레임 — 하나만 쓴다

mimic 이 다루는 모든 pose(EEF·object)는 **cuRobo URDF solver 프레임**이다(tool = `tcp_grasp`).

이유: `so101_contract.eef_kinematics.SO101EndEffectorKinematics` 가 **같은 URDF + 같은
`so101.yml` 의 `tcp_grasp` extra_link** 를 읽으므로, 레포의 EEF 계약 FK/IK 와 cuRobo SkillGen
planner 가 **원래 같은 프레임**이다. 그래서 sim USD ↔ URDF 변환을 이 클래스에서 **정확히 1회**
(`so101_contract.curobo_frames.T_URDF_FROM_USD`) 하고, planner 쪽은 변환하지 않는다.
★두 곳에서 변환하면 오차가 상쇄돼 조용히 어긋난다 — 프레임 불일치는 이 파이프라인에서 가장
비싼 종류의 버그다(gripper 프레임 자세를 tcp 목표로 쓴 과거 이식본이 증강 성공률 0% 였다).

## action 공간 — 관절을 유지한다

이 env 의 action 은 기존과 같은 **6-dim 절대 joint target**(canonical sim radian)이다.
Franka mimic 처럼 IK-rel action term 을 새로 넣지 않는다:

* `target_eef_pose_to_action` = `SO101BoundedIK`(현재 자세 seed, position 우선) → 관절 목표
* `action_to_target_eef_pose` = 관절 FK

이렇게 두면 생성 데이터셋이 기존 HDF5→LeRobot 변환기·`joint_absolute` 스키마와 그대로 호환되고,
5-DOF 투영(임의 6-DOF pose 는 도달 불가)이 **action 경계에서 명시적으로 1회** 일어난다.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv

from so101_contract.curobo_frames import T_URDF_FROM_USD
from so101_contract.eef_ik import SO101BoundedIK
from so101_contract.eef_kinematics import encode_rotation_matrices
from so101_contract.feature_codec import SO101_JOINT_ORDER
from so101_contract.grasp_manifold import project_pose_best_pan

from .pick_cube_env import PickCubeEnv

#: subtask_configs 의 단일 eef 키.
EEF_NAME = "so101"


class SO101PickCubeMimicEnv(PickCubeEnv, ManagerBasedRLMimicEnv):
    """pick-place mimic/SkillGen env.

    `PickCubeEnv` 를 먼저 상속해 동적 gripper effort clamp(step 훅)를 그대로 물려받고,
    mimic 훅만 여기서 더한다. 둘 다 `ManagerBasedRLEnv` 자손이라 MRO 가 성립한다.
    """

    #: 상태 복원 시 rigid object 를 이만큼 띄운다(m). 근거는 :meth:`reset_to`.
    RESTORE_LIFT_M: float = 6e-4

    def reset_to(self, state, env_ids, seed=None, is_relative=False):
        """`initial_state` 복원 — 물체를 **0.6 mm 띄워서** 넣는다.

        ★왜 — PhysX 는 rest 상태에서 물체를 접촉면에 **약간 파고든 채**로 유지한다
        (`restOffset=0`). 녹화가 그 pose 를 그대로 `initial_state` 에 담고, 복원 쪽이 새 솔버에
        같은 pose 를 꽂으면 솔버는 그걸 **관통**으로 보고 `maxDepenetrationVelocity` 로 밀어낸다.
        큐브 USD 의 그 값은 **1 m/s** — 실측에서 복원 1 스텝 만에 속도가 1.19 → **1098 mm/s**,
        각속도 3.95 → **2798 °/s** 로 튀며 큐브가 15~130 mm 날아가 옆면으로 넘어갔다
        (로봇은 190 mm 밖이라 접촉 아님). 그 결과 계획된 파지점이 어긋나 주석 재생이 실패했다.

        `maxDepenetrationVelocity` 를 낮추는 건 답이 아니다 — `author_pick_cube_scene.py` 에
        `0.5 로 낮췄더니 grasp grip 이 92.8 → 77 % 로 회귀`해 원복한 이력이 적혀 있다.

        대신 **관통을 애초에 만들지 않는다**: 복원 pose 를 접촉면 위로 살짝 띄우면 솔버가 밀어낼
        것이 없고, 물체는 그 높이에서 조용히 내려앉아 **기록된 z 로 정확히 되돌아온다**.
        실측(10/10 데모): 회전 변화 **0.0°**, 이동 = 띄운 양뿐. 0.6 mm 면 충분하고
        자유낙하 속도도 0.1 m/s 미만이라 튀지 않는다.
        """
        objects = state.get("rigid_object") if self.RESTORE_LIFT_M else None
        if objects:
            lifted = {}
            for name, fields in objects.items():
                if "root_pose" in fields:
                    pose = fields["root_pose"].clone()
                    pose[..., 2] += self.RESTORE_LIFT_M
                    fields = {**fields, "root_pose": pose}
                lifted[name] = fields
            # 호출자 dict 는 건드리지 않는다 — 주석 재시도가 같은 state 를 재사용한다.
            state = {**state, "rigid_object": lifted}
        return super().reset_to(state, env_ids, seed=seed, is_relative=is_relative)

    # ══ lazy 계약 객체 ═════════════════════════════════════════════════════════════
    @property
    def eef_ik(self) -> SO101BoundedIK:
        """EEF FK/IK — `so101_contract` 단일 소스. planner 투영 IK 와 같은 계약."""
        if getattr(self, "_eef_ik", None) is None:
            self._eef_ik = SO101BoundedIK.from_files(
                self.cfg.mimic_urdf_path, self.cfg.mimic_robot_yaml)
        return self._eef_ik

    @property
    def canonical_joint_idx(self) -> list[int]:
        """articulation joint 순서 → canonical(SO101_JOINT_ORDER) 인덱스."""
        if getattr(self, "_canonical_joint_idx", None) is None:
            names = self.scene["robot"].joint_names
            self._canonical_joint_idx = [names.index(j) for j in SO101_JOINT_ORDER]
        return self._canonical_joint_idx

    @property
    def _t_urdf_from_usd(self) -> torch.Tensor:
        if getattr(self, "_t_urdf_cached", None) is None:
            self._t_urdf_cached = torch.as_tensor(
                T_URDF_FROM_USD, dtype=torch.float32, device=self.device)
        return self._t_urdf_cached

    # ══ 관절 상태 읽기 ═════════════════════════════════════════════════════════════
    def _canonical_joint_pos(self, env_ids) -> torch.Tensor:
        """(N, 6) canonical 순서 joint radian."""
        return self.scene["robot"].data.joint_pos[env_ids][:, self.canonical_joint_idx]

    # ══ mimic 훅: EEF pose ═════════════════════════════════════════════════════════
    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """현재 tool(`tcp_grasp`) pose, URDF solver 프레임 ``(len(env_ids), 4, 4)``.

        측정 관절의 URDF FK 를 쓴다 — `ee_frame` FrameTransformer(USD `gripper` prim)를 쓰면
        프레임이 달라져 :meth:`target_eef_pose_to_action` 의 IK 와 어긋난다.
        """
        if env_ids is None:
            env_ids = slice(None)
        arm = self._canonical_joint_pos(env_ids)[:, :5].detach().cpu().numpy().astype(np.float64)
        matrices = self.eef_ik.kinematics.forward_matrices(arm)
        return torch.as_tensor(matrices, dtype=torch.float32, device=self.device)

    @property
    def _slew_step_rad(self) -> np.ndarray:
        """한 스텝에 허용되는 arm 관절 변화량(rad) = 슬루 상한 × `step_dt`.

        이 bound 는 인위적 제한이 아니라 **물리적 사실**이다 — 액션 term 의 슬루 리미터가
        어차피 같은 값으로 깎으므로, IK 가 애초에 그 범위에서 해를 찾게 해 브랜치 점프만 막는다.

        ★**축별 추가 clamp 는 두지 말 것**(반증됨). 같은 각변위를 부드럽게 만드는 건 clamp 가
        아니라 **시간을 늘리는 것**이고, 그 레버는 `num_interpolation_steps` 와 planner 의
        `motion_step_size` 다. 실측 = `docs/spec/09_TACIT_KNOWLEDGE.md`.
        """
        if getattr(self, "_slew_step_cached", None) is None:
            caps = self.cfg.actions.arm.max_velocity
            values = [float(caps[joint]) for joint in SO101_JOINT_ORDER[:5]] \
                if isinstance(caps, dict) else [float(caps)] * 5
            step = np.asarray(values, dtype=np.float64) * float(self.step_dt)
            self._slew_step_cached = step * float(self.cfg.mimic_step_bound_scale)
        return self._slew_step_cached

    def put_pending_plan_joints(self, env_id: int, arm_q) -> None:
        """cuRobo 전이 waypoint 의 관절 해를 실행 직전에 넣어 둔다(`generate_mimic_dataset` 호출).

        upstream `MultiWaypoint.execute` 는 `waypoint.pose` 만 env 로 넘기므로 이 우회가 필요하다.
        ★multi-env 에서 env 코루틴이 번갈아 도므로 **env_id 로 키잉**한다.
        """
        if not hasattr(self, "_so101_pending_plan_q"):
            self._so101_pending_plan_q: dict[int, np.ndarray | None] = {}
        self._so101_pending_plan_q[int(env_id)] = arm_q

    def take_pending_plan_joints(self, env_id: int) -> np.ndarray | None:
        """넣어 둔 관절 해를 **1회용**으로 꺼낸다. 없으면 None(= pose→IK 경로)."""
        pending = getattr(self, "_so101_pending_plan_q", {}).pop(int(env_id), None)
        return None if pending is None else np.asarray(pending, dtype=np.float64).reshape(5)

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """목표 tool pose(4×4, solver 프레임) → 6-dim 절대 joint target action.

        경로가 둘이다.

        **① cuRobo 전이 구간** — planner 가 이미 관절공간에서 푼 해를 그대로 쓴다
        (:meth:`take_pending_plan_joints`). 그 해를 FK 로 pose 화했다가 여기서 다시 IK 로 푸는
        왕복은 5-DOF 에서 항등이 아니고, 매 프레임 pan 이 재스캔되며 ρ(=`wrist_roll`)가 따라
        움직인다 — 증강본에서 손목이 일그러지는 직접 원인이다. 왕복을 없애면 planner 가 만든
        매끄러운 관절 궤적이 그대로 명령된다.

        **② source 재생·진입 보간·합성 구간** — 목표를 5-DOF 도달 manifold 로 투영
        (`project_pose_best_pan`)한 뒤 IK 를 푼다. 투영 없이 풀면 IK 가 position 과 orientation
        을 절충해 위치까지 어긋난다(실측 실패 5/5: 위치 잔차 20~181 mm · 회전 20~93°).
        planner 도 **같은 함수**로 투영한다 — 다르게 투영하면 그 간극이 곧 추종오차다.

        두 경로 공통으로 **step bound** 를 건다: `측정 자세 ± 슬루상한×step_dt`. 도달 불가
        목표가 "멀리 튀는" 대신 한 스텝만큼만 다가가다 멈추게 하고, DLS 해 브랜치 점프도 막는다.
        ★seed·bound 기준은 반드시 **측정 자세**다(직전 명령 기준은 적분 와인드업을 만든다).
        수치 근거 = `docs/spec/09_TACIT_KNOWLEDGE.md`.

        노이즈는 관절공간에서 더한 뒤 step bound·URDF limit 으로 다시 clamp 한다.
        """
        seed = self._canonical_joint_pos([env_id])[0, :5].detach().cpu().numpy().astype(np.float64)
        step = self._slew_step_rad

        arm_target = self.take_pending_plan_joints(env_id)
        if arm_target is None:
            (target_eef_pose,) = target_eef_pose_dict.values()
            matrix = target_eef_pose.detach().cpu().numpy().astype(np.float64).reshape(4, 4)
            projected = project_pose_best_pan(self.eef_ik, matrix, seed)
            pose_vec = np.concatenate([
                projected[:3, 3], encode_rotation_matrices(projected[:3, :3][None], "rot6d")[0]])
            result = self.eef_ik.solve(pose_vec, seed, representation="rot6d",
                                       step_lower_rad=seed - step, step_upper_rad=seed + step)
            arm_target = np.asarray(result.joint_radians, dtype=np.float64)

        if action_noise_dict is not None:
            # ★값이 `None` 일 수 있다. `Waypoint.noise` 의 기본값이 None 이고
            #   `MultiWaypoint.execute` 는 그 값을 그대로 dict 에 담는다 — dict 존재 여부만
            #   보고 `float()` 하면 `TypeError: float() argument ... not 'NoneType'` 로
            #   생성이 통째로 죽는다(실측: 합성 waypoint 를 noise=None 로 만들었다가 run 전멸).
            scale = float(action_noise_dict.get(EEF_NAME) or 0.0)
            if scale > 0.0:
                arm_target = arm_target + scale * np.random.randn(5)
        arm_target = np.clip(arm_target, seed - step, seed + step)
        limits = self.eef_ik.joint_limits_rad
        arm_target = np.clip(arm_target, limits[:, 0], limits[:, 1])

        (gripper_action,) = gripper_action_dict.values()
        arm_tensor = torch.as_tensor(arm_target, dtype=torch.float32, device=self.device)
        return torch.cat([arm_tensor, gripper_action.to(self.device).reshape(-1)], dim=0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """:meth:`target_eef_pose_to_action` 의 역 — 관절 target FK.

        5-DOF 투영이 들어간 만큼 왕복이 정확히 항등은 아니다. 그래서 mimic 이 source 데모에서
        복원하는 pose 열은 **실현 가능한 pose 열**이 되고, 구간 변환도 그 위에서 이뤄진다.
        """
        arm = action[:, :5].detach().cpu().numpy().astype(np.float64)
        matrices = self.eef_ik.kinematics.forward_matrices(arm)
        return {EEF_NAME: torch.as_tensor(matrices, dtype=torch.float32, device=self.device)}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """action 의 마지막 차원 = gripper joint target(radian)."""
        return {EEF_NAME: actions[..., -1:]}

    # ══ mimic 훅: object pose ══════════════════════════════════════════════════════
    def get_object_poses(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """mimic 대상 물체 pose ``(N,4,4)``, **URDF solver 프레임**.

        기본 구현은 env-origin 상대 USD pose 를 주는데, 우리는 로봇이 책상 위(env-local
        z≈0.675)에 장착돼 있어 그 좌표계는 EEF pose 와 다른 프레임이다. robot root full SE(3)
        를 빼고 USD→URDF 변환까지 적용해 **EEF pose 와 같은 프레임**으로 맞춘다.
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        env_ids = list(env_ids)
        robot = self.scene["robot"]
        root_pos = robot.data.root_pos_w[env_ids]
        root_quat = robot.data.root_quat_w[env_ids]

        poses: dict[str, torch.Tensor] = {}
        for name, obj in self.scene.rigid_objects.items():
            pos_b, quat_b = PoseUtils.subtract_frame_transforms(
                root_pos, root_quat,
                obj.data.root_pos_w[env_ids], obj.data.root_quat_w[env_ids],
            )
            pose_usd = PoseUtils.make_pose(pos_b, PoseUtils.matrix_from_quat(quat_b))
            poses[name] = self._t_urdf_from_usd @ pose_usd
        return poses

    # ══ mimic 훅: subtask 경계 ═════════════════════════════════════════════════════
    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """subtask **종료** 신호 — `observations.subtask_terms` 그룹을 그대로 노출."""
        if env_ids is None:
            env_ids = slice(None)
        terms = self.obs_buf["subtask_terms"]
        return {name: terms[name][env_ids].reshape(-1) for name in self.cfg.mimic_term_signals}

    def get_subtask_start_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """subtask **시작** 신호 — SkillGen(`use_skillgen=True`)에서 필수.

        SkillGen 은 ``[이전 subtask 종료 → 이번 subtask 시작]`` 구간을 motion planner 로 다시
        계획하고 ``[시작, 종료]`` 구간만 source 에서 재생한다. 그래서 시작 지점은

        * **그리퍼가 아직 열려 있을 때** 여야 한다. 폐합 뒤로 잡으면 planner 가 닫힌 그리퍼로
          접근 구간을 계획해 큐브를 밀어낸다.
        * 이전 subtask 종료보다 **엄격히 뒤** 여야 한다(`datagen_info_pool` 이 단조성을 assert).

        키는 종료 신호와 **같은 이름**을 쓴다(`datagen_info_pool` 이 그렇게 조회한다).
        임계값 2개는 cfg 의 knob 이다 — 큐브 크기·grasp 튜닝이 바뀌면 실측으로 다시 잡는다.
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        env_ids = list(env_ids)

        grasp_signal, place_signal = self.cfg.mimic_term_signals
        cube_name = self.cfg.mimic_grasp_object

        tool_pos = self.get_robot_eef_pose(EEF_NAME, env_ids=env_ids)[:, :3, 3]
        cube_pos = self.get_object_poses(env_ids=env_ids)[cube_name][:, :3, 3]
        distance = torch.linalg.vector_norm(tool_pos - cube_pos, dim=-1)
        gripper = self._canonical_joint_pos(env_ids)[:, 5]

        approaching = (distance < self.cfg.mimic_approach_radius_m) & (
            gripper > self.cfg.mimic_gripper_open_rad)

        # place 시작 = 파지 확정 + 상승 + **그릇 상공 도달**.
        #
        # ★상승만으로 끊으면 안 된다 — subtask1 은 `object_ref="Bowl"` 이라 그릇 기준으로 SE(3)
        # 변환되는데 상승 직후 자세는 아직 큐브에 종속이라, 전이가 파지 지점에서 source 큐브
        # 자리로 갔다가 다시 그릇으로 향한다. 그릇 상공에서 끊으면 남는 구간이
        # 하강+투하+retreat+hold 뿐이라 전부 그릇 상대 동작이 된다.
        # ★시작 시점에 그리퍼가 **아직 닫혀 있어야** 한다 — 전이의 그리퍼 명령은 다음 구간 첫
        # waypoint 에서 오므로, 투하 이후로 잡으면 전이 내내 열린 채라 큐브를 떨군다.
        # 실측 수치 = `docs/spec/09_TACIT_KNOWLEDGE.md`.
        grasped = self.obs_buf["subtask_terms"][grasp_signal][env_ids].reshape(-1) > 0.5
        cube_world = self.scene[cube_name].data.root_pos_w[env_ids]
        bowl_world = self.scene[self.cfg.mimic_place_object].data.root_pos_w[env_ids]
        lifted = cube_world[:, 2] > (self.cfg.mimic_desk_top_world_z
                                     + self.cfg.mimic_place_start_lift_m)
        over_bowl = (torch.linalg.vector_norm(cube_world[:, :2] - bowl_world[:, :2], dim=-1)
                     < self.cfg.mimic_place_start_bowl_xy_m)

        return {
            grasp_signal: approaching.float(),
            place_signal: (grasped & lifted & over_bowl).float(),
        }

    # ══ SkillGen 훅: 구간별 부착 물체 ══════════════════════════════════════════════
    def get_expected_attached_object(self, eef_name: str, subtask_index: int, env_cfg) -> str | None:
        """이 subtask 를 수행하는 동안 들고 있어야 하는 물체(없으면 None).

        planner 가 잡은 큐브 부피를 포함해 전이를 계획하도록 알려주는 값이다.
        규칙: place 계열 subtask 는 **직전 grasp subtask 의 대상**을 들고 있다.
        """
        configs = env_cfg.subtask_configs.get(eef_name, [])
        if not 0 <= subtask_index < len(configs):
            return None
        current = str(configs[subtask_index].subtask_term_signal or "").lower()
        if "place" not in current or subtask_index == 0:
            return None
        previous = configs[subtask_index - 1]
        if "grasp" in str(previous.subtask_term_signal or "").lower():
            return previous.object_ref
        return None
