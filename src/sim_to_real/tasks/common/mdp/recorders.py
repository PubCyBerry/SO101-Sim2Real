"""Datagen recorder term — leisaac 방식 IsaacLab HDF5 녹화용 커스텀 term.

정책 obs 가 ``joint_pos_rel``(+concat) 이라 stock ``ActionStateRecorderManagerCfg`` 만으로는
절대 joint 각·카메라 이미지가 HDF5 에 남지 않는다. 이 term 이 캐노니컬(SO101 순서) 형태로
직접 기록해 변환기(``scripts/convert/isaaclab2lerobotv3.py``)가 Isaac 부팅 없이 동작하게 한다.

HDF5 키:
- ``obs_x/joint_pos``           (T, 6)  절대 radian, SO101_JOINT_ORDER (pre-step = obs_t)
- ``obs_x/images/{top,wrist,front}`` (T, H, W, 3) uint8 (pre-step)
- ``applied_target``            (T, 6)  slew 통과한 실제 적용 joint target, SO101 순서 (post-step)
  — pre-slew raw ``actions`` 를 BC target 으로 쓰면 jerky 데이터가 되는 기존 교훈의 방어.
"""

from __future__ import annotations

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from so101_contract.feature_codec import SO101_JOINT_ORDER


class DatagenRecorderTerm(RecorderTerm):
    """절대 joint(SO101 순서) + 3-cam 이미지(pre-step), 적용 target(post-step) 기록."""

    def __init__(self, cfg: RecorderTermCfg, env) -> None:
        super().__init__(cfg, env)
        self._joint_idx: list[int] | None = None  # articulation 순서 → SO101 순서
        self._action_idx: list[int] | None = None  # arm action term 컬럼 순서 → SO101 순서

    def _resolve_indices(self) -> None:
        robot = self._env.scene["robot"]
        self._joint_idx = [robot.joint_names.index(j) for j in SO101_JOINT_ORDER]
        arm = self._env.action_manager.get_term("arm")
        # JointAction 은 preserve_order=False 기본이라 컬럼이 asset 순서일 수 있음 → 이름으로 매핑
        assert len(arm._joint_names) == len(SO101_JOINT_ORDER), (
            f"arm action term joint 수 불일치: {arm._joint_names}"
        )
        self._action_idx = [arm._joint_names.index(j) for j in SO101_JOINT_ORDER]

    def record_pre_step(self):
        if self._joint_idx is None:
            self._resolve_indices()
        robot = self._env.scene["robot"]
        images = self._env.obs_buf["images"]  # {top,wrist,front}: (N,H,W,3) uint8 — obs_t
        return "obs_x", {
            "joint_pos": robot.data.joint_pos[:, self._joint_idx],
            "images": dict(images),
        }

    def record_post_step(self):
        if self._action_idx is None:
            self._resolve_indices()
        arm = self._env.action_manager.get_term("arm")
        return "applied_target", arm.processed_actions[:, self._action_idx]


@configclass
class DatagenRecorderTermCfg(RecorderTermCfg):
    """Configuration for :class:`DatagenRecorderTerm`."""

    class_type: type[RecorderTerm] = DatagenRecorderTerm


@configclass
class SO101DatagenRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """datagen term + stock 중 실제로 쓰이는 2종만.

    stock 5종 가운데 ``states``·``obs``·``processed_actions`` 는 **읽는 코드가 저장소에 없다**
    (변환기 ``scripts/convert/isaaclab2lerobotv3.py`` 는 ``applied_target``·``obs_x/joint_pos``·
    ``obs_x/images/*`` + attrs ``success`` 만 소비한다). 매 스텝 전체 씬 state 를 GPU 에서 복제해
    쌓기만 하므로 끈다 — ``RecorderManager._prepare_terms`` 가 ``None`` term 을 건너뛴다.

    남기는 2종:
    - ``record_initial_state`` — 씬 재현용. reset 당 1회라 무게 없음.
    - ``record_pre_step_actions`` — ``HDF5DatasetFileHandler.write_episode`` 가 demo attrs
      ``num_samples`` 를 ``episode.data["actions"]`` 길이로 잡는다. 끄면 0 이 된다(6 float/step).
    """

    record_datagen = DatagenRecorderTermCfg()
    record_post_step_states: RecorderTermCfg | None = None
    record_pre_step_flat_policy_observations: RecorderTermCfg | None = None
    record_post_step_processed_actions: RecorderTermCfg | None = None
