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

from sim_to_real.data.hdf5_compression import hdf5_handler
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

    # ⚠ 반환 텐서는 **GPU 에 둔다.** RecorderManager 가 `value.clone()` 으로 device 를 보존해
    #   에피소드 내내 쌓으므로(EpisodeData.add) 이미지가 VRAM 에 누적되는 건 맞다
    #   (3-cam 640×480 = 2.64 MiB/step/env → 379-step 에피소드면 999 MiB/env).
    #   여기서 `.cpu()` 로 미리 내리면 VRAM 은 num_envs=8 기준 45.0 → 34.4 GB 로 줄지만,
    #   스텝마다 렌더 파이프라인을 드레인시키는 **동기화** 비용이 붙어 replay 가 8-env 에서
    #   25.9 → 76.5 s (3×) 로 뛴다. 실측 결과 트라이얼 117.3 → 165.1 s (+41%) 라 철회했다.
    #   → VRAM 상한과 복구 방법 = docs/spec/09_TACIT_KNOWLEDGE.md §13.3.

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

    # gzip(4, IsaacLab 기본) → lzf + frame-chunk. export 10.8 → 3.7 s/demo 실측, 디스크는 2배.
    # 되돌리려면 "gzip" 으로 바꾸면 된다(프리셋 표 = hdf5_compression 모듈 docstring).
    dataset_file_handler_class_type: type = hdf5_handler("lzf")

    record_datagen = DatagenRecorderTermCfg()
    record_post_step_states: RecorderTermCfg | None = None
    record_pre_step_flat_policy_observations: RecorderTermCfg | None = None
    record_post_step_processed_actions: RecorderTermCfg | None = None
