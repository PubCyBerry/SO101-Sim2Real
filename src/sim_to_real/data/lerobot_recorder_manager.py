"""LeRobot v3 직기록 RecorderManager — leisaac ``LeRobotRecorderManager`` 이식.

leisaac 설계 그대로: ①부모 init 을 ``EXPORT_NONE`` 으로 위장해 HDF5 handler 생성 억제
②``record_post_step`` 마다 frame 을 스트리밍(add_frame)하고 episode 버퍼를 비움
③export 시 success → commit, 실패 → 버퍼 폐기 ④**single-env 전용**(leisaac 은
``env_idx = 0`` 하드코딩 — 우리는 assert 로 명시. multi-env 는 ``--record_hdf5`` 경로).

백엔드는 lerobot 패키지(``LeRobotDataset``)가 아니라 in-container
:class:`~sim_to_real.data.lerobot_recorder.LeRobotV3DatasetWriter` — isaac-sim 컨테이너에
lerobot 이 없기 때문(leisaac 은 이 경우 pickle fallback 으로 degrade 한다). frame 계약은
``record_state_machine.py`` 와 동일(PolicyFeature 단위 action/state + 3-cam uint8).

HDF5 경로 대비 장점: 이미지가 에피소드 동안 GPU 에 누적되지 않는다(스텝마다 CPU 스트리밍).
"""

from __future__ import annotations

from isaaclab.managers import DatasetExportMode, RecorderManager
from isaaclab.utils.datasets.episode_data import EpisodeData

from so101_contract.feature_codec import CAMERA_KEYS, SO101_JOINT_ORDER

from .lerobot_recorder import LeRobotV3DatasetWriter
from .lerobot_units import to_lerobot_units


class SO101LeRobotRecorderManager(RecorderManager):
    """LeRobot v3 직기록 manager (single-env, leisaac 동형)."""

    def __init__(self, cfg: object, env, writer: LeRobotV3DatasetWriter, task_name: str) -> None:
        # leisaac 트릭: 부모가 dataset file handler 를 만들지 않도록 EXPORT_NONE 위장
        real_mode = cfg.dataset_export_mode
        cfg.dataset_export_mode = DatasetExportMode.EXPORT_NONE
        super().__init__(cfg, env)
        cfg.dataset_export_mode = real_mode

        assert env.num_envs == 1, "record_lerobot 은 single-env 전용 — multi-env 는 --record_hdf5"
        self._writer = writer
        self._task_name = task_name
        self._joint_idx: list[int] | None = None
        self._action_idx: list[int] | None = None

    def __str__(self) -> str:
        return "SO101LeRobotRecorderManager (LeRobotV3DatasetWriter backend)\n" + super().__str__()

    def _resolve_indices(self) -> None:
        robot = self._env.scene["robot"]
        self._joint_idx = [robot.joint_names.index(j) for j in SO101_JOINT_ORDER]
        arm = self._env.action_manager.get_term("arm")
        self._action_idx = [arm._joint_names.index(j) for j in SO101_JOINT_ORDER]

    def record_post_step(self) -> None:
        """frame 1개 조립 → writer 스트리밍. episode 버퍼는 상시 비움(leisaac 동형)."""
        super().record_post_step()
        if self._joint_idx is None:
            self._resolve_indices()
        env = self._env
        robot = env.scene["robot"]
        arm = env.action_manager.get_term("arm")
        # obs_buf 는 이 시점에 아직 obs_t (obs 갱신은 step 말미) → obs_t + action_t 정합
        state = to_lerobot_units(robot.data.joint_pos[0, self._joint_idx].cpu().numpy())
        action = to_lerobot_units(arm.processed_actions[0, self._action_idx].cpu().numpy())
        images = {k: env.obs_buf["images"][k][0].cpu().numpy() for k in CAMERA_KEYS}
        self._writer.add_frame(action, state, images)
        if 0 in self._episodes:
            self._episodes[0]._data.clear()

    def reset(self, env_ids=None):
        """진행 중(미커밋) frame 버퍼 폐기 — record 루프의 쓰레기 프레임 절삭용."""
        self._writer.commit_episode(success=False, task_name=self._task_name)
        return super().reset(env_ids)

    def export_episodes(self, env_ids=None, demo_ids=None) -> None:
        """success(termination "success" term → episode attr) 면 commit, 아니면 폐기."""
        if len(self.active_terms) == 0:
            return
        episode = self._episodes.get(0)
        success = bool(episode.success) if episode is not None and episode.success is not None else False
        committed = self._writer.commit_episode(success=success, task_name=self._task_name)
        if committed:
            self._exported_successful_episode_count[0] = self._exported_successful_episode_count.get(0, 0) + 1
        else:
            self._exported_failed_episode_count[0] = self._exported_failed_episode_count.get(0, 0) + 1
        self._episodes[0] = EpisodeData()

    def close(self) -> None:
        if getattr(self, "_closed", False):  # env.close(del) + gc 이중 호출 가드
            return
        self._closed = True
        summary = self._writer.finalize(self._task_name)
        print(f"[record_lerobot] finalize: {summary}", flush=True)
        for term in self._terms.values():
            term.close("")
