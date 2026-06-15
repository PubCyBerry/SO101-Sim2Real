# SO-101 (3-camera) GR00T-N1.7 modality config — NEW_EMBODIMENT.
#
# NVIDIA Isaac-GR00T `examples/SO100/so100_config.py` 를 베이스로 카메라를 3개
# (top/wrist/front) 로 확장한 사본이다. SO-101 = SO-100 과 동일한 6-DOF(arm 5 +
# gripper 1) 패밀리라 state/action 인덱싱·RELATIVE/ABSOLUTE 표현은 그대로 두고
# video 만 3-cam 으로 늘린다.
#
# - finetune 시 `--modality-config-path` 로 전달되어 체크포인트에 baked 된다.
#   (서빙 `run_gr00t_server.py` 는 baked config 를 자동 로드 → 별도 전달 불요)
# - video.modality_keys 는 데이터셋 `meta/modality.json` 의 video 엔트리 키와
#   일치해야 한다(configs/groot/so101_modality.json: front/wrist/top).
# - 한 프로세스에 NEW_EMBODIMENT modality config 는 단 하나만 register 가능하다.
#   (so100_config.py 와 동시 import 금지 — CLI 는 선택한 한 파일만 import)

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


so101_config = {
    # Video: 현재 프레임만. 키는 meta/modality.json 의 "video" 엔트리와 일치해야 한다.
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["front", "wrist", "top"],  # 3-cam: 전면 + 손목 + 탑뷰
    ),
    # State: 현재 proprioception. 키는 meta/modality.json 의 "state" 엔트리와 일치.
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "single_arm",  # joint positions (5)
            "gripper",  # gripper state (1)
        ],
    ),
    # Action: 16-step 예측 호라이즌. modality key 당 ActionConfig 1개.
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),  # predict 16 future steps
        modality_keys=[
            "single_arm",
            "gripper",
        ],
        action_configs=[
            # single_arm: RELATIVE = 현재 state 기준 delta (일반화 우수).
            #   ※ get_action() 반환값은 unapply_action 이 현재 state 로 절대값을 복원하므로
            #     데이터셋 네이티브 action 공간(절대 joint deg)으로 나온다 — bridge 는 그대로 통과.
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,  # joint-space (not end-effector)
                format=ActionFormat.DEFAULT,
            ),
            # gripper: ABSOLUTE = target position (open/close 는 절대값이 안정적).
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    # Language: 데이터셋 annotation 필드의 task instruction.
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(so101_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
