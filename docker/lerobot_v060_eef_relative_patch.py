#!/usr/bin/env python3
"""LeRobot v0.6.0에 SO-101 schema v2 action representation pipeline을 연결하는 멱등 patch.

적용 범위(Phase 15, universal):

- ACT·SmolVLA·GR00T config에 4-mode ``action_representation`` nested flag 추가
  (``joint_absolute`` · ``joint_relative`` · ``eef_absolute`` · ``eef_relative``)
- train에서 모든 policy processor에 ``dataset_meta`` 전달, 모든 신규 checkpoint(periodic·
  final·Hub)에 schema v2 ``action_representation.json`` 저장
- policy factory에서 dataset 계약 → mode/format/horizon stats profile → registered v2
  encode/decode step 삽입·교체·재연결, pretrained feature schema 교체
- pretrained weight load를 selective-reuse/reinit report 경로로 교체
- mean/std normalizer의 forward/inverse epsilon 대칭성 보정
- async policy server에서 v2 checkpoint의 full chunk를 한 번에 postprocess한 뒤 slice
- sync rollout/eval에서 ``select_action`` 대신 external full-chunk FIFO 사용

설치된 LeRobot의 형태가 v0.6.0 기준과 다르면 즉시 실패하는 version tripwire다.
테스트용 source tree에는 ``--package-dir .../src/lerobot``으로 적용할 수 있다.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
from pathlib import Path

LEROBOT_VERSION = "0.6.0"

ACTION_REPRESENTATION_MODULE = '''"""Policy 공통 action representation config (schema v2).

4-mode enum과 EEF pose format을 CLI/config 수준에서 강제한다. 단일 소스는
``so101_contract.action_representation.ActionRepresentationSpec``이며, 이 dataclass는
LeRobot draccus CLI가 파싱할 수 있는 평평한 표현이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MODES = (
    "joint_absolute",
    "joint_relative",
    "eef_absolute",
    "eef_relative",
)
EEF_POSE_FORMATS = (
    "xyz_rot6d_rows",
    "xyz_quaternion_wxyz",
    "xyz_rpy",
)
_AMBIGUOUS = {"absolute", "relative"}


@dataclass
class ActionRepresentationConfig:
    """--policy.action_representation.* CLI 계약.

    - ``mode``: 정확히 4개 값만 허용한다. 모호한 ``absolute``/``relative``는 거부한다.
    - ``pose_format``: joint mode에서는 ``None``/``not_applicable``만, EEF mode에서는
      3개 format 중 하나를 명시해야 한다(자동 추정 없음).
    - ``state_pose_group``/``action_pose_group``/``reference``는 v1 checkpoint config를
      읽기 위한 legacy alias이며 신규 config에서는 비워 둔다.
    """

    mode: str = "joint_absolute"
    pose_format: str | None = None
    reference_observation: str = "current_observation"
    state_group: str = ""
    action_group: str = ""
    passthrough_action_groups: list[str] = field(
        default_factory=lambda: ["gripper_position"]
    )
    gripper_representation: str = "absolute"
    base_frame: str | None = None
    eef_frame: str | None = None
    stats_file: str = "meta/action_representation_stats.json"
    strict: bool = True

    # --- v1 checkpoint config 호환 alias (신규 config에서는 사용하지 않는다) ---
    reference: str | None = None
    state_pose_group: str | None = None
    action_pose_group: str | None = None

    def __post_init__(self) -> None:
        mode = (self.mode or "").strip().lower()
        if mode in _AMBIGUOUS:
            raise ValueError(
                f"action_representation.mode={self.mode!r} is ambiguous in schema v2; "
                f"choose one of {list(MODES)}"
            )
        if mode not in MODES:
            raise ValueError(
                f"action_representation.mode must be one of {list(MODES)}, got {self.mode!r}"
            )
        self.mode = mode

        pose_format = self.pose_format
        if isinstance(pose_format, str):
            pose_format = pose_format.strip().lower()
            if pose_format in ("", "none", "null", "not_applicable"):
                pose_format = None
        if mode.startswith("joint_"):
            if pose_format is not None:
                raise ValueError(
                    f"joint modes require pose_format=None/not_applicable, got {self.pose_format!r}"
                )
            if self.base_frame or self.eef_frame:
                raise ValueError("joint modes must not declare base_frame/eef_frame")
        else:
            if pose_format is None:
                raise ValueError(
                    f"mode={mode!r} requires an explicit pose_format among "
                    f"{list(EEF_POSE_FORMATS)}"
                )
            if pose_format not in EEF_POSE_FORMATS:
                raise ValueError(
                    f"pose_format must be one of {list(EEF_POSE_FORMATS)}, got {self.pose_format!r}"
                )
        self.pose_format = pose_format

        if self.gripper_representation != "absolute":
            raise ValueError("gripper_representation must stay 'absolute' in every mode")
        if not self.passthrough_action_groups:
            raise ValueError("passthrough_action_groups must contain the gripper group")

        # legacy alias 흡수
        if not self.state_group and self.state_pose_group:
            self.state_group = self.state_pose_group
        if not self.action_group and self.action_pose_group:
            self.action_group = self.action_pose_group
        if self.reference and self.reference_observation == "current_observation":
            self.reference_observation = self.reference
'''


def _lerobot_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        root = explicit.resolve()
        if not (root / "__init__.py").is_file():
            raise RuntimeError(f"--package-dir is not a lerobot package directory: {root}")
        return root
    version = importlib.metadata.version("lerobot")
    if version != LEROBOT_VERSION:
        raise RuntimeError(
            f"action representation patch requires lerobot=={LEROBOT_VERSION}, got {version}"
        )
    spec = importlib.util.find_spec("lerobot")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("installed lerobot package was not found")
    return Path(next(iter(spec.submodule_search_locations)))


def _patch(
    path: Path,
    old: str,
    new: str,
    *,
    sentinel: str,
    label: str,
) -> None:
    text = path.read_text(encoding="utf-8")
    if sentinel in text:
        print(f"[action-representation-patch] already applied: {label}")
    elif old in text:
        path.write_text(text.replace(old, new), encoding="utf-8")
        print(f"[action-representation-patch] applied: {label}")
    else:
        raise RuntimeError(f"Unexpected LeRobot v0.6.0 layout — {label} ({path})")


def _add_file(path: Path, content: str, *, label: str) -> None:
    if not path.exists():
        path.write_text(content, encoding="utf-8")
        print(f"[action-representation-patch] added: {label}")
    elif path.read_text(encoding="utf-8") == content:
        print(f"[action-representation-patch] already added: {label}")
    else:
        raise RuntimeError(f"Unexpected existing file — {label} ({path})")


def _patch_configs(root: Path) -> None:
    _add_file(
        root / "configs" / "action_representation.py",
        ACTION_REPRESENTATION_MODULE,
        label="ActionRepresentationConfig (schema v2)",
    )

    act = root / "policies" / "act" / "configuration_act.py"
    _patch(
        act,
        old="from lerobot.configs import NormalizationMode, PreTrainedConfig\n",
        new=(
            "from lerobot.configs import NormalizationMode, PreTrainedConfig\n"
            "from lerobot.configs.action_representation import ActionRepresentationConfig\n"
        ),
        sentinel="from lerobot.configs.action_representation import ActionRepresentationConfig",
        label="ACT action representation import",
    )
    _patch(
        act,
        old="    n_action_steps: int = 100\n\n    normalization_mapping:",
        new=(
            "    n_action_steps: int = 100\n"
            "    action_representation: ActionRepresentationConfig = field(\n"
            "        default_factory=ActionRepresentationConfig\n"
            "    )\n\n"
            "    normalization_mapping:"
        ),
        sentinel="action_representation: ActionRepresentationConfig",
        label="ACT action representation field",
    )

    smolvla = root / "policies" / "smolvla" / "configuration_smolvla.py"
    _patch(
        smolvla,
        old=(
            "from lerobot.configs import FeatureType, NormalizationMode, "
            "PolicyFeature, PreTrainedConfig\n"
        ),
        new=(
            "from lerobot.configs import FeatureType, NormalizationMode, "
            "PolicyFeature, PreTrainedConfig\n"
            "from lerobot.configs.action_representation import ActionRepresentationConfig\n"
        ),
        sentinel="from lerobot.configs.action_representation import ActionRepresentationConfig",
        label="SmolVLA action representation import",
    )
    _patch(
        smolvla,
        old="    chunk_size: int = 50\n    n_action_steps: int = 50\n",
        new=(
            "    chunk_size: int = 50\n"
            "    n_action_steps: int = 50\n"
            "    action_representation: ActionRepresentationConfig = field(\n"
            "        default_factory=ActionRepresentationConfig\n"
            "    )\n"
        ),
        sentinel="action_representation: ActionRepresentationConfig",
        label="SmolVLA action representation field",
    )

    groot = root / "policies" / "groot" / "configuration_groot.py"
    _patch(
        groot,
        old=(
            "from lerobot.configs import FeatureType, NormalizationMode, "
            "PolicyFeature, PreTrainedConfig\n"
        ),
        new=(
            "from lerobot.configs import FeatureType, NormalizationMode, "
            "PolicyFeature, PreTrainedConfig\n"
            "from lerobot.configs.action_representation import ActionRepresentationConfig\n"
        ),
        sentinel="from lerobot.configs.action_representation import ActionRepresentationConfig",
        label="GR00T action representation import",
    )
    _patch(
        groot,
        old=("    use_relative_actions: bool = False\n\n    # relative_exclude_joints"),
        new=(
            "    use_relative_actions: bool = False\n"
            "    action_representation: ActionRepresentationConfig = field(\n"
            "        default_factory=ActionRepresentationConfig\n"
            "    )\n\n"
            "    # relative_exclude_joints"
        ),
        sentinel="action_representation: ActionRepresentationConfig",
        label="GR00T action representation field",
    )


def _patch_train(root: Path) -> None:
    train = root / "scripts" / "lerobot_train.py"
    _patch(
        train,
        old="from lerobot.utils.collate import lerobot_collate_fn\n",
        new=(
            "from lerobot.utils.collate import lerobot_collate_fn\n"
            "from so101_contract.lerobot_v2_integration import (\n"
            "    push_action_representation_manifest_for,\n"
            "    write_action_representation_manifest_for,\n"
            ")\n"
        ),
        sentinel="from so101_contract.lerobot_v2_integration import (",
        label="train schema v2 manifest imports",
    )
    _patch(
        train,
        old=(
            "    processor_kwargs = {}\n"
            "    if (processor_pretrained_path and not cfg.resume) or not processor_pretrained_path:\n"
            '        processor_kwargs["dataset_stats"] = dataset.meta.stats\n'
            "\n"
            "    if cfg.is_reward_model_training:\n"
            '        processor_kwargs["dataset_meta"] = dataset.meta\n'
        ),
        new=(
            '    processor_kwargs = {"dataset_meta": dataset.meta}\n'
            "    if (processor_pretrained_path and not cfg.resume) or not processor_pretrained_path:\n"
            '        processor_kwargs["dataset_stats"] = dataset.meta.stats\n'
        ),
        sentinel='processor_kwargs = {"dataset_meta": dataset.meta}',
        label="train dataset_meta for every policy",
    )
    _patch(
        train,
        old=(
            "                    optim_state_dict=optim_state_dict,\n"
            "                )\n"
            "                update_last_checkpoint(checkpoint_dir)\n"
        ),
        new=(
            "                    optim_state_dict=optim_state_dict,\n"
            "                )\n"
            "                write_action_representation_manifest_for(\n"
            '                    checkpoint_dir / "pretrained_model",\n'
            "                    cfg.policy,\n"
            "                    preprocessor,\n"
            "                    policy=policy,\n"
            "                )\n"
            "                update_last_checkpoint(checkpoint_dir)\n"
        ),
        sentinel="write_action_representation_manifest_for(",
        label="periodic checkpoint schema v2 manifest",
    )
    _patch(
        train,
        old=(
            "            preprocessor.push_to_hub(active_cfg.repo_id)\n"
            "            postprocessor.push_to_hub(active_cfg.repo_id)\n"
        ),
        new=(
            "            preprocessor.push_to_hub(active_cfg.repo_id)\n"
            "            postprocessor.push_to_hub(active_cfg.repo_id)\n"
            "            push_action_representation_manifest_for(\n"
            "                active_cfg.repo_id, cfg.policy, preprocessor, policy=policy\n"
            "            )\n"
        ),
        sentinel="push_action_representation_manifest_for(",
        label="final Hub schema v2 manifest",
    )


def _patch_factory(root: Path) -> None:
    factory = root / "policies" / "factory.py"
    _patch(
        factory,
        old="import importlib\nimport logging\n",
        new="import importlib\nimport logging\nimport os\n",
        sentinel="import os\n",
        label="policy factory os import",
    )
    # registry 이름(so101_action_representation_*_v2)을 해석하려면 step module이 import돼
    # 있어야 한다. checkpoint만 로드하는 경로(context 없음)에서도 등록되도록 factory에서
    # 최상위 import한다.
    _patch(
        factory,
        old="from lerobot.envs import EnvConfig, env_to_policy_features\n",
        new=(
            "from lerobot.envs import EnvConfig, env_to_policy_features\n"
            "import so101_contract.action_representation_processor  # noqa: F401\n"
        ),
        sentinel="import so101_contract.action_representation_processor",
        label="schema v2 processor step registration",
    )
    _patch(
        factory,
        old=(
            "    cfg.output_features = {key: ft for key, ft in features.items() "
            "if ft.type is FeatureType.ACTION}\n"
            "    if not cfg.input_features:\n"
            "        cfg.input_features = {key: ft for key, ft in features.items() "
            "if key not in cfg.output_features}\n"
        ),
        new=(
            "    from so101_contract.lerobot_v2_integration import override_policy_features\n"
            "\n"
            "    # 대상 policy(ACT/SmolVLA/GR00T)는 mode와 무관하게 dataset schema가 진실이다.\n"
            "    # pretrained VLA config가 들고 있는 옛 state/action 차원을 그대로 두면\n"
            "    # projection/head가 잘못된 크기로 만들어진다.\n"
            "    if not override_policy_features(\n"
            "        cfg, features, rename_map, feature_type_action=FeatureType.ACTION\n"
            "    ):\n"
            "        cfg.output_features = {\n"
            "            key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION\n"
            "        }\n"
            "        if not cfg.input_features:\n"
            "            cfg.input_features = {\n"
            "                key: ft for key, ft in features.items() if key not in cfg.output_features\n"
            "            }\n"
        ),
        sentinel="from so101_contract.lerobot_v2_integration import override_policy_features",
        label="schema v2 dataset feature schema override",
    )
    _patch(
        factory,
        old=("    if pretrained_path:\n        if isinstance(policy_cfg, GrootConfig):\n"),
        new=(
            "    from so101_contract.lerobot_v2_integration import (\n"
            "        is_target_policy,\n"
            "        prepare_action_representation_context,\n"
            "    )\n"
            "\n"
            "    representation_context = None\n"
            "    if is_target_policy(policy_cfg):\n"
            '        dataset_meta = kwargs.get("dataset_meta")\n'
            '        dataset_stats = kwargs.get("dataset_stats")\n'
            "        if dataset_meta is not None and dataset_stats is not None:\n"
            "            representation_context = prepare_action_representation_context(\n"
            "                policy_cfg,\n"
            "                dataset_meta=dataset_meta,\n"
            "                dataset_stats=dataset_stats,\n"
            '                verify_source_columns=os.environ.get("RANK", "0") == "0",\n'
            "            )\n"
            '            kwargs["dataset_stats"] = representation_context.dataset_stats\n'
            "            if not isinstance(policy_cfg, GrootConfig):\n"
            '                pre_overrides = dict(kwargs.get("preprocessor_overrides") or {})\n'
            '                pre_norm = dict(pre_overrides.get("normalizer_processor") or {})\n'
            '                pre_norm["stats"] = representation_context.dataset_stats\n'
            '                pre_norm["features"] = {\n'
            "                    **policy_cfg.input_features,\n"
            "                    **policy_cfg.output_features,\n"
            "                }\n"
            '                pre_norm["norm_map"] = policy_cfg.normalization_mapping\n'
            '                pre_overrides["normalizer_processor"] = pre_norm\n'
            '                kwargs["preprocessor_overrides"] = pre_overrides\n'
            '                post_overrides = dict(kwargs.get("postprocessor_overrides") or {})\n'
            '                post_norm = dict(post_overrides.get("unnormalizer_processor") or {})\n'
            '                post_norm["stats"] = representation_context.dataset_stats\n'
            '                post_norm["features"] = policy_cfg.output_features\n'
            '                post_norm["norm_map"] = policy_cfg.normalization_mapping\n'
            '                post_overrides["unnormalizer_processor"] = post_norm\n'
            '                kwargs["postprocessor_overrides"] = post_overrides\n'
            "\n"
            "    if pretrained_path:\n"
            "        if isinstance(policy_cfg, GrootConfig):\n"
        ),
        sentinel="representation_context = prepare_action_representation_context(",
        label="policy factory schema v2 context",
    )
    _patch(
        factory,
        old=(
            "            return make_groot_pre_post_processors_from_pretrained(\n"
            "                config=policy_cfg,\n"
            "                pretrained_path=pretrained_path,\n"
            '                dataset_stats=kwargs.get("dataset_stats"),\n'
            '                dataset_meta=kwargs.get("dataset_meta"),\n'
            '                preprocessor_overrides=kwargs.get("preprocessor_overrides"),\n'
            '                postprocessor_overrides=kwargs.get("postprocessor_overrides"),\n'
            "                preprocessor_config_filename=kwargs.get(\n"
            '                    "preprocessor_config_filename", '
            'f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"\n'
            "                ),\n"
            "                postprocessor_config_filename=kwargs.get(\n"
            '                    "postprocessor_config_filename", '
            'f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json"\n'
            "                ),\n"
            "            )\n"
        ),
        new=(
            "            processors = make_groot_pre_post_processors_from_pretrained(\n"
            "                config=policy_cfg,\n"
            "                pretrained_path=pretrained_path,\n"
            '                dataset_stats=kwargs.get("dataset_stats"),\n'
            '                dataset_meta=kwargs.get("dataset_meta"),\n'
            '                preprocessor_overrides=kwargs.get("preprocessor_overrides"),\n'
            '                postprocessor_overrides=kwargs.get("postprocessor_overrides"),\n'
            "                preprocessor_config_filename=kwargs.get(\n"
            '                    "preprocessor_config_filename", '
            'f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"\n'
            "                ),\n"
            "                postprocessor_config_filename=kwargs.get(\n"
            '                    "postprocessor_config_filename", '
            'f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json"\n'
            "                ),\n"
            "            )\n"
            "            processors = _wire_action_representation_steps(\n"
            "                processors[0],\n"
            "                processors[1],\n"
            "                policy_cfg,\n"
            "                representation_context,\n"
            "                pretrained_path,\n"
            "            )\n"
            "            return processors\n"
        ),
        sentinel="            processors = _wire_action_representation_steps(",
        label="GR00T pretrained schema v2 processors",
    )
    _patch(
        factory,
        old=(
            "        _reconnect_relative_absolute_steps(preprocessor, postprocessor)\n"
            "        if isinstance(policy_cfg, Evo1Config):\n"
        ),
        new=(
            "        _reconnect_relative_absolute_steps(preprocessor, postprocessor)\n"
            "        preprocessor, postprocessor = _wire_action_representation_steps(\n"
            "            preprocessor,\n"
            "            postprocessor,\n"
            "            policy_cfg,\n"
            "            representation_context,\n"
            "            pretrained_path,\n"
            "        )\n"
            "        if isinstance(policy_cfg, Evo1Config):\n"
        ),
        sentinel="        preprocessor, postprocessor = _wire_action_representation_steps(",
        label="ACT/SmolVLA pretrained schema v2 processors",
    )
    _patch(
        factory,
        old="    return processors\n\n\ndef make_policy(\n",
        new=(
            "    if representation_context is not None:\n"
            "        from so101_contract.lerobot_v2_integration import (\n"
            "            attach_action_representation_steps,\n"
            "        )\n"
            "\n"
            "        processors = attach_action_representation_steps(\n"
            "            processors[0], processors[1], representation_context\n"
            "        )\n"
            "    return processors\n\n\n"
            "def _wire_action_representation_steps(\n"
            "    preprocessor,\n"
            "    postprocessor,\n"
            "    policy_cfg,\n"
            "    representation_context,\n"
            "    pretrained_path,\n"
            "):\n"
            '    """Checkpoint에서 로드한 pipeline에 schema v2 step을 연결/교체한다."""\n'
            "    from so101_contract.lerobot_v2_integration import (\n"
            "        attach_action_representation_steps,\n"
            "        has_action_representation_steps,\n"
            "        is_target_policy,\n"
            "        reconnect_action_representation_steps,\n"
            "        replace_action_representation_steps,\n"
            "        validate_checkpoint_manifest,\n"
            "    )\n"
            "\n"
            "    if not is_target_policy(policy_cfg):\n"
            "        return preprocessor, postprocessor\n"
            "    has_steps = has_action_representation_steps(preprocessor, postprocessor)\n"
            "    if has_steps:\n"
            "        validate_checkpoint_manifest(pretrained_path, policy_cfg, preprocessor)\n"
            "    if representation_context is not None:\n"
            "        if has_steps:\n"
            "            preprocessor, postprocessor = replace_action_representation_steps(\n"
            "                preprocessor, postprocessor, representation_context\n"
            "            )\n"
            "        else:\n"
            "            preprocessor, postprocessor = attach_action_representation_steps(\n"
            "                preprocessor, postprocessor, representation_context\n"
            "            )\n"
            "    elif not has_steps:\n"
            "        raise ValueError(\n"
            '            "checkpoint has no serialized schema v2 action representation processor; "\n'
            '            "action representation is never inferred"\n'
            "        )\n"
            "    return reconnect_action_representation_steps(preprocessor, postprocessor)\n\n\n"
            "def make_policy(\n"
        ),
        sentinel="def _wire_action_representation_steps(",
        label="fresh attachment and checkpoint wiring helper",
    )


def _patch_pretrained_loader(root: Path) -> None:
    """Pretrained weight load를 selective-reuse/reinit report 경로로 교체."""
    pretrained = root / "policies" / "pretrained.py"
    _patch(
        pretrained,
        old=(
            "        # Load the model with appropriate kwargs\n"
            "        missing_keys, unexpected_keys = load_model_as_safetensor(model, model_file, **kwargs)\n"
            "        log_model_loading_keys(missing_keys, unexpected_keys)\n"
        ),
        new=(
            "        # Load the model with appropriate kwargs.\n"
            "        # representation 전환 시 dimension에 종속된 layer(ACT state projection/action\n"
            "        # head 등)는 shape가 달라진다. stock loader는 이때 죽거나 조용히 넘어가므로\n"
            "        # 재사용/재초기화 목록을 명시적으로 report한다.\n"
            "        from so101_contract.lerobot_v2_integration import (\n"
            "            load_pretrained_with_selective_reuse,\n"
            "        )\n"
            "\n"
            "        report = load_pretrained_with_selective_reuse(\n"
            "            model,\n"
            "            model_file,\n"
            '            device=kwargs.get("device", "cpu"),\n'
            "            strict=strict,\n"
            "        )\n"
            "        log_model_loading_keys(list(report.missing), list(report.unexpected))\n"
        ),
        sentinel="from so101_contract.lerobot_v2_integration import (\n"
        "            load_pretrained_with_selective_reuse,\n"
        "        )",
        label="selective-reuse pretrained weight load",
    )


def _patch_async_server(root: Path) -> None:
    server = root / "async_inference" / "policy_server.py"
    _patch(
        server,
        old="from lerobot.processor import PolicyProcessorPipeline\n",
        new=(
            "from lerobot.processor import PolicyProcessorPipeline\n"
            "from so101_contract.lerobot_v2_integration import "
            "has_action_representation_steps\n"
        ),
        sentinel="from so101_contract.lerobot_v2_integration import "
        "has_action_representation_steps",
        label="async schema v2 detector import",
    )
    _patch(
        server,
        old=(
            "        self.postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None\n"
        ),
        new=(
            "        self.postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None\n"
            "        self._full_chunk_actions = False\n"
        ),
        sentinel="self._full_chunk_actions = False",
        label="async full-chunk mode state",
    )
    _patch(
        server,
        old="        self._full_chunk_actions = False\n",
        new=(
            "        self._full_chunk_actions = False\n"
            "        self._raw_policy_image_features = None\n"
        ),
        sentinel="self._raw_policy_image_features = None",
        label="async raw image schema state",
    )
    _patch(
        server,
        old=(
            '            postprocessor_overrides={"device_processor": device_override},\n'
            "        )\n"
            "\n"
            "        end = time.perf_counter()\n"
        ),
        new=(
            '            postprocessor_overrides={"device_processor": device_override},\n'
            "        )\n"
            "        self._full_chunk_actions = has_action_representation_steps(\n"
            "            self.preprocessor, self.postprocessor\n"
            "        )\n"
            "\n"
            "        end = time.perf_counter()\n"
        ),
        sentinel="self._full_chunk_actions = has_action_representation_steps(",
        label="async full-chunk mode activation",
    )
    _patch(
        server,
        old=(
            "        self._full_chunk_actions = has_action_representation_steps(\n"
            "            self.preprocessor, self.postprocessor\n"
            "        )\n"
            "\n"
            "        end = time.perf_counter()\n"
        ),
        new=(
            "        self._full_chunk_actions = has_action_representation_steps(\n"
            "            self.preprocessor, self.postprocessor\n"
            "        )\n"
            "        # raw observation conversion precedes the rename processor. Build a\n"
            "        # source-key image schema (top/wrist/front) from the model-key schema\n"
            "        # (camera1/2/3) so resizing succeeds before the rename step.\n"
            "        raw_image_features = dict(self.policy_image_features)\n"
            "        for source_key, target_key in policy_specs.rename_map.items():\n"
            "            if target_key not in raw_image_features:\n"
            "                continue\n"
            "            if source_key in raw_image_features and source_key != target_key:\n"
            "                raise ValueError(\n"
            '                    f"camera rename collides with policy image feature: "\n'
            '                    f"{source_key!r} -> {target_key!r}"\n'
            "                )\n"
            "            raw_image_features[source_key] = raw_image_features.pop(target_key)\n"
            "        self._raw_policy_image_features = raw_image_features\n"
            "\n"
            "        end = time.perf_counter()\n"
        ),
        sentinel="self._raw_policy_image_features = raw_image_features",
        label="async source-key image schema",
    )
    _patch(
        server,
        old="            self.policy_image_features,\n",
        new="            self._raw_policy_image_features,\n",
        sentinel="            self._raw_policy_image_features,\n",
        label="async raw image schema use",
    )
    _patch(
        server,
        old=(
            "        elif observations_similar(obs, previous_obs, lerobot_features=self.lerobot_features):\n"
        ),
        new=(
            "        elif not self._full_chunk_actions and observations_similar(\n"
            "            obs, previous_obs, lerobot_features=self.lerobot_features\n"
            "        ):\n"
        ),
        sentinel="elif not self._full_chunk_actions and observations_similar(",
        label="async bypass joint-distance filter for v2 observations",
    )
    _patch(
        server,
        old="        return chunk[:, : self.actions_per_chunk, :]\n",
        new="        return chunk\n",
        sentinel="        return chunk\n",
        label="async preserve full prediction chunk",
    )
    _patch(
        server,
        old=(
            "        # Apply postprocessor (handles unnormalization and device movement)\n"
            "        # Postprocessor expects (B, action_dim) per action, but we have "
            "(B, chunk_size, action_dim)\n"
            "        # So we process each action in the chunk individually\n"
            "        start_postprocess = time.perf_counter()\n"
            "        _, chunk_size, _ = action_tensor.shape\n"
            "\n"
            "        # Process each action in the chunk\n"
            "        processed_actions = []\n"
            "        for i in range(chunk_size):\n"
            "            # Extract action at timestep i: (B, action_dim)\n"
            "            single_action = action_tensor[:, i, :]\n"
            "            processed_action = self.postprocessor(single_action)\n"
            "            processed_actions.append(processed_action)\n"
            "\n"
            "        # Stack back to (B, chunk_size, action_dim), then remove batch dim\n"
            "        action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)\n"
        ),
        new=(
            "        # schema v2 checkpoint는 mode와 무관하게 full chunk를 한 번에 postprocess한\n"
            "        # 뒤 slice한다. 그 외(stock absolute) checkpoint는 기존 step별 동작을 유지한다.\n"
            "        start_postprocess = time.perf_counter()\n"
            "        if self._full_chunk_actions:\n"
            "            action_tensor = self.postprocessor(action_tensor).squeeze(0)\n"
            "            action_tensor = action_tensor[: self.actions_per_chunk]\n"
            "        else:\n"
            "            action_tensor = action_tensor[:, : self.actions_per_chunk, :]\n"
            "            processed_actions = []\n"
            "            for i in range(action_tensor.shape[1]):\n"
            "                single_action = action_tensor[:, i, :]\n"
            "                processed_actions.append(self.postprocessor(single_action))\n"
            "            action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)\n"
        ),
        sentinel="if self._full_chunk_actions:\n"
        "            action_tensor = self.postprocessor(action_tensor).squeeze(0)",
        label="async full-chunk postprocess",
    )


def _patch_sync_inference(root: Path) -> None:
    sync = root / "rollout" / "inference" / "sync.py"
    _patch(
        sync,
        old="from lerobot.processor import PolicyProcessorPipeline\n",
        new=(
            "from lerobot.processor import PolicyProcessorPipeline\n"
            "from so101_contract.lerobot_full_chunk import FullChunkPolicyRunner\n"
            "from so101_contract.lerobot_v2_integration import "
            "has_action_representation_steps\n"
        ),
        sentinel="from so101_contract.lerobot_full_chunk import FullChunkPolicyRunner",
        label="sync full-chunk imports",
    )
    _patch(
        sync,
        old="        self._robot_type = robot_type\n        logger.info(\n",
        new=(
            "        self._robot_type = robot_type\n"
            "        self._full_chunk_runner = None\n"
            "        if has_action_representation_steps(preprocessor, postprocessor):\n"
            "            self._full_chunk_runner = FullChunkPolicyRunner(\n"
            "                policy,\n"
            "                preprocessor,\n"
            "                postprocessor,\n"
            '                execution_horizon=getattr(policy.config, "n_action_steps", None),\n'
            "            )\n"
            "        logger.info(\n"
        ),
        sentinel="self._full_chunk_runner = FullChunkPolicyRunner(",
        label="sync full-chunk runner initialization",
    )
    _patch(
        sync,
        old=(
            "        self._policy.reset()\n"
            "        self._preprocessor.reset()\n"
            "        self._postprocessor.reset()\n"
        ),
        new=(
            "        if self._full_chunk_runner is not None:\n"
            "            self._full_chunk_runner.reset()\n"
            "        else:\n"
            "            self._policy.reset()\n"
            "            self._preprocessor.reset()\n"
            "            self._postprocessor.reset()\n"
        ),
        sentinel="if self._full_chunk_runner is not None:\n"
        "            self._full_chunk_runner.reset()",
        label="sync full-chunk reset",
    )
    _patch(
        sync,
        old=(
            "            observation = self._preprocessor(observation)\n"
            "            action = self._policy.select_action(observation)\n"
            "            action = self._postprocessor(action)\n"
        ),
        new=(
            "            if self._full_chunk_runner is not None:\n"
            "                action = self._full_chunk_runner.next_action(observation)\n"
            "            else:\n"
            "                observation = self._preprocessor(observation)\n"
            "                action = self._policy.select_action(observation)\n"
            "                action = self._postprocessor(action)\n"
        ),
        sentinel="action = self._full_chunk_runner.next_action(observation)",
        label="sync external full-chunk queue",
    )

    evaluation = root / "scripts" / "lerobot_eval.py"
    _patch(
        evaluation,
        old="from lerobot.processor import PolicyProcessorPipeline\n",
        new=(
            "from lerobot.processor import PolicyProcessorPipeline\n"
            "from so101_contract.lerobot_full_chunk import FullChunkPolicyRunner\n"
            "from so101_contract.lerobot_v2_integration import "
            "has_action_representation_steps\n"
        ),
        sentinel="from so101_contract.lerobot_full_chunk import FullChunkPolicyRunner",
        label="eval full-chunk imports",
    )
    _patch(
        evaluation,
        old=(
            "    # Reset the policy and environments.\n"
            "    policy.reset()\n"
            "    observation, info = env.reset(seed=seeds)\n"
        ),
        new=(
            "    # schema v2 checkpoint는 external absolute-action FIFO를 쓴다. select_action은\n"
            "    # cached chunk를 tick마다 re-anchor하므로 사용하지 않는다.\n"
            "    full_chunk_runner = None\n"
            "    if has_action_representation_steps(preprocessor, postprocessor):\n"
            "        full_chunk_runner = FullChunkPolicyRunner(\n"
            "            policy,\n"
            "            preprocessor,\n"
            "            postprocessor,\n"
            '            execution_horizon=getattr(policy.config, "n_action_steps", None),\n'
            "        )\n"
            "        full_chunk_runner.reset()\n"
            "    else:\n"
            "        policy.reset()\n"
            "    observation, info = env.reset(seed=seeds)\n"
        ),
        sentinel="full_chunk_runner = FullChunkPolicyRunner(",
        label="eval full-chunk runner initialization",
    )
    _patch(
        evaluation,
        old=(
            "            observation = preprocessor(observation)\n"
            "            with torch.inference_mode():\n"
            "                action = policy.select_action(observation)\n"
            "            if predicted_latents_callback is not None:\n"
            "                predicted_latents_callback(policy)\n"
            "            action = postprocessor(action)\n"
        ),
        new=(
            "            with torch.inference_mode():\n"
            "                if full_chunk_runner is not None:\n"
            "                    action = full_chunk_runner.next_action(observation)\n"
            "                else:\n"
            "                    observation = preprocessor(observation)\n"
            "                    action = policy.select_action(observation)\n"
            "                    action = postprocessor(action)\n"
            "            if predicted_latents_callback is not None:\n"
            "                predicted_latents_callback(policy)\n"
        ),
        sentinel="action = full_chunk_runner.next_action(observation)",
        label="eval external full-chunk queue",
    )


def _patch_normalizer(root: Path) -> None:
    normalizer = root / "processor" / "normalize_processor.py"
    _patch(
        normalizer,
        old=(
            "            if inverse:\n"
            "                return tensor * std + mean\n"
            "            return (tensor - mean) / denom\n"
        ),
        new=(
            "            if inverse:\n"
            "                # Forward divides by std + eps; use the same denominator for an exact inverse.\n"
            "                return tensor * denom + mean\n"
            "            return (tensor - mean) / denom\n"
        ),
        sentinel="Forward divides by std + eps; use the same denominator for an exact inverse.",
        label="mean/std epsilon-symmetric inverse",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-dir",
        type=Path,
        help="테스트용 lerobot package root; 생략하면 설치된 lerobot==0.6.0",
    )
    args = parser.parse_args()
    root = _lerobot_dir(args.package_dir)
    _patch_configs(root)
    _patch_train(root)
    _patch_factory(root)
    _patch_pretrained_loader(root)
    _patch_normalizer(root)
    _patch_async_server(root)
    _patch_sync_inference(root)
    print(f"[action-representation-patch] PASS: {root}")


if __name__ == "__main__":
    main()
