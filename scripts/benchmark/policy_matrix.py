#!/usr/bin/env python3
"""벤치마크 대상 policy 7종 × 2 mode 의 실행 인자 단일 소스.

`sweep_train_cost.py` 가 이 표만 읽는다. 인자·베이스 체크포인트를 바꿀 일이 생기면 여기만 고친다.

mode 정의
---------
``train``     : 무작위 초기화에서 시작(``--policy.type=<X>``). 백본이 pretrained 를 강제로
                내려받는 정책(vla_jepa 의 Qwen3-VL·V-JEPA2, groot 의 N1.7-3B)은 구조적으로
                순수 scratch 가 불가능하다 — 표의 ``train_note`` 에 실제 의미를 적었다.
``finetune``  : 공개 base 체크포인트에서 시작(``--policy.path=...``). config 도 체크포인트에서
                오므로 freeze 플래그·해상도·dtype 이 발행자 기준으로 바뀐다.
                → 두 mode 의 비용 차이는 대부분 (a) trainable 파라미터 집합,
                  (b) dtype, (c) 입력 해상도 에서 온다.

`docker/policy-entrypoint.sh train` 이 env var 로 조립하는 인자(action_representation 등)는
정책별로 존재 여부가 다르다. ``has_action_representation=False`` 인 정책에 그 플래그가 붙으면
draccus 가 unrecognized argument 로 죽으므로 sweep 이 빈 env 로 눌러야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModeSpec:
    """한 (policy, mode) 조합의 lerobot-train 인자."""

    policy_type: str | None = None  # --policy.type (None 이면 --policy.path 만)
    policy_path: str | None = None  # --policy.path (HF repo id 또는 컨테이너 내부 경로)
    base_model_path: str | None = None  # --policy.base_model_path (GR00T native base)
    # config 필드 override. lerobot-train 에는 `--policy.<key>=<value>` 로,
    # 추론 측정기에는 dict 로 그대로 전달된다 (한 곳만 고치면 둘 다 반영).
    overrides: dict[str, Any] = field(default_factory=dict)
    # dataset 카메라 키 → 체크포인트가 기대하는 키. `--rename_map` 으로 전달된다.
    # 이걸 주면 lerobot 의 visual feature 일치 검사가 **건너뛰어지기도** 한다
    # (`make_policy`: `if not rename_map: validate_visual_features_consistency(...)`).
    rename_map: dict[str, str] = field(default_factory=dict)
    seed_from_self: bool = False  # 공개 base 가 없어 1-step 자가 체크포인트에서 출발
    note: str = ""


@dataclass(frozen=True)
class PolicySpec:
    label: str
    train: ModeSpec
    finetune: ModeSpec
    has_action_representation: bool  # so101 patch 가 이 config 에 필드를 넣었는지
    approx_params: str
    input_resolution: str
    frames_per_sample: str
    # 파라미터 1개가 실제로 차지하는 평균 바이트 (AdamW 정적 VRAM 추정용).
    # accelerate 의 mixed_precision 은 **파라미터 dtype 을 바꾸지 않는다** — bf16 autocast 는
    # 연산만 bf16 이고 가중치·grad·optimizer state 는 파라미터 dtype 그대로다.
    # 4.0 = fp32 파라미터, 2.0 = 정책이 스스로 bf16 으로 캐스팅, 그 사이 = 혼합.
    param_bytes: float
    downloads: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()

    def mode(self, name: str) -> ModeSpec:
        if name == "train":
            return self.train
        if name == "finetune":
            return self.finetune
        raise KeyError(f"unknown mode {name!r}")


# 2-cam dummy 데이터셋의 카메라 키 (make_dummy_dataset.py 기본값과 일치해야 한다)
CAMERA_KEYS = ("observation.images.top", "observation.images.wrist")
# LingBot: 카메라 키 + 6D joint action(arm 5 + gripper) → 30ch action 공간 중 앞 6채널만 사용.
# 학습 forward 는 attn_mode="flex" 를 요구한다.
_LINGBOT_OVERRIDES: dict[str, Any] = {
    "attn_mode": "flex",
    "obs_cam_keys": list(CAMERA_KEYS),
    "used_action_channel_ids": list(range(6)),
}
# VLA-JEPA base 는 action 7D·state 8D 로 학습됐다 → 6D 데이터셋에 붙이려면 두 projection 재초기화
# X-VLA 는 `florence_config` 가 비어 있으면 "vision_config is required" 로 죽는다.
# 기본값 `{}` 이라 scratch 학습에는 반드시 넣어야 하고, 공개 체크포인트 중에도
# 비어 있는 것(jadechoghari/X-VLA-Libero)이 있어 못 쓴다.
# 아래 파일은 lerobot 포맷 SO-101 체크포인트(imstevenpmwork/xvla_super_chatton_2)에서 복사했다.
def _cam_map(first: str, second: str) -> dict[str, str]:
    """top/wrist → 체크포인트 카메라 키 2개. 남는 슬롯은 정책이 0-pad·mask 로 처리한다."""
    return {
        "observation.images.top": f"observation.images.{first}",
        "observation.images.wrist": f"observation.images.{second}",
    }


_XVLA_FLORENCE_DICT: dict[str, Any] = json.loads(
    (Path(__file__).with_name("xvla_florence_config.json")).read_text(encoding="utf-8")
)
# action 7D·state 8D 로 학습된 base → 6D 데이터셋에 붙이려면 세 projection 전부 재초기화.
# (encoder/state 만 넣으면 `action_decoder.layer2.bias` (7,) vs (6,) 에서 걸린다.)
_VLA_JEPA_REINIT = [
    "model.action_model.action_encoder",
    "model.action_model.state_encoder",
    "model.action_model.action_decoder",
]

POLICIES: dict[str, PolicySpec] = {
    "act": PolicySpec(
        label="ACT",
        param_bytes=4.0,  # fp32 (dtype 필드 없음 → accelerate autocast 만)
        approx_params="~51M (ResNet18×2 + enc4/dec1 transformer)",
        input_resolution="리사이즈 없음 → 480×640 원본",
        frames_per_sample="obs 1 / action chunk 100",
        has_action_representation=True,
        train=ModeSpec(policy_type="act", note="정상적인 from-scratch (ImageNet backbone 만 pretrained)"),
        finetune=ModeSpec(
            seed_from_self=True,
            note="SO-101 6D 공개 ACT base 가 없어 1-step 자가 체크포인트에서 출발. "
            "구조가 같으므로 비용은 train 과 동일해야 한다(대조군)",
        ),
        caveats=(
            "유일하게 이미지 리사이즈가 없다 → 480×640 이 그대로 backbone 에 들어가 "
            "다른 정책(224~518 정사각)과 픽셀 수가 다르다",
        ),
    ),
    "smolvla": PolicySpec(
        label="SmolVLA",
        param_bytes=4.0,  # fp32 파라미터 + bf16 autocast
        approx_params="~0.45B (SmolVLM2-500M 16층 + 0.75× action expert)",
        input_resolution="512×512 pad-resize",
        frames_per_sample="obs 1 / action chunk 50",
        has_action_representation=True,
        downloads=("HuggingFaceTB/SmolVLM2-500M-Video-Instruct (train 모드 config/tokenizer)",),
        train=ModeSpec(
            policy_type="smolvla",
            note="load_vlm_weights=False → VLM 도 무작위. train_expert_only=True 가 기본이라 "
            "학습 파라미터는 action expert 로 한정된다",
        ),
        finetune=ModeSpec(
            policy_path="lerobot/smolvla_base",
            note="base config: load_vlm_weights=True·freeze_vision_encoder=True·train_expert_only=True",
        ),
    ),
    "pi05": PolicySpec(
        label="Pi05 (π0.5)",
        param_bytes=2.0,  # --policy.dtype=bfloat16 강제 → 파라미터 자체가 bf16
        approx_params="~3.6B (PaliGemma gemma_2b + gemma_300m expert)",
        input_resolution="224×224",
        frames_per_sample="obs 1 / action chunk 50",
        has_action_representation=False,
        downloads=("lerobot/pi05_base (14.5GB, fp32)",),
        train=ModeSpec(
            policy_type="pi05",
            overrides={"dtype": "bfloat16"},
            note="dtype 기본값 float32 는 48GB 카드에서 batch 1 도 불가(가중치+grad+Adam 58GB) → bf16 고정",
        ),
        finetune=ModeSpec(
            policy_path="lerobot/pi05_base",
            overrides={"dtype": "bfloat16"},
            rename_map=_cam_map("base_0_rgb", "left_wrist_0_rgb"),
            note="base config dtype=float32 을 bf16 으로 덮어써야 GPU 에 올라간다",
        ),
        caveats=(
            "OOM 이면 --policy.gradient_checkpointing=true 가 1차 완화 knob (config 에 있음)",
            "전체 3.6B 가 trainable (freeze_vision_encoder·train_expert_only 기본 False)",
        ),
    ),
    "groot": PolicySpec(
        label="GR00T-N1.7",
        param_bytes=4.0,  # model_params_fp32=True 가 기본 (bf16 autocast 와 병행)
        approx_params="~3.4B (Eagle VLM backbone + flow-matching action head)",
        input_resolution="256×256 (backbone image processor)",
        frames_per_sample="obs 1 / action chunk 40",
        has_action_representation=True,
        downloads=("nvidia/GR00T-N1.7-3B (6.9GB)",),
        train=ModeSpec(
            policy_type="groot",
            base_model_path="nvidia/GR00T-N1.7-3B",
            overrides={"tune_llm": True, "tune_visual": True},
            note="lerobot GrootConfig 는 base_model_path 기본값이 N1.7-3B 라 무작위 초기화 경로가 "
            "없다 → 'train' 은 **full-tune**(LLM+vision 까지 학습)으로 정의한다",
        ),
        finetune=ModeSpec(
            policy_type="groot",
            base_model_path="nvidia/GR00T-N1.7-3B",
            note="기본 freeze 정책: tune_projector+tune_diffusion_model+tune_vlln 만 True",
        ),
        caveats=(
            "model_params_fp32=True + use_bf16=True 조합이 기본 (fp32 마스터 + bf16 autocast)",
            "--policy.compile_model 미지원 (entrypoint 가 자동으로 건너뜀)",
            "LoRA knob 보유: --policy.lora_rank>0 (OOM 완화용, 기본 0)",
        ),
    ),
    "xvla": PolicySpec(
        label="X-VLA",
        param_bytes=2.0,  # --policy.dtype=bfloat16
        approx_params="~0.9B (Florence2 VLM + 24층 1024d soft-prompt DiT)",
        input_resolution="224×224 (양쪽 동일하게 맞춤)",
        frames_per_sample="obs 1 / action chunk 32",
        has_action_representation=False,
        downloads=("facebook/bart-large tokenizer",),
        train=ModeSpec(
            policy_type="xvla",
            overrides={
                "dtype": "bfloat16",
                "resize_imgs_with_padding": [224, 224],
                "florence_config": _XVLA_FLORENCE_DICT,
            },
            note="florence_config 가 필수다(기본 {} → vision_config is required). finetune 베이스에서 "
            "복사해 해상도까지 224 로 맞춰 두 mode 를 비교 가능하게 했다",
        ),
        finetune=ModeSpec(
            seed_from_self=True,
            overrides={
                "dtype": "bfloat16",
                "resize_imgs_with_padding": [224, 224],
                "florence_config": _XVLA_FLORENCE_DICT,
            },
            note="lerobot 0.6.0 과 호환되는 공개 X-VLA 체크포인트가 없다 — 셋 다 어긋난다: "
            "jadechoghari/X-VLA-Libero(action head 7D, 0.6.0 은 franka_joint7=20D) · "
            "imstevenpmwork/xvla_super_chatton_2(신형 VLM 키 multi_modal_projector.*) · "
            "2toINF/X-VLA-*(upstream 포맷, key prefix 'model.' 없음). "
            "비용 측정이 목적이므로 ACT 처럼 1-step 자가 체크포인트에서 fine-tune 한다 "
            "(아키텍처·trainable 집합이 같아 비용은 실제 fine-tune 과 동일하다)",
        ),
        caveats=(
            "XVLAPolicy.from_pretrained 이 strict=True 로 load_state_dict 한다 → 공개 체크포인트의 "
            "키 이름·action head 폭이 조금이라도 다르면 즉시 실패 (0.6.0 에서 franka_joint7 이 "
            "7D→20D 로 바뀌었다: action_hub.FrankaJoint7ActionSpace.dim_action=20)",
            "florence_config 는 2toINF/X-VLA-Pt 와 imstevenpmwork 체크포인트가 완전히 동일 "
            "(diff 0) — 구현 차이는 키 이름뿐이다",
        ),
    ),
    "vla_jepa": PolicySpec(
        label="VLA-JEPA",
        param_bytes=2.2,  # Qwen3-VL·V-JEPA2 는 bf16 로드, action DiT 는 fp32 → 혼합
        # (VLA-JEPA-Pretrain safetensors 6.16GB / 2.77B param = 2.22 B/param 실측)
        approx_params="~3.1B (Qwen3-VL-2B + V-JEPA2 ViT-L + DiT-B 16층 action model)",
        input_resolution="Qwen 프로세서 기본 / V-JEPA2 256",
        frames_per_sample="obs 8 프레임 clip / action chunk 7",
        has_action_representation=False,
        downloads=(
            "lerobot/VLA-JEPA-Pretrain (6.2GB)",
            "Qwen/Qwen3-VL-2B-Instruct (~4.5GB)",
            "facebook/vjepa2-vitl-fpc64-256 (~1.2GB)",
        ),
        train=ModeSpec(
            policy_type="vla_jepa",
            note="action model 만 무작위. Qwen3-VL·V-JEPA2 백본은 __init__ 에서 pretrained 를 "
            "무조건 내려받는다 → 완전한 scratch 아님",
        ),
        finetune=ModeSpec(
            policy_path="lerobot/VLA-JEPA-Pretrain",
            overrides={"reinit_modules": _VLA_JEPA_REINIT},
            rename_map=_cam_map("exterior_1_left", "exterior_2_left"),
            note="base 는 action 7D·state 8D → 6D 데이터셋과 shape 이 달라 두 projection 재초기화 필수",
        ),
        caveats=(
            "repeated_diffusion_steps=8 → 실효 배치가 batch×8 이다. 다른 정책과 배치 숫자를 "
            "그대로 비교하면 안 된다",
            "샘플마다 8프레임 × 2 cam = 16 프레임 decode → dataloader 부하가 가장 크다",
            "freeze_qwen=False 기본 → 2B backbone 이 trainable",
        ),
    ),
    "lingbot_va": PolicySpec(
        label="LingBot-VA",
        param_bytes=2.0,  # transformer 를 config.dtype(bf16) 로 캐스팅
        approx_params="~5.1B trainable transformer + 프리즈 Wan VAE(2.8GB)·UMT5-XXL(11.4GB)",
        input_resolution="train: 128×128 / finetune: 256×320 (체크포인트 값)",
        frames_per_sample="train: 16프레임 clip / finetune: 8프레임 clip",
        has_action_representation=False,
        downloads=(
            "lerobot/lingbot_va_base (10.2GB, bf16 transformer)",
            "robbyant/lingbot-va-base 의 vae/·text_encoder/·tokenizer/ 만 (~14.3GB; "
            "transformer/ 10GB 는 불필요)",
        ),
        train=ModeSpec(
            policy_type="lingbot_va",
            overrides=_LINGBOT_OVERRIDES,
            note="Wan2.2 dual-stream transformer 를 무작위 초기화. 학습에는 attn_mode=flex 필수",
        ),
        finetune=ModeSpec(
            policy_path="lerobot/lingbot_va_base",
            # rename 후의 키로 obs_cam_keys 를 맞춰야 VAE 입력이 잡힌다
            overrides={
                **_LINGBOT_OVERRIDES,
                "obs_cam_keys": [
                    "observation.images.cam_high",
                    "observation.images.cam_left_wrist",
                ],
            },
            rename_map=_cam_map("cam_high", "cam_left_wrist"),
            note="base 는 14D action(aloha) → used_action_channel_ids 를 6 으로 줄인다 "
            "(action_dim=30 고정이라 가중치 shape 은 그대로)",
        ),
        caveats=(
            "학습 forward 가 frozen VAE + UMT5-XXL 를 매 스텝 호출한다. text_encoder_device=cpu "
            "기본값이라 CPU 인코딩이 스텝 시간에 그대로 들어간다",
            "VRAM 은 5.1B transformer + VAE(GPU) 기준 — 48GB 에서 배치를 크게 못 잡을 것",
            "attn_mode=flex 없이 학습하면 forward 가 거부한다",
        ),
    ),
}

MODES = ("train", "finetune")
DEFAULT_BATCH_SIZES = (1, 2, 4, 8, 16, 32)

# sweep 전에 미리 받아둘 가중치 (repo_id, hf download --include 패턴).
# GPU 를 전혀 쓰지 않으므로 GPU 경합 중에도 실행 가능하다. 총 ~62GB.
PREFETCH: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("lerobot/smolvla_base", ()),
    ("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", ()),
    ("lerobot/pi05_base", ()),
    # Pi05/Pi0 processor 가 하드코딩한 **gated** tokenizer. 가중치는 필요 없다.
    # 401 이면 HF 계정으로 모델 페이지에서 라이선스를 수락해야 한다.
    ("google/paligemma-3b-pt-224", ("*.json", "*.model", "tokenizer*")),
    ("nvidia/GR00T-N1.7-3B", ()),
    ("facebook/bart-large", ()),
    ("lerobot/VLA-JEPA-Pretrain", ()),
    ("Qwen/Qwen3-VL-2B-Instruct", ()),
    ("facebook/vjepa2-vitl-fpc64-256", ()),
    ("lerobot/lingbot_va_base", ()),
    # frozen VAE + UMT5-XXL + tokenizer 만 필요. transformer/ 10GB 는 lerobot 체크포인트가 대신한다.
    ("robbyant/lingbot-va-base", ("vae/*", "text_encoder/*", "tokenizer/*")),
)


def format_cli_value(value: Any) -> str:
    """draccus 가 받을 수 있는 형태로 직렬화. 문자열은 따옴표 없이, 나머지는 compact JSON."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, separators=(",", ":"))


def policy_args(name: str, mode: str, *, seed_checkpoint: str | None = None) -> list[str]:
    """(policy, mode) → lerobot-train 에 붙일 --policy.* 인자 목록."""
    spec = POLICIES[name].mode(mode)
    args: list[str] = []
    if spec.seed_from_self:
        if not seed_checkpoint:
            raise ValueError(f"{name}/{mode} 는 자가 seed 체크포인트 경로가 필요하다")
        args.append(f"--policy.path={seed_checkpoint}")
    elif spec.policy_type:
        args.append(f"--policy.type={spec.policy_type}")
        if spec.base_model_path:
            args.append(f"--policy.base_model_path={spec.base_model_path}")
    elif spec.policy_path:
        args.append(f"--policy.path={spec.policy_path}")
    else:
        raise ValueError(f"{name}/{mode} 에 출발 모델이 정의되지 않았다")
    args.extend(f"--policy.{key}={format_cli_value(value)}" for key, value in spec.overrides.items())
    return args


def policy_spec_json(name: str, mode: str, *, seed_checkpoint: str | None = None) -> dict[str, Any]:
    """(policy, mode) → 추론 측정기(`measure_inference_cost.py`)에 넘길 spec dict."""
    spec = POLICIES[name].mode(mode)
    overrides = dict(spec.overrides)
    payload: dict[str, Any] = {"overrides": overrides, "rename_map": dict(spec.rename_map)}
    if spec.seed_from_self:
        if not seed_checkpoint:
            raise ValueError(f"{name}/{mode} 는 자가 seed 체크포인트 경로가 필요하다")
        payload["path"] = seed_checkpoint
    elif spec.policy_type:
        payload["type"] = spec.policy_type
        if spec.base_model_path:
            overrides["base_model_path"] = spec.base_model_path
    else:
        payload["path"] = spec.policy_path
    return payload


def _self_check() -> None:
    for name, spec in POLICIES.items():
        for mode in MODES:
            args = policy_args(name, mode, seed_checkpoint="/tmp/seed")
            has_type = any(a.startswith("--policy.type=") for a in args)
            has_path = any(a.startswith("--policy.path=") for a in args)
            # lerobot parser 는 --policy.path 와 --policy.type 동시 지정을 금지한다
            assert has_type != has_path, f"{name}/{mode}: type/path 동시 지정 또는 둘 다 없음 → {args}"
            assert args, f"{name}/{mode}: 빈 인자"
    assert set(MODES) == {"train", "finetune"}
    assert len(POLICIES) == 7, f"대상 정책 7종이어야 한다 (현재 {len(POLICIES)})"
    print(f"self-check PASS — {len(POLICIES)} policies × {len(MODES)} modes")


def main() -> None:
    _self_check()
    print()
    header = f"{'policy':<11} {'mode':<9} args"
    print(header)
    print("-" * len(header))
    for name in POLICIES:
        for mode in MODES:
            args = " ".join(policy_args(name, mode, seed_checkpoint="<seed>"))
            print(f"{name:<11} {mode:<9} {args}")
    print()
    for name, spec in POLICIES.items():
        print(f"■ {spec.label} ({name})")
        print(f"    params      : {spec.approx_params}")
        print(f"    resolution  : {spec.input_resolution}")
        print(f"    frames      : {spec.frames_per_sample}")
        print(f"    action_repr : {spec.has_action_representation}")
        for d in spec.downloads:
            print(f"    download    : {d}")
        for c in spec.caveats:
            print(f"    ⚠ {c}")


if __name__ == "__main__":
    main()
