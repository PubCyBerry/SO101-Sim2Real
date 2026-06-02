#!/usr/bin/env python
"""groot_compat_patch.py — LeRobot 0.5.1 GR00T-N1.5 호환 site-packages 패치.

Dockerfile.policy 빌드(policy-deps 스테이지)에서 `lerobot[smolvla,async]==0.5.1`
설치 직후 1회 실행한다. transformers 5.3 + torch 2.10 조합에서 LeRobot 0.5.1 의
GR00T wrapper 가 네 지점에서 즉시 실패하므로, upstream 반영 전까지 설치된
site-packages 를 최소 수정한다.

■ 패치 지점 (모두 멱등 — 이미 적용됐으면 통과, 형태가 다르면 RuntimeError 로 중단)
  1) FlowmatchingActionHead.__init__ : Beta(...) 기본 validation 이 meta tensor 에서
     Tensor.item() 을 호출 → validate_args=False
  2) FlowmatchingActionHead.sample_time : meta 로 생성된 beta_dist 가 학습 forward 에서도
     meta 로 남음 → 실제 device tensor 로 Beta 재생성
  3) GR00TN15 : transformers 5.3 이 기대하는 all_tied_weights_keys 속성 미정의 → 추가
  4) processor_groot.collate : Eagle processor 의 return_tensors 가 image processor 로
     전달되지 않아 pixel_values 가 list 로 남음 → text_kwargs / images_kwargs 분리 전달

■ 버전 트립와이어
  RuntimeError("Unexpected ...") 가 나면 설치된 lerobot 의 GR00T 코드가 0.5.1 과 달라진
  것이다. lerobot/transformers 를 올렸다면(예: 0.5.2 + transformers 5.4) 본 패치가
  더 이상 필요 없거나 형태가 바뀐 것이므로, 각 패치의 필요성을 재검토할 것.
  (0.5.2 기준: #1·#4 는 upstream 반영됨, #2·#3 은 transformers 버전 동작에 종속.)

■ 호스트(Isaac) 환경에는 적용하지 않는다 — policy-server 이미지 전용.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _lerobot_dir() -> Path:
    """설치된 lerobot 패키지 디렉터리. (Python 버전·venv 경로 하드코딩 회피)"""
    spec = importlib.util.find_spec("lerobot")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("lerobot 패키지를 찾을 수 없습니다 (설치 후 실행해야 함)")
    return Path(next(iter(spec.submodule_search_locations)))


def _patch(path: Path, old: str, new: str, *, sentinel: str, label: str) -> None:
    """old→new 치환. 이미 적용(sentinel 존재)됐으면 통과. 둘 다 없으면 RuntimeError."""
    text = path.read_text()
    if old in text:
        path.write_text(text.replace(old, new))
        print(f"[groot-patch] applied: {label}")
    elif sentinel in text:
        print(f"[groot-patch] already applied: {label}")
    else:
        raise RuntimeError(f"Unexpected GR00T layout — {label} ({path})")


def main() -> None:
    groot = _lerobot_dir() / "policies" / "groot"

    # ── 1) FlowmatchingActionHead: Beta 기본 validation 회피 ──────────────────
    action_head = groot / "action_head" / "flow_matching_action_head.py"
    _patch(
        action_head,
        old="self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)",
        new="self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta, validate_args=False)",
        sentinel="self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta, validate_args=False)",
        label="action_head Beta(validate_args=False)",
    )

    # ── 4) sample_time: 학습 forward 에서 meta tensor 회피 (실제 device 로 재생성) ──
    _patch(
        action_head,
        old=(
            "    def sample_time(self, batch_size, device, dtype):\n"
            "        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)\n"
            "        return (self.config.noise_s - sample) / self.config.noise_s\n"
        ),
        new=(
            "    def sample_time(self, batch_size, device, dtype):\n"
            "        def _as_float(value, fallback):\n"
            "            if isinstance(value, torch.Tensor):\n"
            "                if value.is_meta:\n"
            "                    return fallback\n"
            "                value = value.detach().cpu().item()\n"
            "            return float(value)\n"
            "\n"
            "        alpha = torch.tensor(_as_float(self.config.noise_beta_alpha, 1.5), device=device, dtype=torch.float32)\n"
            "        beta = torch.tensor(_as_float(self.config.noise_beta_beta, 1.0), device=device, dtype=torch.float32)\n"
            "        sample = Beta(alpha, beta, validate_args=False).sample([batch_size]).to(dtype=dtype)\n"
            "        return (self.config.noise_s - sample) / self.config.noise_s\n"
        ),
        sentinel="Beta(alpha, beta, validate_args=False)",
        label="action_head sample_time (meta-safe)",
    )

    # ── 2) GR00TN15: transformers 5.3 이 기대하는 all_tied_weights_keys 추가 ──────
    _patch(
        groot / "groot_n1.py",
        old=(
            "class GR00TN15(PreTrainedModel):\n"
            "    supports_gradient_checkpointing = True\n"
        ),
        new=(
            "class GR00TN15(PreTrainedModel):\n"
            "    supports_gradient_checkpointing = True\n"
            "    all_tied_weights_keys = {}\n"
        ),
        sentinel="all_tied_weights_keys = {}",
        label="GR00TN15.all_tied_weights_keys",
    )

    # ── 3) processor_groot.collate: Eagle processor return_tensors 분리 전달 ──────
    _patch(
        groot / "processor_groot.py",
        old=(
            "            eagle_inputs = eagle_processor(\n"
            "                text=text_list,\n"
            "                images=image_inputs,\n"
            '                images_kwargs={"min_dynamic_tiles": 1, "max_dynamic_tiles": 1, "use_thumbnail": False},\n'
            '                return_tensors="pt",\n'
            "                padding=True,\n"
            "            )\n"
        ),
        new=(
            "            eagle_inputs = eagle_processor(\n"
            "                text=text_list,\n"
            "                images=image_inputs,\n"
            '                text_kwargs={"padding": True, "return_tensors": "pt"},\n'
            "                images_kwargs={\n"
            '                    "min_dynamic_tiles": 1,\n'
            '                    "max_dynamic_tiles": 1,\n'
            '                    "use_thumbnail": False,\n'
            '                    "return_tensors": "pt",\n'
            "                },\n"
            "            )\n"
        ),
        sentinel='"text_kwargs"',
        label="processor_groot collate return_tensors",
    )


if __name__ == "__main__":
    main()
