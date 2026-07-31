#!/usr/bin/env python3
"""3D Gaussian Splatting ``.ply`` → **NuRec USDZ** (Isaac Sim 이 그대로 레이트레이싱).

Isaac Sim 5.1(Kit 107.3) RTX 렌더러는 NuRec volume 을 **내장 지원**한다
(``librtx.scenedb.plugin.so`` 의 ``NuRecVolume``). 확장(extension) 이름으로는 안 잡히니
파일명 검색으로 "미지원" 이라 결론내지 말 것. 3dgrut 문서 기준 지원 범위 =
Kit 107.3~110.1 / Isaac Sim 5.0~6.0.

변환 자체는 **3dgrut 의 ``NuRecExporter`` 를 그대로 쓴다**(포맷 정본은 NVIDIA 쪽).
이 스크립트가 더하는 건 두 가지뿐:

1. ``MixtureOfGaussians.__init__`` 이 무조건 만드는 **tracer 를 스텁으로 대체**.
   tracer 는 slang 커널 JIT(``slangc``)를 요구하는데 export 경로에서 한 번도 호출되지
   않는다. 이거 없으면 변환하려고 CUDA/slang 툴체인 전체를 깔아야 한다.
2. ``export_cameras=False``. 학습 checkpoint 가 아니라 맨 ply 에서 시작하므로 카메라
   포즈/내부파라미터를 가진 dataset 이 없다.

■ 선행 (일회성 오프라인 변환 — 런타임 의존성 아님, 루트 pyproject 핀 건드리지 말 것)

    git clone https://github.com/nv-tlabs/3dgrut ref_repos/3dgrut   # 이미 있으면 생략
    uv pip install --python .venv/bin/python --target scratch/pydeps \
        msgpack plyfile hydra-core omegaconf rich "nvidia-ncore>=19.0.0" \
        kornia simplejpeg tensorboard scikit-learn

■ 실행

    PYTHONPATH=$PWD/scratch/pydeps:$PWD/ref_repos/3dgrut \
    .venv/bin/python scripts/convert/gsplat_ply_to_nurec_usdz.py \
        assets/test_3dgs_background.ply --output assets/scenes/scan_nurec.usdz

■ 좌표계 — **이 스크립트는 프레임을 손대지 않는다.**
  폰 스캔 ply 는 보통 Y-up 이고 NuRec USDZ 는 stage 를 Z-up 으로 선언하지만 gaussian
  좌표는 원본 그대로다(``xformOp:transform`` = identity). 세우는 건 배치 단계
  (``SO101_BACKGROUND_RPY_DEG`` 의 roll=90)가 담당한다 — 3dgrut 출력 byte 는 불변.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _stub_tracers() -> None:
    """threedgrut import 전에 렌더러를 자리채움으로 바꾼다 (slangc 불요)."""
    import threedgrt_tracer
    import threedgut_tracer

    class _NoTracer:
        def __init__(self, conf):
            self.conf = conf

    threedgrt_tracer.Tracer = _NoTracer
    threedgut_tracer.Tracer = _NoTracer


def verify_nurec_usdz(path: Path) -> dict[str, object]:
    """생성된 USDZ 가 실제로 NuRec volume 인지 구조 검증. 실패 시 AssertionError."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    assert stage, f"{path}: USD stage 열기 실패"

    volumes = [p for p in stage.Traverse()
               if p.GetAttribute("omni:nurec:isNuRecVolume").Get() is True]
    assert volumes, f"{path}: omni:nurec:isNuRecVolume prim 없음 — NuRec 자산 아님"

    fields = [p for p in stage.Traverse() if p.GetTypeName() == "OmniNuRecFieldAsset"]
    names = {p.GetAttribute("fieldName").Get() for p in fields}
    assert {"density", "emissiveColor"} <= names, f"{path}: field 누락 {names}"
    for f in fields:
        ref = f.GetAttribute("filePath").Get()
        assert ref is not None, f"{path}: {f.GetPath()} filePath 없음"

    return {
        "volume_path": str(volumes[0].GetPath()),
        "fields": sorted(names),
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("input", type=Path, help="3DGS .ply")
    ap.add_argument("--output", type=Path, required=True, help="출력 .usdz")
    ap.add_argument("--config-name", default="apps/colmap_3dgut.yaml",
                    help="3dgrut hydra config (render.method 가 3dgut/3dgrt 여야 한다)")
    ap.add_argument("--config-dir", default="", help="기본값 = ref_repos/3dgrut/configs")
    ap.add_argument("--verify-only", action="store_true",
                    help="변환 없이 기존 --output USDZ 구조만 검증")
    args = ap.parse_args(argv)

    if args.verify_only:
        print(f"[verify] {args.output}: {verify_nurec_usdz(args.output)}")
        return 0

    if not args.input.is_file():
        raise SystemExit(f"입력 없음: {args.input}")

    _stub_tracers()

    from hydra.compose import compose
    from hydra.initialize import initialize_config_dir

    from threedgrut.export.usd.nurec.exporter import NuRecExporter
    from threedgrut.model.model import MixtureOfGaussians

    config_dir = args.config_dir or str(Path(__file__).resolve().parents[2] / "ref_repos/3dgrut/configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        conf = compose(config_name=args.config_name)

    model = MixtureOfGaussians(conf)
    model.init_from_ply(str(args.input), init_model=False)
    n = int(model.get_positions().shape[0])
    print(f"[ply→nurec] {args.input}: gaussians={n} "
          f"sh_active={model.get_n_active_features()} sh_max={model.get_max_n_features()}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    NuRecExporter(export_cameras=False).export(model, args.output, dataset=None, conf=conf)

    info = verify_nurec_usdz(args.output)
    print(f"[ply→nurec] wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")
    print(f"[ply→nurec] verify PASS {info}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
