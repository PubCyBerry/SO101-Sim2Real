"""IsaacLab HDF5 recorder 의 압축기·청크를 교체하는 file handler.

IsaacLab ``HDF5DatasetFileHandler.write_episode`` 는 모든 배열을 ``compression="gzip"``
(level 4, h5py 자동 청크)으로 쓴다. 카메라 3대 640×480 RGB 를 녹화하면 에피소드당 원본
999 MiB 이고, gzip 은 단일 스레드 blocking 이라 env 하나당 ~10.8 s 씩 심 루프를 세운다.
export 는 env 순차(``RecorderManager.export_episodes``)라 ``--num_envs`` 에 비례해 는다.

실측 (2026-07-28, RTX PRO 5000 Blackwell, e2e 트라이얼에서 뽑은 **실제 렌더 프레임**):

    설정                       MiB/s   압축률   s/demo   1000 ep 디스크
    gzip(4) auto-chunk          123     6.56     8.13      152 GB   ← IsaacLab 기본
    gzip(1) auto-chunk          158     5.50     6.31      182 GB
    lzf     auto-chunk          198     3.70     5.04      270 GB
    lzf     frame-chunk         359     3.26     2.79      306 GB   ← 채택
    none    frame-chunk        2066     1.00     0.48      999 GB

프리셋을 바꿔도 읽은 배열은 비트 동일하다(아래 self-check). 압축기 선택은 **속도 vs 디스크**
트레이드오프일 뿐 데이터 계약과 무관하다 — 변환기 ``scripts/convert/isaaclab2lerobotv3.py``
는 압축 필터에 의존하지 않는다(h5py 가 투명하게 처리).

청크는 이미지에만 손댄다. ``(T,H,W,C)`` 를 프레임 1장 = 청크 1개로 두면 lzf 가 프레임 내부
공간 지역성만 보고 빠르게 돈다. 작은 배열(``applied_target`` ``(T,6)`` 등)은 h5py 자동에
맡긴다 — 청크가 shape 보다 크면 h5py 가 거부한다.

ponytail: ``write_episode`` 를 복사해 재구현하지 않고, 그 호출 동안만
``h5py.Group.create_dataset`` 을 감싼다. 상류가 본문을 바꿔도 따라온다(복사본은 조용히 갈라진다).
대신 상류가 압축 인자 전달 방식을 바꾸면 무력화될 수 있으므로, 산출 HDF5 의 ``.compression``
을 확인하는 것이 회귀 감지 지점이다.

self-check (h5py 만 있으면 됨 — Isaac 불요)::

    python src/sim_to_real/data/hdf5_compression.py

``-m`` 이 아니라 **파일 경로로** 돌린다. ``sim_to_real/__init__.py`` 가 ``isaaclab_tasks`` 를
끌어오기 때문에 패키지 경유 import 는 Isaac 런타임을 요구한다.
"""

from __future__ import annotations

import contextlib
import time

import h5py

#: h5py ``create_dataset`` 압축 kwargs. 키를 바꾸면 그게 곧 정책 변경이다.
COMPRESSION_PRESETS: dict[str, dict] = {
    "none": {},
    "lzf": {"compression": "lzf"},
    "gzip1": {"compression": "gzip", "compression_opts": 1},
    "gzip": {"compression": "gzip"},  # IsaacLab 기본 (level 4)
}


def _image_chunks(data):
    """이미지 ``(T,H,W,C)`` 면 프레임 1장 청크, 아니면 None(h5py 자동)."""
    shape = getattr(data, "shape", ())
    if len(shape) == 4 and shape[0] > 0:
        return (1, *shape[1:])
    return None


@contextlib.contextmanager
def forced_compression(preset: str):
    """블록 안의 모든 ``h5py.Group.create_dataset`` 압축·청크 인자를 ``preset`` 으로 덮는다."""
    kwargs = COMPRESSION_PRESETS[preset]
    original = h5py.Group.create_dataset

    def patched(self, name, *args, **kw):
        kw.pop("compression", None)
        kw.pop("compression_opts", None)
        if kwargs and "chunks" not in kw:
            chunks = _image_chunks(kw.get("data"))
            if chunks is not None:
                kw["chunks"] = chunks
        kw.update(kwargs)
        return original(self, name, *args, **kw)

    h5py.Group.create_dataset = patched
    try:
        yield
    finally:
        h5py.Group.create_dataset = original


def hdf5_handler(preset: str) -> type:
    """``recorders.dataset_file_handler_class_type`` 에 넣을 handler 타입을 만든다.

    IsaacLab 이 이 타입을 **인자 없이** 생성하므로(``RecorderManager.__init__``) 압축 설정을
    클래스에 구워 넣는다. ``isaaclab`` import 는 여기서 지연 — 모듈 자체는 Isaac 없이도
    import 돼야 위 self-check 가 호스트에서 돈다.
    """
    if preset not in COMPRESSION_PRESETS:
        raise ValueError(f"알 수 없는 압축 프리셋 {preset!r} — {sorted(COMPRESSION_PRESETS)}")

    from isaaclab.utils.datasets import HDF5DatasetFileHandler

    class _Handler(HDF5DatasetFileHandler):
        __doc__ = f"HDF5DatasetFileHandler + compression={preset!r} (frame-chunked images)."
        compression_preset = preset

        def write_episode(self, episode, demo_id=None):
            # export 는 env 순차 blocking 이라 트라이얼 벽시간의 큰 축이다. 에피소드당 한 줄이면
            # 압축 프리셋을 바꿨을 때 효과를 로그만 보고 판정할 수 있다.
            t0 = time.perf_counter()
            with forced_compression(preset):
                super().write_episode(episode, demo_id)
            print(f"[export] demo {time.perf_counter() - t0:.2f}s (compression={preset})", flush=True)

    _Handler.__name__ = f"HDF5DatasetFileHandler_{preset}"
    return _Handler


def _self_check():
    """전 프리셋 왕복 배열 동일성 + 이미지 frame-chunk 적용 확인."""
    import tempfile
    from pathlib import Path

    import numpy as np

    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (7, 12, 16, 3), dtype=np.uint8)  # (T,H,W,C) 이미지
    vec = rng.standard_normal((7, 6)).astype(np.float32)        # (T,6) 작은 배열

    with tempfile.TemporaryDirectory() as td:
        for preset in COMPRESSION_PRESETS:
            path = Path(td) / f"{preset}.hdf5"
            with forced_compression(preset), h5py.File(path, "w") as f:
                f.create_dataset("img", data=img, compression="gzip")   # 원래 인자는 무시돼야
                f.create_dataset("vec", data=vec, compression="gzip")
            with h5py.File(path, "r") as f:
                assert np.array_equal(f["img"][:], img), f"{preset}: 이미지 왕복 불일치"
                assert np.array_equal(f["vec"][:], vec), f"{preset}: 벡터 왕복 불일치"
                want = COMPRESSION_PRESETS[preset].get("compression")
                got, chunks = f["img"].compression, f["img"].chunks
                assert got == want, f"{preset}: 압축 {got} != {want}"
                if want is not None:
                    assert chunks == (1, 12, 16, 3), f"{preset}: 이미지 청크 {chunks}"
                    # (T,6) 은 frame-chunk 대상이 아니다 — h5py 자동 청크여야 한다.
                    assert f["vec"].chunks != (1, 6) or vec.shape[0] == 1, f"{preset}: 벡터 청크 강제됨"
            print(f"  {preset:6s} ok  compression={want} chunks={chunks}")

    # 컨텍스트를 벗어나면 원래 h5py 동작으로 복귀해야 한다.
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "restored.hdf5"
        with h5py.File(path, "w") as f:
            f.create_dataset("img", data=img, compression="gzip")
        with h5py.File(path, "r") as f:
            assert f["img"].compression == "gzip", "contextmanager 이탈 후 복구 실패"
    print("  restore ok  (contextmanager 이탈 후 h5py 원복)")


if __name__ == "__main__":
    _self_check()
    print("SELFCHECK_OK")
