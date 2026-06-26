"""LeRobot v3 데이터셋을 Hugging Face Hub 에 업로드한다 (Isaac 무의존, 호스트 uv).

recorder(`pick_cube_curobo_demo.py --record_dir` / `rollout_to_lerobot.py`) 산출 폴더를
dataset repo 로 올린다. HF_TOKEN/HF_USER 는 환경변수 또는 레포 루트 `.env` 에서 읽는다.

    uv run python scripts/data/upload_to_huggingface.py \
        --dataset_dir outputs/so101_sim_pick_cube \
        --repo_id taehunkim/so101_sim_pick_cube \
        --commit_message "10 cuRobo sim episodes"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_env_value(key: str) -> str | None:
    """환경변수 우선, 없으면 레포 루트 .env 에서 KEY=VALUE 직접 파싱(변수 보간 미지원)."""
    if os.getenv(key):
        return os.getenv(key)
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == key:
            return v.strip().strip('"').strip("'")
    return None


def main() -> int:
    p = argparse.ArgumentParser(description="Upload LeRobot v3 dataset to Hugging Face Hub")
    p.add_argument("--dataset_dir", required=True, help="로컬 데이터셋 폴더(meta/ data/ videos/)")
    p.add_argument("--repo_id", default=None, help="user/repo. 기본=<HF_USER>/so101_sim_pick_cube")
    p.add_argument("--commit_message", default="upload sim LeRobot v3 dataset")
    p.add_argument("--private", action="store_true", help="private dataset repo 로 생성")
    p.add_argument("--token", default=None, help="HF 토큰(기본=env/.env HF_TOKEN)")
    args = p.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    info = dataset_dir / "meta" / "info.json"
    if not info.is_file():
        print(f"[upload] ✗ 유효한 LeRobot 데이터셋 아님(meta/info.json 없음): {dataset_dir}", flush=True)
        return 1

    token = args.token or _load_env_value("HF_TOKEN")
    if not token:
        print("[upload] ✗ HF_TOKEN 없음(인자/환경/.env 모두 미설정)", flush=True)
        return 1
    hf_user = _load_env_value("HF_USER") or "taehunkim"
    repo_id = args.repo_id or f"{hf_user}/so101_sim_pick_cube"

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    print(f"[upload] repo={repo_id} (dataset) ← {dataset_dir}", flush=True)
    api.create_repo(repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(dataset_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=args.commit_message,
        ignore_patterns=[".cache/*", "**/.ipynb_checkpoints/*"],
    )
    # LeRobot 은 dataset repo 의 codebase_version 태그(예: v3.0)를 revision 으로 찾는다.
    # upload_folder 는 태그를 안 만들므로 info.json 의 버전으로 태그 생성(없으면 train 이 RevisionNotFound).
    import json as _json
    cv = _json.loads(info.read_text(encoding="utf-8")).get("codebase_version", "v3.0")
    # 태그를 최신 main HEAD 로 이동(delete+create). exist_ok 만으론 옛 커밋에 고정돼 재업로드분이 안 보임.
    try:
        try:
            api.delete_tag(repo_id, tag=cv, repo_type="dataset")
        except Exception:
            pass
        api.create_tag(repo_id, tag=cv, repo_type="dataset", revision="main")
        print(f"[upload] codebase_version 태그 최신화: {cv} → main HEAD", flush=True)
    except Exception as exc:
        print(f"[upload] ⚠ 태그({cv}) 갱신 실패: {exc}", flush=True)
    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"[upload] ✓ 완료: {url}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
