#!/usr/bin/env python3
"""VLA/WAM 정책 학습 비용 스윕 — policy 7종 × mode 2종 × batch 6종.

각 조합마다 `policy-server:0.6.0` 컨테이너의 `train` 모드(= lerobot-train)를 짧게(기본 30스텝)
돌려 **정상 상태 1스텝 시간**과 **VRAM 피크**를 재고, 200k 스텝으로 외삽한다.

측정값 출처
-----------
* ``updt_s`` / ``data_s`` / ``mem_gb`` — lerobot 이 이미 로그로 찍는 값
  (`lerobot/scripts/lerobot_train.py`: `update_policy()` 가 스텝마다
  `torch.cuda.reset_peak_memory_stats()` 후 `max_memory_allocated()` 를 기록).
  ``--log_freq`` 창 단위 평균이라 **마지막 창**을 정상 상태로 채택한다(첫 창은 워밍업).
* ``nvsmi_peak_mb`` — 호스트에서 `nvidia-smi --query-compute-apps` 를 폴링해 얻은
  **프로세스 단위** 실제 점유(= allocator reserved + CUDA context). 다른 프로세스가 GPU 를
  쓰고 있어도 오염되지 않는다. ``mem_gb`` 는 tensor allocated 라 이 값보다 항상 작다.

GPU 경합
--------
다른 작업(Isaac Sim 등)이 같은 GPU 를 쓰는 중이면 **시간 측정이 오염된다**(VRAM 은 무관).
`--require-free-mb` 로 최소 여유 VRAM 을, `--require-idle-util` 로 최대 타 프로세스 사용률을
게이트할 수 있다. 기본은 경고만 하고 진행한다.

사용
----
.. code-block:: bash

    # 0) 가중치 미리 받기 (GPU 불필요, ~62GB)
    python3 scripts/benchmark/sweep_train_cost.py --prefetch

    # 1) dry-run: 실행될 docker 명령 확인
    python3 scripts/benchmark/sweep_train_cost.py --dry-run

    # 2) 실제 스윕 (재실행하면 이미 끝난 조합은 건너뜀)
    python3 scripts/benchmark/sweep_train_cost.py --steps 30

    # 3) 리포트
    python3 scripts/benchmark/sweep_train_cost.py --report
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

from policy_matrix import (  # noqa: E402
    DEFAULT_BATCH_SIZES,
    MODES,
    POLICIES,
    PREFETCH,
    policy_args,
    policy_spec_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE = "policy-server:0.6.0"
DEFAULT_DATASET_ROOT = Path("/DISK1/so101-sim2real/datasets/benchmark_dummy_2cam")
DEFAULT_DATASET_REPO_ID = "local/so101_benchmark_dummy_2cam"
DEFAULT_WORKDIR = Path("/DISK1/so101-sim2real/benchmark")
TARGET_STEPS = 200_000

_METRIC_RE = re.compile(r"([A-Za-z_][\w/]*):(-?\d+\.?\d*(?:[eE][-+]?\d+)?)")


def hf_token() -> str | None:
    """환경변수 → 상위 디렉터리의 `.env` 순으로 HF_TOKEN 을 찾는다.

    gated repo(`google/paligemma-3b-pt-224` — Pi05/Pi0 tokenizer)가 있어 토큰 없이는
    일부 조합이 401 로 실패한다. worktree 에는 `.env` 가 없으므로 위로 걸어 올라간다.
    """
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    for parent in [REPO_ROOT, *list(REPO_ROOT.parents)[:5]]:
        env_file = parent / ".env"
        if not env_file.exists():
            continue
        for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("HF_TOKEN="):
                value = line.split("=", 1)[1].strip().strip("'\"")
                if value:
                    return value
    return None


# ── GPU 상태 ──────────────────────────────────────────────────────────────────
def _nvidia_smi(query: str, extra: list[str] | None = None) -> list[str]:
    cmd = ["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits", *(extra or [])]
    out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if out.returncode != 0:
        return []
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def gpu_snapshot() -> dict[str, float]:
    rows = _nvidia_smi("gpu=memory.total,memory.used,utilization.gpu")
    if not rows:
        return {}
    total, used, util = (float(x) for x in rows[0].split(","))
    return {"total_mb": total, "used_mb": used, "free_mb": total - used, "util_pct": util}


def compute_app_pids() -> dict[int, float]:
    """{host pid: used MiB} — GPU 를 쓰는 프로세스."""
    result: dict[int, float] = {}
    for row in _nvidia_smi("compute-apps=pid,used_gpu_memory"):
        pid_s, mem_s = (x.strip() for x in row.split(","))
        try:
            result[int(pid_s)] = float(mem_s)
        except ValueError:
            continue
    return result


class VramPoller(threading.Thread):
    """실행 중 새로 뜬 GPU 프로세스의 점유 피크를 추적한다 (동거 프로세스와 분리)."""

    def __init__(self, interval: float = 1.0) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.baseline = set(compute_app_pids())
        self.peak_mb = 0.0
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            apps = compute_app_pids()
            mine = sum(mb for pid, mb in apps.items() if pid not in self.baseline)
            self.peak_mb = max(self.peak_mb, mine)
            self._stop.wait(self.interval)

    def stop(self) -> float:
        self._stop.set()
        self.join(timeout=5)
        return self.peak_mb


# ── docker 명령 조립 ──────────────────────────────────────────────────────────
def redacted(cmd: list[str]) -> str:
    """로그·콘솔 출력용 — HF_TOKEN 값을 가린다."""
    return " ".join("HF_TOKEN=<redacted>" if part.startswith("HF_TOKEN=") else part for part in cmd)


def container_name(job_name: str) -> str:
    # job_name 이 이미 bench_ 로 시작하면 두 번 붙이지 않는다
    return job_name if job_name.startswith("bench_") else f"bench_{job_name}"


def docker_kill(job_name: str) -> None:
    """`docker run` 클라이언트를 죽여도 **컨테이너는 계속 돈다** — 이름으로 확실히 죽인다.

    이걸 빼면 timeout·marker 조기종료 때 좀비 컨테이너가 남아 GPU/CPU 를 물고
    다음 조합 측정을 오염시킨다.
    """
    subprocess.run(
        ["docker", "rm", "-f", container_name(job_name)],
        capture_output=True,
        check=False,
    )



def docker_cmd(
    policy: str,
    mode: str,
    batch_size: int,
    args: argparse.Namespace,
    *,
    steps: int,
    log_freq: int,
    save_checkpoint: bool,
    seed_checkpoint: str | None,
    job_name: str,
    device: str = "cuda",
) -> list[str]:
    spec = POLICIES[policy]
    workdir = Path(args.workdir)
    dataset_name = Path(args.dataset_root).name
    uid, gid = os.getuid(), os.getgid()

    # `policy-entrypoint.sh train` 을 쓰지 않고 lerobot-train 을 직접 부른다.
    # 이유: 그 entrypoint 는 `${VAR:-default}` 로 env 를 조립해 **빈 문자열을 미설정으로 취급**한다.
    # 그래서 `TRAIN_ACTION_REPRESENTATION_MODE=` 로도 `--policy.action_representation.mode` 를
    # 끌 수 없고, 그 필드가 없는 정책(pi05/xvla/vla_jepa/lingbot_va)은 draccus 에서
    # "The fields `action_representation` are not valid for PI05Config" 로 죽는다.
    # entrypoint 가 하는 나머지 일은 ACCELERATE_MIXED_PRECISION export 뿐이라 여기서 직접 넣는다.
    env: dict[str, str] = {
        "ACCELERATE_MIXED_PRECISION": args.mixed_precision if device == "cuda" else "no",
        # HF 캐시를 /DISK1 로 — 루트 파티션(여유 100GB)에 62GB 를 쌓지 않는다.
        "HF_HOME": "/workspace/hf",
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
        "WANDB_MODE": "disabled",
        # 컨테이너를 호스트 uid 로 돌리면 /root 에 못 쓴다. ACT 의 torchvision resnet18
        # 다운로드가 여기로 떨어지므로 쓰기 가능한 경로 + /DISK1 캐시로 고정한다.
        "HOME": "/tmp",
        "TORCH_HOME": "/workspace/hf/torch",
    }
    token = hf_token()
    if token:
        env["HF_TOKEN"] = token

    cmd = ["docker", "run", "--rm", "--name", container_name(job_name)]
    if device == "cuda":
        cmd += ["--gpus", "all"]
    cmd += [
        "--ipc=host",
        "--shm-size=16g",
        "-u",
        f"{uid}:{gid}",
        "-v",
        f"{Path(args.dataset_root).parent}:/workspace/datasets",
        "-v",
        f"{workdir / 'hf'}:/workspace/hf",
        "-v",
        f"{workdir / 'outputs'}:/workspace/outputs",
        "-v",
        f"{REPO_ROOT / 'scripts'}:/workspace/scripts:ro",
        "-v",
        f"{REPO_ROOT / 'src'}:/workspace/src:ro",
    ]
    for key, value in env.items():
        cmd += ["-e", f"{key}={value}"]
    cmd += ["--entrypoint", "lerobot-train", args.image]
    cmd += policy_args(policy, mode, seed_checkpoint=seed_checkpoint)
    cmd += [
        f"--dataset.repo_id={args.dataset_repo_id}",
        f"--dataset.root=/workspace/datasets/{dataset_name}",
        f"--dataset.video_backend={args.video_backend}",
        f"--policy.device={device}",
        "--policy.push_to_hub=false",
        f"--output_dir=/workspace/outputs/{job_name}",
        f"--job_name={job_name}",
        f"--steps={steps}",
        f"--batch_size={batch_size}",
        f"--num_workers={args.num_workers}",
        f"--log_freq={log_freq}",
        f"--save_checkpoint={'true' if save_checkpoint else 'false'}",
        "--save_freq=1000000",
        "--env_eval_freq=0",
        "--wandb.enable=false",
    ]
    rename_map = POLICIES[policy].mode(mode).rename_map
    if rename_map:
        cmd.append(f"--rename_map={json.dumps(rename_map, separators=(',', ':'))}")
    # action_representation 은 config 기본값이 이미 joint_absolute 라 명시하지 않는다
    # (필드가 없는 정책에 붙이면 draccus 가 거부한다). 다른 mode 를 재려면 아래를 켠다.
    if spec.has_action_representation and args.action_representation_mode != "joint_absolute":
        cmd.append(f"--policy.action_representation.mode={args.action_representation_mode}")
    return cmd


# ── 실행/파싱 ─────────────────────────────────────────────────────────────────
def parse_metrics(log_text: str) -> dict[str, float] | None:
    """마지막 metrics 로그 줄(= 정상 상태 창)을 파싱한다."""
    last: dict[str, float] | None = None
    for line in log_text.splitlines():
        if "updt_s:" not in line:
            continue
        values = {k: float(v) for k, v in _METRIC_RE.findall(line.split("INFO")[-1])}
        if "updt_s" in values:
            last = values
    return last


def parse_param_counts(log_text: str) -> dict[str, int] | None:
    """lerobot-train 이 찍는 num_learnable_params/num_total_params 를 뽑는다."""
    learnable = re.search(r"num_learnable_params=(\d+)", log_text)
    total = re.search(r"num_total_params=(\d+)", log_text)
    if not (learnable and total):
        return None
    return {"learnable": int(learnable.group(1)), "total": int(total.group(1))}


def static_vram_gb(total: int, learnable: int, param_bytes: float) -> dict[str, float]:
    """AdamW 정적 VRAM — 가중치 + grad + optimizer state (활성값 제외).

    activation 은 batch 에 비례해 여기 없다. 측정값(mem_gb)에서 이 값을 빼면 배치당
    activation 비용이 나온다.
    """
    weights = total * param_bytes / 1024**3
    grads = learnable * param_bytes / 1024**3
    adam = 2 * learnable * param_bytes / 1024**3
    return {
        "weights_gb": round(weights, 2),
        "grads_gb": round(grads, 2),
        "adam_gb": round(adam, 2),
        "static_gb": round(weights + grads + adam, 2),
    }


def classify_failure(log_text: str) -> str:
    lowered = log_text.lower()
    if "out of memory" in lowered or "outofmemoryerror" in lowered:
        return "oom"
    if "no space left on device" in lowered:
        return "disk_full"
    return "error"


def run_one(
    policy: str,
    mode: str,
    batch_size: int,
    args: argparse.Namespace,
    *,
    steps: int,
    log_freq: int,
    seed_checkpoint: str | None,
) -> dict[str, object]:
    job_name = f"bench_{policy}_{mode}_b{batch_size}"
    cmd = docker_cmd(
        policy,
        mode,
        batch_size,
        args,
        steps=steps,
        log_freq=log_freq,
        save_checkpoint=False,
        seed_checkpoint=seed_checkpoint,
        job_name=job_name,
    )
    log_path = Path(args.workdir) / "logs" / f"{job_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    docker_kill(job_name)  # 이전 실행 잔재 제거 (--name 충돌 방지)
    before = gpu_snapshot()
    poller = VramPoller()
    poller.start()
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(redacted(cmd) + "\n\n")
        log_file.flush()
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)
        try:
            returncode = proc.wait(timeout=args.timeout)
            status = "ok" if returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            docker_kill(job_name)
            proc.kill()
            proc.wait(timeout=60)
            returncode, status = -1, "timeout"
    wall_s = time.time() - started
    nvsmi_peak_mb = poller.stop()

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    metrics = parse_metrics(log_text)
    if status == "failed":
        status = classify_failure(log_text)
    if status == "ok" and metrics is None:
        status = "no_metrics"

    # 파라미터 수는 학습 로그에도 찍힌다 → params.json 을 여기서도 채워
    # `--validate-cpu` 를 먼저 돌리지 않아도 정적 VRAM 표가 나온다.
    counts = parse_param_counts(log_text)
    if counts:
        params_path = Path(args.workdir) / "params.json"
        params = (
            json.loads(params_path.read_text(encoding="utf-8")) if params_path.exists() else {}
        )
        params[f"{policy}/{mode}"] = counts
        params_path.write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")

    record: dict[str, object] = {
        "policy": policy,
        "mode": mode,
        "batch_size": batch_size,
        "status": status,
        "returncode": returncode,
        "steps": steps,
        "log_freq": log_freq,
        "num_workers": args.num_workers,
        "mixed_precision": args.mixed_precision,
        "wall_s": round(wall_s, 1),
        "nvsmi_peak_mb": round(nvsmi_peak_mb, 1),
        "gpu_used_mb_before": before.get("used_mb"),
        "gpu_util_pct_before": before.get("util_pct"),
        "log": str(log_path),
    }
    if counts:
        record["num_learnable_params"] = counts["learnable"]
        record["num_total_params"] = counts["total"]
    if metrics:
        update_s = metrics.get("updt_s", 0.0)
        data_s = metrics.get("data_s", 0.0)
        step_s = update_s + data_s
        record.update(
            {
                "update_s": update_s,
                "data_s": data_s,
                "step_s": round(step_s, 4),
                "mem_gb_allocated": metrics.get("mem_gb"),
                "loss": metrics.get("loss"),
                f"hours_{TARGET_STEPS // 1000}k": round(TARGET_STEPS * step_s / 3600, 2),
                f"hours_{TARGET_STEPS // 1000}k_update_only": round(
                    TARGET_STEPS * update_s / 3600, 2
                ),
            }
        )
    return record


def ensure_seed_checkpoint(args: argparse.Namespace, policy: str) -> str:
    """공개 base 를 못 쓰는 정책(ACT·X-VLA)의 fine-tune 출발점.

    1스텝만 학습해 저장한 자가 체크포인트다. 아키텍처·trainable 집합이 같으므로 fine-tune
    **비용**은 공개 base 에서 출발한 것과 동일하다(가중치 값만 다르다).
    """
    job_name = f"bench_seed_{policy}"
    host_root = Path(args.workdir) / "outputs" / job_name / "checkpoints"

    def _found() -> str | None:
        # 체크포인트 디렉터리 이름은 step 자릿수에 따라 결정된다(`max(6, len(str(total_steps)))`)
        # → 하드코딩하지 않고 찾는다.
        for candidate in sorted(host_root.glob("*/pretrained_model")):
            return f"/workspace/outputs/{job_name}/checkpoints/{candidate.parent.name}/pretrained_model"
        return None

    existing = _found()
    if existing:
        return existing
    cmd = docker_cmd(
        policy,
        "train",
        1,
        args,
        steps=1,
        log_freq=1,
        save_checkpoint=True,
        seed_checkpoint=None,
        job_name=job_name,
    )
    log_path = Path(args.workdir) / "logs" / f"{job_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[seed] {policy} 1-step 체크포인트 생성 …")
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(redacted(cmd) + "\n\n")
        rc = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, check=False).returncode
    created = _found()
    if rc != 0 or not created:
        raise RuntimeError(f"{policy} seed 체크포인트 생성 실패 (rc={rc}) — {log_path}")
    return created


# ── 추론 비용 측정 ────────────────────────────────────────────────────────────
def infer_cmd(
    policy: str,
    mode: str,
    batch_size: int,
    args: argparse.Namespace,
    *,
    seed_checkpoint: str | None,
    job_name: str,
    device: str = "cuda",
) -> list[str]:
    """컨테이너 안에서 measure_inference_cost.py 를 돌리는 docker 명령."""
    train_cmd = docker_cmd(
        policy,
        mode,
        batch_size,
        args,
        steps=1,
        log_freq=1,
        save_checkpoint=False,
        seed_checkpoint=seed_checkpoint,
        job_name=job_name,
        device=device,
    )
    # docker 옵션(마운트·env)은 학습과 동일하게 재사용하고, 이미지 뒤쪽만 갈아끼운다.
    image_index = train_cmd.index("--entrypoint")
    docker_opts = train_cmd[:image_index]
    spec = policy_spec_json(policy, mode, seed_checkpoint=seed_checkpoint)
    return [
        *docker_opts,
        "--entrypoint",
        "python",
        args.image,
        "/workspace/scripts/benchmark/measure_inference_cost.py",
        f"--policy-json={json.dumps(spec, separators=(',', ':'))}",
        f"--repo-id={args.dataset_repo_id}",
        f"--root=/workspace/datasets/{Path(args.dataset_root).name}",
        f"--batch-size={batch_size}",
        f"--iters={args.infer_iters}",
        f"--device={device}",
    ]


def run_infer(
    policy: str,
    mode: str,
    batch_size: int,
    args: argparse.Namespace,
    *,
    seed_checkpoint: str | None,
) -> dict[str, object]:
    job_name = f"infer_{policy}_{mode}_b{batch_size}"
    cmd = infer_cmd(
        policy, mode, batch_size, args, seed_checkpoint=seed_checkpoint, job_name=job_name
    )
    log_path = Path(args.workdir) / "logs" / f"{job_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    docker_kill(job_name)
    poller = VramPoller()
    poller.start()
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(redacted(cmd) + "\n\n")
        log_file.flush()
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)
        try:
            returncode = proc.wait(timeout=args.timeout)
            status = "ok" if returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            docker_kill(job_name)
            proc.kill()
            proc.wait(timeout=60)
            returncode, status = -1, "timeout"
    wall_s = time.time() - started
    nvsmi_peak_mb = poller.stop()
    text = log_path.read_text(encoding="utf-8", errors="replace")
    payload: dict[str, object] = {}
    for line in text.splitlines():
        if line.startswith("INFER_RESULT "):
            payload = json.loads(line[len("INFER_RESULT ") :])
    if status == "failed":
        status = classify_failure(text)
    if status == "ok" and not payload:
        status = "no_metrics"
    return {
        "policy": policy,
        "mode": mode,
        "batch_size": batch_size,
        "status": status,
        "returncode": returncode,
        "wall_s": round(wall_s, 1),
        "nvsmi_peak_mb": round(nvsmi_peak_mb, 1),
        "log": str(log_path),
        **payload,
    }


def report_infer(path: Path, batch_sizes: tuple[int, ...]) -> None:
    records = load_results(path)
    if not records:
        print(f"추론 결과 없음: {path}")
        return
    print("\n## 추론 비용 (full-chunk latency)\n")
    print("| policy | mode | batch | chunk p50 | p95 | n_action_steps | 스텝당 | 30Hz 여유 | mem_gb | nvsmi GB |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for (policy, mode, batch), r in sorted(records.items()):
        if r["status"] != "ok":
            print(f"| {policy} | {mode} | {batch} | **{r['status']}** | | | | | | {r['nvsmi_peak_mb'] / 1024:.1f} |")
            continue
        print(
            f"| {policy} | {mode} | {batch} | {r['chunk_p50_s'] * 1000:.0f} ms "
            f"| {r['chunk_p95_s'] * 1000:.0f} ms | {r['n_action_steps']} "
            f"| {r['per_step_ms']:.1f} ms | ×{r['realtime_headroom_30hz']:.2f} "
            f"| {r.get('mem_gb_allocated', 0):.1f} | {r['nvsmi_peak_mb'] / 1024:.1f} |"
        )


# ── CPU 검증 (GPU 경합 없이 config·feature·state_dict 계약만 확인) ──────────────
# 이 문구가 나오면 dataset·policy·processor·optimizer 가 전부 만들어진 것이다.
_READY_MARKER = "Start offline training on a fixed dataset"


def validate_cpu(args: argparse.Namespace, policies: list[str], modes: list[str]) -> int:
    """정책마다 CPU 로 1스텝 학습을 띄워 '학습 루프 진입'까지 도달하는지만 본다.

    GPU 를 쓰지 않으므로 경합 중에도 돌릴 수 있다. 스텝 자체는 CPU 에서 매우 느리니
    marker 가 보이면 즉시 죽인다.
    """
    params_path = Path(args.workdir) / "params.json"
    params: dict[str, dict[str, int]] = (
        json.loads(params_path.read_text(encoding="utf-8")) if params_path.exists() else {}
    )
    failures = 0
    for policy in policies:
        for mode in modes:
            seed = None
            if POLICIES[policy].mode(mode).seed_from_self:
                print(f"[cpu-validate] {policy}/{mode}: 자가 seed 필요 → GPU sweep 에서 확인 (skip)")
                continue
            job_name = f"cpuval_{policy}_{mode}"
            cmd = docker_cmd(
                policy,
                mode,
                1,
                args,
                steps=1,
                log_freq=1,
                save_checkpoint=False,
                seed_checkpoint=seed,
                job_name=job_name,
                device="cpu",
            )
            log_path = Path(args.workdir) / "logs" / f"{job_name}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[cpu-validate] {policy}/{mode} …", end=" ", flush=True)
            docker_kill(job_name)
            with log_path.open("w", encoding="utf-8") as log_file:
                log_file.write(redacted(cmd) + "\n\n")
                log_file.flush()
                proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)
                deadline = time.time() + args.validate_timeout
                reached = False
                while time.time() < deadline:
                    if proc.poll() is not None:
                        break
                    if _READY_MARKER in log_path.read_text(encoding="utf-8", errors="replace"):
                        reached = True
                        break
                    time.sleep(3.0)
                docker_kill(job_name)
                proc.kill()
                proc.wait(timeout=60)
            text = log_path.read_text(encoding="utf-8", errors="replace")
            reached = reached or _READY_MARKER in text
            counts = parse_param_counts(text)
            if counts:
                params[f"{policy}/{mode}"] = counts
                params_path.write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
            if reached:
                extra = (
                    f"  (trainable {counts['learnable'] / 1e6:.0f}M / total {counts['total'] / 1e6:.0f}M)"
                    if counts
                    else ""
                )
                print(f"PASS{extra}")
            else:
                failures += 1
                tail = [line for line in text.strip().splitlines() if line.strip()][-3:]
                print("FAIL")
                for line in tail:
                    print(f"        {line[:200]}")
                print(f"        log: {log_path}")
    print(f"[cpu-validate] 실패 {failures}")
    return failures


# ── prefetch ─────────────────────────────────────────────────────────────────
def prefetch(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir)
    (workdir / "hf").mkdir(parents=True, exist_ok=True)
    failures = 0
    for repo_id, includes in PREFETCH:
        print(f"[prefetch] {repo_id} {' '.join(includes)}")
        cmd = [
            "docker",
            "run",
            "--rm",
            "-u",
            f"{os.getuid()}:{os.getgid()}",
            "-v",
            f"{workdir / 'hf'}:/workspace/hf",
            "-e",
            "HF_HOME=/workspace/hf",
        ]
        token = hf_token()
        if token:
            cmd += ["-e", f"HF_TOKEN={token}"]
        cmd += ["--entrypoint", "hf", args.image, "download", repo_id]
        for pattern in includes:
            cmd += ["--include", pattern]
        if args.dry_run:
            print("   " + redacted(cmd))
            continue
        if subprocess.run(cmd, check=False).returncode != 0:
            print(f"   ✗ 실패: {repo_id}")
            failures += 1
    if not args.dry_run:
        used = shutil.disk_usage(workdir)
        print(f"[prefetch] 완료 (실패 {failures}). {workdir} 여유 {used.free / 1e9:.1f}GB")
    return failures


# ── 리포트 ───────────────────────────────────────────────────────────────────
def load_results(path: Path) -> dict[tuple[str, str, int], dict]:
    records: dict[tuple[str, str, int], dict] = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        records[(rec["policy"], rec["mode"], rec["batch_size"])] = rec
    return records


def report_params(workdir: Path) -> None:
    """--validate-cpu 로 모은 파라미터 수 → AdamW 정적 VRAM 표 (배치 무관)."""
    params_path = workdir / "params.json"
    if not params_path.exists():
        return
    params = json.loads(params_path.read_text(encoding="utf-8"))
    print("\n## AdamW 정적 VRAM (activation 제외, batch 무관)\n")
    print("| policy | mode | total | trainable | B/param | weights | grads | AdamW | 합계 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for key, counts in params.items():
        policy = key.split("/")[0]
        pb = POLICIES[policy].param_bytes
        est = static_vram_gb(counts["total"], counts["learnable"], pb)
        print(
            f"| {policy} | {key.split('/')[1]} | {counts['total'] / 1e6:.0f}M "
            f"| {counts['learnable'] / 1e6:.0f}M | {pb:g} | {est['weights_gb']:.1f} "
            f"| {est['grads_gb']:.1f} | {est['adam_gb']:.1f} | **{est['static_gb']:.1f} GB** |"
        )


def report(path: Path, batch_sizes: tuple[int, ...]) -> None:
    report_params(path.parent)
    records = load_results(path)
    if not records:
        print(f"\n결과 없음: {path}")
        return
    hours_key = f"hours_{TARGET_STEPS // 1000}k"
    for policy, spec in POLICIES.items():
        rows = [(m, b) for m in MODES for b in batch_sizes if (policy, m, b) in records]
        if not rows:
            continue
        print(f"\n### {spec.label} (`{policy}`)")
        print(f"| mode | batch | s/step | update_s | data_s | mem_gb(alloc) | nvsmi peak GB | {TARGET_STEPS // 1000}k steps |")
        print("|---|---|---|---|---|---|---|---|")
        for mode, batch in rows:
            r = records[(policy, mode, batch)]
            if r["status"] != "ok":
                print(f"| {mode} | {batch} | **{r['status']}** | | | | {r['nvsmi_peak_mb'] / 1024:.1f} | |")
                continue
            hours = r.get(hours_key, 0.0)
            print(
                f"| {mode} | {batch} | {r['step_s']:.3f} | {r['update_s']:.3f} | {r['data_s']:.3f} "
                f"| {r.get('mem_gb_allocated', 0):.1f} | {r['nvsmi_peak_mb'] / 1024:.1f} "
                f"| {hours:.1f} h ({hours / 24:.1f} d) |"
            )


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-repo-id", default=DEFAULT_DATASET_REPO_ID)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR, help="hf 캐시·outputs·logs·results")
    parser.add_argument("--policies", default=",".join(POLICIES), help="쉼표 구분")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument(
        "--batch-sizes", default=",".join(str(b) for b in DEFAULT_BATCH_SIZES)
    )
    parser.add_argument("--steps", type=int, default=30, help="조합당 학습 스텝 (3×log_freq 이상 권장)")
    parser.add_argument("--log-freq", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mixed-precision", default="bf16", choices=("bf16", "fp16", "no"))
    parser.add_argument("--video-backend", default="torchcodec")
    parser.add_argument("--action-representation-mode", default="joint_absolute")
    parser.add_argument("--timeout", type=int, default=1800, help="조합당 초 단위 상한")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prefetch", action="store_true", help="가중치만 미리 받고 종료 (GPU 불필요)")
    parser.add_argument(
        "--validate-cpu",
        action="store_true",
        help="GPU 없이 정책별 config/feature/state_dict 계약만 확인하고 종료",
    )
    parser.add_argument("--validate-timeout", type=int, default=1200)
    parser.add_argument(
        "--infer",
        action="store_true",
        help="학습 대신 **추론** 비용(full-chunk latency·VRAM)을 측정한다",
    )
    parser.add_argument("--infer-iters", type=int, default=10)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--rerun", action="store_true", help="이미 기록된 조합도 다시 실행")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="status!=ok 인 조합만 다시 실행 (GPU 경합 때문에 OOM 난 조합 재시도용)",
    )
    parser.add_argument("--require-free-mb", type=float, default=0.0)
    parser.add_argument("--require-idle-util", type=float, default=100.0)
    args = parser.parse_args()

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    batch_sizes = tuple(int(b) for b in args.batch_sizes.split(",") if b.strip())
    unknown = set(policies) - set(POLICIES)
    if unknown:
        parser.error(f"unknown policies: {sorted(unknown)}")

    results_path = Path(args.workdir) / ("inference_results.jsonl" if args.infer else "results.jsonl")
    if args.report:
        if args.infer:
            report_infer(results_path, batch_sizes)
        else:
            report(results_path, batch_sizes)
        return
    if args.prefetch:
        sys.exit(1 if prefetch(args) else 0)

    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    (Path(args.workdir) / "logs").mkdir(exist_ok=True)
    (Path(args.workdir) / "outputs").mkdir(exist_ok=True)
    (Path(args.workdir) / "hf").mkdir(exist_ok=True)

    if args.validate_cpu:
        if not Path(args.dataset_root).exists():
            parser.error(f"데이터셋 없음: {args.dataset_root}")
        sys.exit(1 if validate_cpu(args, policies, modes) else 0)

    if not args.dry_run and not Path(args.dataset_root).exists():
        parser.error(
            f"데이터셋 없음: {args.dataset_root}\n"
            "먼저 scripts/benchmark/make_dummy_dataset.py 를 컨테이너에서 실행할 것"
        )

    snap = gpu_snapshot()
    if snap:
        print(
            f"[gpu] free {snap['free_mb'] / 1024:.1f}GB / {snap['total_mb'] / 1024:.1f}GB, "
            f"util {snap['util_pct']:.0f}%"
        )
        if snap["free_mb"] < args.require_free_mb:
            parser.error(f"여유 VRAM 부족: {snap['free_mb']:.0f}MB < {args.require_free_mb:.0f}MB")
        if snap["util_pct"] > args.require_idle_util:
            parser.error(f"GPU 사용률 초과: {snap['util_pct']:.0f}% > {args.require_idle_util:.0f}%")
        if snap["util_pct"] > 5 and not args.dry_run:
            print("[gpu] ⚠ 다른 작업이 GPU 를 쓰는 중 — 시간 측정이 오염된다 (VRAM 은 무관)")

    done = {} if args.rerun else load_results(results_path)
    if args.retry_failed:
        done = {k: v for k, v in done.items() if v.get("status") == "ok"}
    seed_checkpoints: dict[str, str] = {}
    planned = [(p, m, b) for p in policies for m in modes for b in batch_sizes]
    print(f"[plan] {len(planned)} 조합 (이미 완료 {sum(1 for k in planned if k in done)})")

    for policy, mode, batch in planned:
        key = (policy, mode, batch)
        if key in done:
            continue
        seed_checkpoint = seed_checkpoints.get(policy)
        if POLICIES[policy].mode(mode).seed_from_self and seed_checkpoint is None:
            seed_checkpoint = (
                f"<{policy}-seed-checkpoint>"
                if args.dry_run
                else ensure_seed_checkpoint(args, policy)
            )
            seed_checkpoints[policy] = seed_checkpoint
        if args.dry_run:
            builder = (
                (
                    lambda: infer_cmd(
                        policy,
                        mode,
                        batch,
                        args,
                        seed_checkpoint=seed_checkpoint,
                        job_name=f"infer_{policy}_{mode}_b{batch}",
                    )
                )
                if args.infer
                else (
                    lambda: docker_cmd(
                        policy,
                        mode,
                        batch,
                        args,
                        steps=args.steps,
                        log_freq=args.log_freq,
                        save_checkpoint=False,
                        seed_checkpoint=seed_checkpoint,
                        job_name=f"bench_{policy}_{mode}_b{batch}",
                    )
                )
            )
            print(f"\n# {policy}/{mode}/b{batch}\n" + redacted(builder()))
            continue
        print(f"[run] {policy}/{mode}/b{batch} …", flush=True)
        if args.infer:
            record = run_infer(
                policy, mode, batch, args, seed_checkpoint=seed_checkpoint
            )
        else:
            record = run_one(
                policy,
                mode,
                batch,
                args,
                steps=args.steps,
                log_freq=args.log_freq,
                seed_checkpoint=seed_checkpoint,
            )
        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if record["status"] != "ok":
            summary = f"status={record['status']} (log: {record['log']})"
        elif args.infer:
            summary = (
                f"chunk {record['chunk_p50_s'] * 1000:.0f}ms  "
                f"스텝당 {record['per_step_ms']:.1f}ms  "
                f"alloc {record.get('mem_gb_allocated', 0):.1f}GB"
            )
        else:
            summary = (
                f"{record['step_s']:.3f}s/step  alloc {record.get('mem_gb_allocated', 0):.1f}GB  "
                f"nvsmi {record['nvsmi_peak_mb'] / 1024:.1f}GB"
            )
        print(f"      → {summary}")

    if not args.dry_run:
        if args.infer:
            report_infer(results_path, batch_sizes)
        else:
            report(results_path, batch_sizes)


if __name__ == "__main__":
    main()
