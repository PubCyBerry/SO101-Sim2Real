#!/usr/bin/env bash
# SO-101 PickCube — 레퍼런스(ref_repos/pick_and_place, IsaacLab Lift-Cube-Place) 정합 PPO 학습.
#
# 성공이 확인된 레퍼런스의 보상/obs/arch/PPO 를 SO-101+그릇 환경에 그대로 맞춘 단일 경로.
# train.py 기본값이 이미 ref 정합이라 이 스크립트는 num_envs/iteration/DR 레벨만 지정한다.
#   - 보상 : ref dense 4항(reaching1·lifting30·tracking16·lowering7) + smoothness −1e-4 (env_cfg 기본)
#            tracking/lowering 이 큐브를 그릇 **안**(3D center)으로 끌어 "그릇에 넣기" 학습.
#   - obs  : ref_policy 54-dim (joint_pos/vel + TCP/cube/bowl 6d pose+vel + last_action)
#   - arch : MLP [128,64,32] + obs_normalization, init_noise_std 1.0
#   - PPO  : γ0.98, lr 8e-5(adaptive), entropy 0.006, 24/5/4, max_grad_norm 0.4
#   - 큐브 : 40mm 1개(active_objects 1 = Cube1). 그리퍼 연속(North Star 6-dim).
#   - DR   : DR_LEVEL(기본 0=완전고정) → 1 spawn → 2 sensor → 3 물리/시각. 단계적으로 올린다.
#
# 사용:
#   DR_LEVEL=0 bash scripts/reinforcement_learning/run_expert_policy.sh        # 완전고정 학습
#   DR_LEVEL=1 RESUME=<ckpt> bash scripts/reinforcement_learning/run_expert_policy.sh  # 이어서 spawn 랜덤
# 검증: eval_success.py (success_rate) 로 단일 큐브 성공률 확인.
set -euo pipefail

ROOT=/home/konan147/Workspaces/SO101-Sim2Real
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT/.venv/bin/python"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$(pwd)/src"

LOGROOT="$ROOT/outputs/rl/rsl_rl"
DR_LEVEL="${DR_LEVEL:-0}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-4000}"
DEVICE="${DEVICE:-cuda:0}"
RESUME="${RESUME:-}"   # DR 단계 상승 시 이전 단계 체크포인트(.pt) 경로

resume_args=()
if [[ -n "$RESUME" ]]; then
  resume_args=(--resume_checkpoint "$RESUME")
fi

echo "[ref] STAGE train: 레퍼런스 정합 (DR_LEVEL=$DR_LEVEL, num_envs=$NUM_ENVS, iters=$MAX_ITERS)"
$PY scripts/reinforcement_learning/train.py \
  --task SimToReal-SO101-PickCube-v0 \
  --active_objects 1 --dr_level "$DR_LEVEL" \
  --num_envs "$NUM_ENVS" --max_iterations "$MAX_ITERS" \
  --device "$DEVICE" --headless --seed 42 --save_interval 50 \
  --experiment_name ref_lift_place --log_root_path "$LOGROOT" \
  --run_name "ref_dr${DR_LEVEL}" \
  "${resume_args[@]}"
echo "[ref] done (DR_LEVEL=$DR_LEVEL)"
