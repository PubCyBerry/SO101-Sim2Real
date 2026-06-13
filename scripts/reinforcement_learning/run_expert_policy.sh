#!/usr/bin/env bash
# SO-101 PickCube — pick-and-place RL expert policy (단일 end-to-end, BC + reverse curriculum).
#
# 전략(2026-06-12, scratch-only 폐기 후 재설계): scratch reward-shaping(8회)·demo-reset-only
# (v16~20) 실패의 공통 누락 = expert ACTION 미주입. 검증된 SM(해석적 IK side-approach, 1-cube
# ~90%) 전궤적을 BC clone(obs→action) → RL finetune(camp-free full_bc 프리셋) + reverse
# curriculum(demo_reset_prob anneal, NVIDIA IndustReal/RFCL 정석). 단일 정책 end-to-end
# (skill-chaining 폐기 — 데모 있으면 단일정책 우세, VLA 궤적 연속).
#
# 사용: bash scripts/reinforcement_learning/run_expert_policy.sh [demos|bc|train|all]
#   resume/mid-run 튜닝 자유(재현성 제약 해제). MLP [256,128] no-norm(BC 가 MLP 전용).
#
# 검증: monitor_eval.py --skill full --bootstrap_prob 0 --demo_reset_prob 0 → success ≥0.90.
set -euo pipefail

ROOT=/home/konan147/Workspaces/SO101-Sim2Real
cd "$ROOT/.claude/worktrees/lstm-ppo-pickcube"
PY="$ROOT/.venv/bin/python"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$(pwd)/src"

EXP=lstm_ppo_pickcube
LOGROOT="$ROOT/outputs/rl/rsl_rl"
RUNDIR="$LOGROOT/$EXP"
SM_DEMO_DIR="$(pwd)/outputs/demos/sm_c1"     # SM 전문가 전궤적(BC + demo-reset 양쪽)
BC_DIR="$RUNDIR/bc_full_lstm"                 # BC warmstart 산출 LSTM ckpt
RUN=lstm_ppo_gb                               # 순수 PPO scratch + grasp-bootstrap run 이름
# LSTM + **순수 PPO scratch** (BC·SM·demo-reset 미사용 — 사용자 지시 2026-06-13).
# grasp_bootstrap 은 기하적 reset(default 자세 grasp point, SM 데이터 아님)이라 사용 OK.
RECURRENT="--recurrent --rnn_type lstm --rnn_hidden_dim 256 --rnn_num_layers 1"
COMMON_PPO="--num_envs 4096 --num_steps_per_env 48 --num_learning_epochs 6 --num_mini_batches 4 \
  --schedule adaptive --learning_rate 1e-4 --entropy_coef 0.005 --gamma 0.99 --lam 0.95 \
  --device cuda:0 --headless --seed 42 --save_interval 25 \
  --experiment_name $EXP --log_root_path $LOGROOT"

# Stage demos — SM 전문가 시연 수집(1-cube 성공 전궤적). 이미 있으면 skip 가능.
stage_demos() {
  echo "[expert] STAGE demos: SM 전문가 demo 수집 → $SM_DEMO_DIR"
  $PY scripts/environments/pick_cube_state_machine.py \
    --record_demos "$SM_DEMO_DIR" --demo_tag c1 \
    --active_objects 1 --num_envs 512 --num_episodes 4 --headless --device cuda:0
}

# Stage bc — SM 전궤적을 MLP ActorCritic 에 BC clone(actor MSE). full task phase 유지
#   (settle/retreat/home 만 제외 = reach→grasp→lift→transport→lower→release). SM 은 성공만
#   저장하므로 --no-require_success 안전(meta.placed_and_released 키 부재 회피).
stage_bc() {
  echo "[expert] STAGE bc: BC warmstart(full task) → $BC_DIR"
  $PY scripts/reinforcement_learning/bc_warmstart.py \
    --task SimToReal-SO101-PickCube-v0 $RECURRENT \
    --expert_dataset_pt "$SM_DEMO_DIR"/demo_*.pt --output_dir "$BC_DIR" \
    --no-require_success --exclude_phase_contains SETTLE RETREAT HOME DONE DRAG \
    --active_objects 1 --epochs 50 --device cuda:0
}

# Stage train — **순수 PPO scratch + grasp-bootstrap 중심**(BC·demo-reset 없음, 사용자 지시).
#   grasp_v4 가 유일하게 scratch grasp 점화시킨 구조(LSTM+PPO+grasp_bootstrap+grasp shaping+RND).
#   - grasp_bootstrap **0.7**(↑) anneal→0 over 1200: full-grasp(든 큐브, **다양한 yaw 쿼터니온** →
#     하류 transport/place robust + 다양 궤적) + pre-grasp(pregrasp_frac 0.3 = 그리퍼 open→닫기 연습).
#   - 다양 quat = `_bootstrap_grasp` 가 full-grasp 큐브를 random yaw + wrist_roll 정합(grip 유효)으로 reset.
#   - 보상 full_bc: grasp_align 1.0 + grasp_close 3.0(γ=0.99 라 camp-free 점화) + task_progress 80 + place + terminal.
#   - RND grasp_focus: grasp close 탐색.
stage_train() {
  echo "[expert] STAGE train: 순수 PPO scratch (full_bc, LSTM, grasp-bootstrap 0.7 + 다양 quat + RND)"
  $PY scripts/reinforcement_learning/train.py --task SimToReal-SO101-PickCube-v0 \
    --skill full_bc $RECURRENT $COMMON_PPO \
    --max_iterations 1500 --active_objects 1 \
    --grasp_bootstrap_prob 0.7 --grasp_bootstrap_prob_final 0.0 --grasp_bootstrap_anneal_iters 1200 \
    --grasp_bootstrap_pregrasp_frac 0.3 \
    --rnd --rnd_weight 0.5 --rnd_state_group grasp_focus \
    --run_name $RUN
}

case "${1:-train}" in
  demos) stage_demos ;;         # (이력) SM demo 수집 — 현 전략 미사용
  bc)    stage_bc ;;            # (이력) BC warmstart — 현 전략 미사용(사용자 지시)
  train) stage_train ;;         # 순수 PPO scratch (현 전략)
  all)   stage_train ;;
  *) echo "usage: $0 [train]  (순수 PPO; demos/bc 는 이력)"; exit 1 ;;
esac
echo "[expert] done: ${1:-all}"
