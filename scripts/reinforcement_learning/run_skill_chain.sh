#!/usr/bin/env bash
# SO-101 PickCube skill-chaining 재현 파이프라인 — 단일 절차(reproducible).
#
# 원칙(사용자 요구):
#   - 각 stage = 고정 config 단일 run. **mid-run reward 변경/수동 resume 금지.**
#   - config 가 source of truth. 실패하면 config(=프리셋/이 스크립트)만 고치고 stage 를
#     전체 재실행한다. 최종 config + 이 스크립트 = 재현 절차.
#   - skill chaining 은 본질상 2-policy(skill1→수집→skill2→chain)라 단일 연속 run 이 아니므로,
#     전 과정을 이 한 스크립트로 묶어 "하나의 학습 과정"으로 만든다.
#
# 아키텍처(D7/D9): scratch 점화 8회 실패(telescoping PBRS 가 grasp '유지'를 보상 못함) →
#   검증된 SM(해석적 IK side-approach) 전문가를 BC warmstart 해 grasp+hold+transport 를 주입,
#   RL 로 finetune. BC 가 MLP 전용이라 전 stage MLP [256,128] no-norm 으로 통일.
#
# 사용:
#   bash scripts/reinforcement_learning/run_skill_chain.sh [sm_demos|bc|skill1|collect|skill2|chain|all]
#   (기본 all). 각 stage 는 앞 stage 산출물(최신 ckpt/demo)을 자동으로 집어 쓴다.
#
# 주의: GPU 1장 점유. sm_demos ~10분, bc ~5분, skill1 ~3h, skill2 ~1-2h. 백그라운드로 돌리고 로그로 모니터.
set -euo pipefail

ROOT=/home/konan147/Workspaces/SO101-Sim2Real
cd "$ROOT/.claude/worktrees/lstm-ppo-pickcube"
PY="$ROOT/.venv/bin/python"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$(pwd)/src"

EXP=lstm_ppo_pickcube
LOGROOT="$ROOT/outputs/rl/rsl_rl"
RUNDIR="$LOGROOT/$EXP"
DEMO_DIR="$(pwd)/outputs/demos/skill1_overbowl"   # skill1 over-bowl 스냅샷(skill2 demo-reset)
SM_DEMO_DIR="$(pwd)/outputs/demos/sm_c1"          # SM 전문가 전궤적(BC 입력)
BC_DIR="$RUNDIR/bc_skill1"                         # BC warmstart 산출 MLP ckpt
S1_RUN=mlp_skill1_bcft                             # skill1 run 이름(BC→RL finetune)
S2_RUN=mlp_skill2                                  # skill2 run 이름
# 전 stage 공통 아키텍처 — MLP [256,128] no-norm(train/monitor/collect/chain/bc 모두 동일 기본값).
# 어긋나면 ckpt 로드 실패. LSTM 으로 되돌리려면 아래에 --recurrent --rnn_type lstm --rnn_hidden_dim 256
# --rnn_num_layers 1 --obs_normalization 를 넣는다(단 BC 는 MLP 전용이라 BC 경로와 불호환).
COMMON_POLICY=""
COMMON_PPO="--num_envs 4096 --num_steps_per_env 48 --num_learning_epochs 6 --num_mini_batches 4 \
  --schedule adaptive --learning_rate 1e-4 --entropy_coef 0.005 --gamma 0.997 --lam 0.95 \
  --device cuda:0 --headless --seed 42 --save_interval 25 \
  --experiment_name $EXP --log_root_path $LOGROOT"

latest_ckpt() { ls -t "$1"/model_*.pt | head -1; }
latest_run()  { ls -dt $RUNDIR/*"$1" | head -1; }

# Stage 0a — SM 전문가 시연 수집(1-cube 성공 궤적 전체). BC 입력.
stage_sm_demos() {
  echo "[pipeline] STAGE 0a: SM 전문가 demo 수집 → $SM_DEMO_DIR"
  $PY scripts/environments/pick_cube_state_machine.py \
    --record_demos "$SM_DEMO_DIR" --demo_tag c1 \
    --active_objects 1 --num_envs 512 --num_episodes 4 \
    --headless --device cuda:0
}

# Stage 0b — SM demo 를 MLP ActorCritic 에 BC clone(actor MSE). acquire+transport phase 만(place 제외).
#   SM 은 성공 에피소드만 저장 → --no-require_success 안전(meta.placed_and_released 키 없음 회피).
stage_bc() {
  echo "[pipeline] STAGE 0b: BC warmstart → $BC_DIR"
  $PY scripts/reinforcement_learning/bc_warmstart.py \
    --task SimToReal-SO101-PickCube-v0 \
    --expert_dataset_pt "$SM_DEMO_DIR"/demo_*.pt --output_dir "$BC_DIR" \
    --no-require_success --exclude_phase_contains settle lower release retreat home \
    --active_objects 1 --epochs 50 --device cuda:0
}

# Stage 1 — skill1(acquire+transport). reward = apply_skill_acquire 고정 프리셋.
#   BC ckpt 에서 resume(actor warmstart, optimizer 는 새로) → terminal 도달로 credit assignment 해결.
#   grasp_bootstrap 로 보조 점화, over_bowl_grasped 종료로 handoff.
stage_skill1() {
  echo "[pipeline] STAGE 1: skill1 BC→RL finetune"
  $PY scripts/reinforcement_learning/train.py --task SimToReal-SO101-PickCube-v0 \
    --skill acquire $COMMON_POLICY $COMMON_PPO \
    --resume_checkpoint "$BC_DIR/model_0.pt" --resume_without_optimizer \
    --max_iterations 1500 --rnd --rnd_weight 0.5 --rnd_state_group grasp_focus \
    --active_objects 1 \
    --grasp_bootstrap_prob 0.5 --grasp_bootstrap_prob_final 0.0 --grasp_bootstrap_anneal_iters 800 \
    --run_name $S1_RUN
}

# Stage 2 — skill1 최신 ckpt 로 over-bowl-grasped 상태 수집 → skill2 reset 분포.
stage_collect() {
  echo "[pipeline] STAGE 2: skill1 over-bowl 상태 수집"
  local run; run="$(latest_run $S1_RUN)"
  $PY scripts/reinforcement_learning/collect_skill1_states.py \
    --checkpoint "$(latest_ckpt "$run")" --output_dir "$DEMO_DIR" \
    --num_envs 512 --target_states 2000 $COMMON_POLICY --active_objects 1 --device cuda:0
}

# Stage 3 — skill2(place+release) scratch. reward = apply_skill_place 고정 프리셋(단기 5s,
#   grasp_close off, require_open). demo_reset 1.0 = 전부 skill1 수집 상태에서 시작.
stage_skill2() {
  echo "[pipeline] STAGE 3: skill2 scratch 학습(demo-reset from skill1)"
  $PY scripts/reinforcement_learning/train.py --task SimToReal-SO101-PickCube-v0 \
    --skill place $COMMON_POLICY $COMMON_PPO \
    --max_iterations 1000 --active_objects 1 \
    --demo_reset_prob 1.0 --demo_dataset_dir "$DEMO_DIR" \
    --run_name $S2_RUN
}

# Stage 4 — skill1→skill2 chained end-to-end eval(부트스트랩·demo 없음, DR-on).
stage_chain() {
  echo "[pipeline] STAGE 4: chained eval"
  local s1 s2; s1="$(latest_run $S1_RUN)"; s2="$(latest_run $S2_RUN)"
  $PY scripts/reinforcement_learning/eval_chain.py \
    --skill1_checkpoint "$(latest_ckpt "$s1")" --skill2_checkpoint "$(latest_ckpt "$s2")" \
    $COMMON_POLICY --num_envs 64 --num_episodes 128 --active_objects 1 --device cuda:0
}

case "${1:-all}" in
  sm_demos) stage_sm_demos ;;
  bc)       stage_bc ;;
  skill1)   stage_skill1 ;;
  collect)  stage_collect ;;
  skill2)   stage_skill2 ;;
  chain)    stage_chain ;;
  all)      stage_sm_demos; stage_bc; stage_skill1; stage_collect; stage_skill2; stage_chain ;;
  *) echo "usage: $0 [sm_demos|bc|skill1|collect|skill2|chain|all]"; exit 1 ;;
esac
echo "[pipeline] done: ${1:-all}"
