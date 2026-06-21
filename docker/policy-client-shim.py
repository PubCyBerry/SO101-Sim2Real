#!/usr/bin/env python
"""policy-client-shim.py — Workaround for huggingface/lerobot#3078 (+ rerun viewer).

lerobot 0.4.4 의 `lerobot/async_inference/robot_client.py` 는 import 블록에서
built-in robot config 모듈들 (`so_follower`, `bi_so_follower`, `koch_follower`,
`omx_follower`) 을 import 하지 않는다. 결과적으로 `RobotConfig.register_subclass`
데코레이터가 실행되지 않아 draccus choice registry 가 비어 있고,
``--robot.type=so101_follower`` 같은 인자가 ``invalid choice ... (choose from )`` 로 거부된다.

해당 회귀는 upstream PR #3081 에서 수정되었으나 0.4.4 태그에는 미반영. 본 shim 은
robot_client 의 진입점을 실행하기 전에 필요한 robot config 모듈을 선행 import 해
registry 를 채운다.

■ 실행 방식 — runpy 대신 async_client() 직접 호출
  이전에는 ``runpy.run_module(run_name="__main__")`` 로 robot_client 의 __main__
  블록을 재실행했으나, 이 경우 robot_client 가 ``__main__`` 네임스페이스로 다시
  로드되어 ``RobotClient`` 클래스가 새 객체로 재정의된다. 아래 rerun monkey patch
  (DISPLAY_DATA=true) 가 import 한 모듈의 ``RobotClient`` 에 적용돼도, 실제 실행되는
  것은 재정의된 별개 클래스라 patch 가 무효화된다 (runpy RuntimeWarning 동반).
  따라서 import 한 ``async_client`` / ``register_third_party_plugins`` 를 직접 호출해
  patch 대상과 실행 대상이 동일한 모듈 객체가 되도록 한다.

upstream 패치가 들어간 lerobot 버전으로 올라가면 본 파일을 삭제하고
``entrypoint.sh`` 의 ``policy-client`` 분기가 ``python -m
lerobot.async_inference.robot_client`` 를 직접 호출하도록 되돌리면 된다.
"""

# ── Built-in robot configs — side-effect imports (PR #3081 와 동일 목록) ────
# 각 모듈의 모듈-레벨 코드가 ``@RobotConfig.register_subclass(...)`` 를 실행해
# choice registry 에 항목을 추가한다.
import lerobot.robots.so_follower.config_so_follower  # noqa: F401  (so101_follower, so100_follower)
import lerobot.robots.bi_so_follower.config_bi_so_follower  # noqa: F401
import lerobot.robots.koch_follower.config_koch_follower  # noqa: F401
import lerobot.robots.omx_follower.config_omx_follower  # noqa: F401

import os

from lerobot.async_inference.robot_client import (
    RobotClient,
    async_client,
    register_third_party_plugins,
)

# ── (선택) rerun viewer — DISPLAY_DATA=true 일 때 control loop 데이터 로깅 ──
# robot_client 0.4.4 는 teleop/record 와 달리 rerun 옵션이 없다. control loop 이
# 매 틱 반환하는 raw_observation(카메라 + follower state) / performed_action 을
# monkey patch 로 가로채 log_rerun_data 에 흘려 실시간 시각화한다.
#   DISPLAY_DATA=true            → 활성화 (rr.spawn 로컬 뷰어)
#   DISPLAY_IP / DISPLAY_PORT    → 설정 시 원격 rerun 서버로 송출 ("null"/빈값이면 무시)
if os.getenv("DISPLAY_DATA", "false").lower() == "true":
    from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

    _ip = os.getenv("DISPLAY_IP") or None
    _port = os.getenv("DISPLAY_PORT") or None
    _ip = None if _ip in (None, "", "null") else _ip
    _port = int(_port) if _port not in (None, "", "null") else None
    init_rerun(session_name="policy_client", ip=_ip, port=_port)

    _orig_obs = RobotClient.control_loop_observation
    _orig_act = RobotClient.control_loop_action

    def _obs_with_rerun(self, task, verbose=False):
        raw = _orig_obs(self, task, verbose)
        if raw is not None:
            log_rerun_data(observation=raw)
        return raw

    def _act_with_rerun(self, verbose=False):
        act = _orig_act(self, verbose)
        if act is not None:
            log_rerun_data(action=act)
        return act

    RobotClient.control_loop_observation = _obs_with_rerun
    RobotClient.control_loop_action = _act_with_rerun

# ── (선택) Sim↔Real per-joint frame 보정 (Option B) ─────────────────────────────
# SmolVLA 의 정규화(MEAN_STD)는 데이터셋 stats 를 pre/post 가 공유 → 수학적으로 상쇄,
# 모델 I/O 는 **sim(데이터셋) 프레임 그대로**. 따라서 sim 학습 모델을 real 에 배포하면
# **real 원시값 ↔ sim 프레임** 차이가 그대로 오차가 된다. 이 차이는 joint 별로:
#   · arm 5축 = `MotorNormMode.DEGREES`(절대 기계각, 스케일 1:1). real 0° = calibration home
#     (set_half_turn_homings), sim 0° = URDF zero → **영점 offset + (URDF 축 vs 모터 방향) 부호**.
#   · gripper = `RANGE_0_100`(캘리브 full-travel %) vs sim rad×31.75 → **스케일 + offset**.
#
# 해결: 추론 경계(robot 직전/직후)에서 joint 별 affine 으로 real↔sim 프레임을 변환한다.
# closed-loop(이전 action 이 다음 obs.state) 라 **양방향** 필수:
#   forward (real→model, get_observation): model = A_j * real + B_j        → 서버/모델로
#   inverse (model→real, send_action)    : real  = (model - B_j) / A_j      → 로봇으로
#   (입력 미변환 시 real proprioception 이 모델 학습 분포 밖(OOD)이 됨.)
#
# 파라미터:
#   arm  : model_deg = SIGN * real_deg + OFFSET  (SIGN=±1, scale 없음). env
#          AFFINE_<JOINT>_SIGN / AFFINE_<JOINT>_OFFSET. **기본 identity(1,0)** → 측정 전엔
#          arm 무변경(안전). 측정법은 docs/SIM_REAL_INFERENCE_PARITY.md §5.3 / scripts/test.
#   grip : 2점 anchor(같은 물리 자세를 sim·real 단위로). 기본 sim[-1.59,27=48.7°]↔real[1,51].
#          A_g=(SIM_OPEN-SIM_CLOSE)/(REAL_OPEN-REAL_CLOSE), B_g=SIM_CLOSE-A_g*REAL_CLOSE.
#
# GRIPPER_AFFINE 미설정 시 전체 무동작(sim 추론·real 학습 모델 경로 무영향). real 에 sim-모델
# 추론할 때만 =1. (※ 토글 이름은 하위호환으로 GRIPPER_AFFINE 유지 — arm offset 안 주면 gripper 만.)
if os.getenv("GRIPPER_AFFINE", "false").lower() in ("1", "true", "yes"):
    from lerobot.robots.so_follower.so_follower import SOFollower

    _f = float
    _ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

    # gripper forward(real→model) A,B. 직접 지정(GRIPPER_A/B, measure_joint_affine 피팅 출력)
    # 우선, 없으면 2점 anchor(close/open)에서 계산.
    _ga_env, _gb_env = os.getenv("GRIPPER_A"), os.getenv("GRIPPER_B")
    if _ga_env is not None and _gb_env is not None:
        _Ag, _Bg = _f(_ga_env), _f(_gb_env)
        if _Ag == 0.0:
            raise ValueError("GRIPPER_A 0 불가")
    else:
        _gsc = _f(os.getenv("GRIPPER_SIM_CLOSE", "-1.59"))
        _gso = _f(os.getenv("GRIPPER_SIM_OPEN", "27.0"))
        _grc = _f(os.getenv("GRIPPER_REAL_CLOSE", "1.0"))
        _gro = _f(os.getenv("GRIPPER_REAL_OPEN", "51.0"))
        if _gro == _grc or _gso == _gsc:
            raise ValueError("gripper anchor close==open → affine 기울기 정의 불가")
        _Ag = (_gso - _gsc) / (_gro - _grc)
        _Bg = _gsc - _Ag * _grc

    # per-joint (A, B) — forward: model = A*real + B
    _AFFINE: dict[str, tuple[float, float]] = {}
    for _j in _ARM_JOINTS:
        _sign = _f(os.getenv(f"AFFINE_{_j.upper()}_SIGN", "1.0"))
        _off = _f(os.getenv(f"AFFINE_{_j.upper()}_OFFSET", "0.0"))
        if _sign == 0.0:
            raise ValueError(f"AFFINE_{_j.upper()}_SIGN 0 불가")
        _AFFINE[f"{_j}.pos"] = (_sign, _off)  # model = sign*real + off
    _AFFINE["gripper.pos"] = (_Ag, _Bg)
    _GRIP_KEY = "gripper.pos"

    _orig_send_action = SOFollower.send_action
    _orig_get_observation = SOFollower.get_observation

    def _get_observation_affine(self):
        # robot 보고값(real 프레임) → 모델(sim 프레임). forward: model = A*real + B.
        obs = _orig_get_observation(self)
        for _k, (_A, _B) in _AFFINE.items():
            if _k in obs:
                obs[_k] = _A * float(obs[_k]) + _B
        return obs

    def _send_action_affine(self, action):
        # 모델 출력(sim 프레임) → robot(real 프레임). inverse: real = (model - B)/A. gripper 만 [0,100] 클램프.
        action = dict(action)
        for _k, (_A, _B) in _AFFINE.items():
            if _k in action:
                _v = (float(action[_k]) - _B) / _A
                action[_k] = min(100.0, max(0.0, _v)) if _k == _GRIP_KEY else _v
        return _orig_send_action(self, action)

    SOFollower.get_observation = _get_observation_affine
    SOFollower.send_action = _send_action_affine
    _arm_active = [j for j in _ARM_JOINTS if _AFFINE[f"{j}.pos"] != (1.0, 0.0)]
    print(
        f"[shim] joint affine ON (양방향): gripper model={_Ag:.3f}*real{_Bg:+.3f}; "
        f"arm offset 적용={_arm_active or '없음(identity)'}",
        flush=True,
    )


# robot_client 의 ``if __name__ == "__main__":`` 블록과 동일한 진입.
# async_client 는 @draccus.wrap() 데코레이터가 붙어 sys.argv 를 직접 파싱한다.
if __name__ == "__main__":
    register_third_party_plugins()
    async_client()
