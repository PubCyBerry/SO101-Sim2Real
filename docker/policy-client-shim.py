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

# robot_client 의 ``if __name__ == "__main__":`` 블록과 동일한 진입.
# async_client 는 @draccus.wrap() 데코레이터가 붙어 sys.argv 를 직접 파싱한다.
if __name__ == "__main__":
    register_third_party_plugins()
    async_client()
