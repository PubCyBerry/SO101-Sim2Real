"""IsaacLab 임시 패치 — leisaac ``utils/monkey_patch.py`` 이식 + 버전 게이트.

IsaacLab 의 ``TerminationManager.compute()`` 가 ``_term_dones`` 를 매 term 마다 올바르게
갱신하지 못하던 버그 패치. 한 번 True 가 된 ``_term_dones`` 가 False 로 되돌아가지 못해
종료 신호가 누적되던 문제(IsaacLab commit f498245 에서 수정됨).

leisaac 원본은 무조건 패치한다. 우리는 IsaacLab 2.3.2 가 **이미 수정본을 포함**하면 패치를
스킵한다(redundant override 로 공식 수정과 divergence 방지). 게이트 = 현재 ``compute`` 소스에
per-term 기록 라인(``_term_dones[:, i]``)이 있으면 fixed 로 간주.
"""

from __future__ import annotations


def patch_termination_manager() -> bool:
    """``TerminationManager.compute`` 를 수정본으로 교체. 이미 수정됐으면 스킵.

    Returns:
        패치를 실제로 적용했으면 True, (이미 fixed/임포트 불가로) 스킵했으면 False.
    """
    import inspect

    import torch
    from isaaclab.managers import TerminationManager

    # 버전 게이트: 현재 구현이 이미 per-term 으로 _term_dones 를 기록하면 수정본 → 스킵.
    try:
        current_src = inspect.getsource(TerminationManager.compute)
        if "_term_dones[:, i]" in current_src:
            return False
    except (OSError, TypeError):
        pass  # 소스 확인 불가 시 안전하게 패치 적용으로 진행

    def compute(self) -> torch.Tensor:
        # reset computation
        self._truncated_buf[:] = False
        self._terminated_buf[:] = False
        # iterate over all the termination terms
        for i, term_cfg in enumerate(self._term_cfgs):
            value = term_cfg.func(self._env, **term_cfg.params)
            # store timeout signal separately
            if term_cfg.time_out:
                self._truncated_buf |= value
            else:
                self._terminated_buf |= value
            # add to episode dones
            self._term_dones[:, i] = value  # [core fix]
            rows = value.nonzero(as_tuple=True)[0]  # indexing is cheaper than boolean advance indexing
            if rows.numel() > 0:
                self._term_dones[rows] = False
                self._term_dones[rows, i] = True
        # return combined termination signal
        return self._truncated_buf | self._terminated_buf

    TerminationManager.compute = compute
    return True


def monkey_patch() -> None:
    """모든 IsaacLab 임시 패치 적용."""
    patch_termination_manager()
