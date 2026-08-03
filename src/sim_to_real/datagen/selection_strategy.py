"""source 데모 선택 전략 — 대칭 인지 최근접. **⚠ 폐기됨, 실측 기록용.**

폐루프에서 원본보다 **크게 나빴다**(증강 81.6 → 22.2 %). 아래 §폐기 참조. 재시도 금지.
파이프라인은 Isaac Lab 의 `nearest_neighbor_object` 를 그대로 쓴다.

## 왜 만들었나 (동기는 실측이었다)

Isaac Lab 의 `nearest_neighbor_object` 는 거리를

    pos_weight · ‖Δpos‖  +  rot_weight · ‖Δrot‖        (기본 1.0, 1.0)

로 재는데 두 가지가 어긋난다.

1. **스케일** — 위치는 0.02~0.46 m, 회전은 0~π rad 이라 회전이 10 배 지배한다. 그래서
   위치가 10 cm 넘게 먼 데모가 선택된다. 실측(p12 증강 47 trial): 선택된 source 와 새 배치의
   큐브 이동 **중앙 112.6 mm** — 위치만으로 고르면 11.4 mm 다.
2. **대칭** — 큐브는 z 축 4 회 대칭이라 90 ° 회전이 물리적으로 **같은 물체**다. 그런데
   `acos` 측지거리는 그걸 큰 차이로 센다. 접힌 yaw 어긋남 실측 최대 **44.2 °** — 집게가
   면이 아니라 **모서리를 향해** 내려간다는 뜻이다.

실패군에서 둘 다 크다(yaw 24.3 → 38.0 °, 이동 112.6 → 165.9 mm).

## 왜 `rot_weight` 를 낮추는 건 답이 아닌가

이전 A/B 에서 `rot_weight=0.05` 는 증강 성공률을 **76.1 → 52.4 %** 로 떨어뜨렸다. 회전은
중요하다 — 어느 face 를 잡느냐가 5-DOF 도달성을 가른다. 문제는 회전의 **가중치**가 아니라
**지표**다. 대칭을 접어서 재면 회전을 살리면서도 위치를 볼 수 있다.

## 하는 일

* yaw 를 `symmetry_fold` 로 접어(큐브 = 4 → 90 °) 물리적으로 같은 자세를 0 으로 만든다.
* 회전 거리를 **길이 단위로 환산**해 위치와 같은 축에서 더한다 — `rot_arm_m` 은 회전이
  집게 접촉점에서 만드는 변위의 지렛대(pad 이 TCP 에서 54 mm).

## 폐기 — 실측 결과

폐루프에서 **더 나빴다**: 증강 81.6 % → **22.2 %**, `pad_gap >30 mm` 6.2 % → **40.0 %**
(n=9, `p13`). 오프라인에선 큐브 이동이 128.6 → 21.2 mm 로 줄어 좋아 보였지만 같이 늘어난
yaw 어긋남(5.9 → 9.1 °)이 지배했다.

**결론**: 위치 변환량은 pad_gap 을 키우지 않는다. **어느 face 를 무느냐**(회전)가 5-DOF
도달성을 가르고 그게 성공률 절벽을 만든다. 앞선 `rot_weight=0.05` A/B(76.1 → 52~57 %)와
같은 방향이며, 그 실패의 원인도 "지표가 틀려서"가 아니라 **회전이 실제로 위치보다 훨씬
중요해서**였다. 원본 `nearest_neighbor_object` 를 그대로 쓴다.

되살리려면 env cfg 에서::

    selection_strategy="nearest_neighbor_object_symmetric"
    selection_strategy_kwargs={"nn_k": 1, "symmetry_fold": 4, "rot_arm_m": 0.054}

`python -m sim_to_real.datagen.selection_strategy` 로 self-check(2 종) 실행.
"""

from __future__ import annotations

import math

import torch

try:  # 런타임에는 Isaac Lab 레지스트리에 등록되어야 한다.
    from isaaclab_mimic.datagen.selection_strategy import SelectionStrategy
except ImportError:  # self-check 는 isaaclab 없이 돈다.
    SelectionStrategy = object  # type: ignore[assignment, misc]


def folded_yaw_distance(source_rot: torch.Tensor, target_rot: torch.Tensor,
                        symmetry_fold: int) -> torch.Tensor:
    """(N,3,3) ↔ (3,3) 의 z-yaw 차이를 대칭으로 접은 절대각(rad, N).

    큐브는 z 축 `symmetry_fold` 회 대칭이라 `2π/fold` 의 배수만큼 돌아간 자세는 같은 자세다.
    접은 뒤 남는 값은 `[0, π/fold]` 이므로 4 회 대칭이면 최대 45 °.
    """
    def yaw(rot: torch.Tensor) -> torch.Tensor:
        flat = rot.reshape(-1, 3, 3)
        return torch.atan2(flat[:, 1, 0], flat[:, 0, 0])

    period = 2.0 * math.pi / symmetry_fold
    delta = yaw(source_rot) - yaw(target_rot).reshape(1)
    folded = torch.remainder(delta + period / 2.0, period) - period / 2.0
    return folded.abs()


class NearestNeighborObjectSymmetricStrategy(SelectionStrategy):
    """물체 대칭을 접어 재는 최근접 — 위치와 회전을 같은 길이 단위로 더한다."""

    NAME = "nearest_neighbor_object_symmetric"

    def select_source_demo(
        self,
        eef_pose,
        object_pose,
        src_subtask_datagen_infos,
        pos_weight: float = 1.0,
        rot_arm_m: float = 0.054,
        symmetry_fold: int = 4,
        nn_k: int = 1,
    ):
        """`nearest_neighbor_object` 와 같은 시그니처 계약(추가 인자는 kwargs 로 온다).

        Returns:
            source_demo_ind (int)
        """
        poses = []
        for info in src_subtask_datagen_infos:
            values = list(info.object_poses.values())
            assert len(values) == 1, "subtask 당 object_ref 는 1개다"
            poses.append(values[0][0])          # 구간 **시작** 시점 물체 pose
        source_poses = torch.stack(poses)

        source_pos, source_rot = source_poses[:, :3, 3], source_poses[:, :3, :3]
        target_pos, target_rot = object_pose[:3, 3], object_pose[:3, :3]

        position = torch.linalg.vector_norm(source_pos - target_pos.reshape(1, 3), dim=-1)
        # 회전을 **길이로** 환산해 위치와 같은 축에서 더한다. rad 를 그대로 더하면 회전이
        # 10 배 지배해 위치가 사실상 무시된다(원본 전략의 실측 결함).
        rotation = folded_yaw_distance(source_rot, target_rot, symmetry_fold) * rot_arm_m
        distance = pos_weight * position + rotation

        k = max(1, min(int(nn_k), distance.shape[0]))
        top = torch.topk(distance, k=k, largest=False).indices
        return int(top[torch.randint(k, (1,))].item()) if k > 1 else int(top[0].item())


if __name__ == "__main__":
    def rz(deg: float) -> torch.Tensor:
        a = math.radians(deg)
        return torch.tensor([[math.cos(a), -math.sin(a), 0.0],
                             [math.sin(a), math.cos(a), 0.0],
                             [0.0, 0.0, 1.0]])

    # ① 4회 대칭: 0·90·180·270° 는 접으면 전부 0, 45° 가 최악(45°)
    target = rz(0.0)
    src = torch.stack([rz(d) for d in (0.0, 90.0, 180.0, 270.0, 45.0, -30.0, 100.0)])
    folded = folded_yaw_distance(src, target, 4)
    got = [round(math.degrees(float(v)), 3) for v in folded]
    assert got[:4] == [0.0, 0.0, 0.0, 0.0], f"90° 배수는 0 이어야: {got[:4]}"
    assert abs(got[4] - 45.0) < 1e-6, f"45° 는 45 여야: {got[4]}"
    assert abs(got[5] - 30.0) < 1e-6 and abs(got[6] - 10.0) < 1e-6, got[5:]
    # 허용오차 1e-6: torch 기본 float32 라 π/4 가 float64 값보다 5.6e-8 크게 담긴다.
    assert float(folded.max()) <= math.pi / 4 + 1e-6, "접은 값은 45° 를 못 넘는다"

    # ② 선택: 회전이 같은 두 후보 중 **위치가 가까운** 쪽을 고른다
    class _Info:
        def __init__(self, pose):
            self.object_poses = {"Cube1": pose.unsqueeze(0)}

    def pose(x: float, deg: float) -> torch.Tensor:
        out = torch.eye(4)
        out[:3, :3] = rz(deg)
        out[0, 3] = x
        return out

    infos = [_Info(pose(0.30, 0.0)), _Info(pose(0.01, 90.0)), _Info(pose(0.20, 45.0))]
    strategy = NearestNeighborObjectSymmetricStrategy()
    pick = strategy.select_source_demo(None, pose(0.0, 0.0), infos, nn_k=1)
    assert pick == 1, f"90°(대칭상 동일)+가까운 위치를 골라야: {pick}"

    # 원본 지표(rad 를 그대로 더함)라면 90° = 1.571 이라 먼 0° 후보를 고른다 — 회귀 대조
    naive = [0.30 + 0.0, 0.01 + math.pi / 2, 0.20 + math.pi / 4]
    assert min(range(3), key=lambda i: naive[i]) == 0, "대조군은 먼 후보를 골라야"

    print("selection_strategy self-check PASS (대칭 접기 7케이스 · 선택 2케이스)")
