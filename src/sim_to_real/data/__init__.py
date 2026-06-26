"""LeRobot v3 데이터 라이브러리 (writer + 단위/카메라 helper).

VLA-only 리팩토링: sim 데이터 생성기(RL rollout·cuRobo)는 제거됐고, 이 writer 는
sim teleop / 실기기 record 기반 향후 데이터 생성용 라이브러리로 보존한다(현재 producer 없음).
단위 codec 은 ``so101_contract`` (affine) 단일 소스를 사용한다.
"""
