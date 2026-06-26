# scratch/ — 임시 작업 공간 (커밋 안 함)

이 디렉터리는 **버려도 되는 임시물** 전용이다. `.gitignore` 가 `scratch/*` 를 추적 제외하므로(이 `README.md` 만 예외) 여기 둔 파일은 커밋되지 않는다.

## 여기에 두는 것

- smoke test / 일회성 검증 스크립트
- 디버그 덤프, 로그, faulthandler 크래시 txt
- 실험 산출물, 중간 결과, 한 번 쓰고 버릴 plot/csv

## 규칙 (AGENTS.md §운영 규칙)

1. **임시물은 전부 `scratch/` 에.** repo 루트나 `scripts/` 아무 데나 흩뿌리지 않는다.
2. **작업이 끝나면 정리한다.** 영구히 둘 가치가 있으면 `scripts/<범주>/` 로 옮기고(promote) AGENTS.md 스크립트 표에 등재한다. 아니면 삭제한다.
3. 하위 구조는 자유. 예: `scratch/2026-06-26-grasp-jitter/`.

> 영구 코드는 `scratch/` 가 아니라 반드시 `scripts/<범주>/`(assets·contract·data·environments·inference) 아래에 둔다. — anti-fragmentation 규칙.
