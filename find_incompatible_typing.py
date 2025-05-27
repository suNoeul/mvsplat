import re
from pathlib import Path

# 감지할 정규표현식 패턴 (Python 3.10+ 문법)
patterns = [
    re.compile(r"\b\w+\s*\|\s*\w+"),             # A | B
    re.compile(r"\btuple\[[^\]]*\]"),            # tuple[int, ...]
    re.compile(r"\bdict\[[^\]]*\]"),             # dict[str, int]
    re.compile(r"\blist\[[^\]]*\]"),             # list[int]
]

# 검사할 파일들
files = list(Path("src").rglob("*.py"))

# 결과 저장
matches = []

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            for pattern in patterns:
                if pattern.search(line):
                    matches.append((file, i, line.strip()))
                    break  # 한 줄에서 여러 패턴 매치되도 한 번만 저장

# 출력
if matches:
    print(f"\n🔍 총 {len(matches)}개의 문법 충돌 라인 발견됨:\n")
    for file, lineno, line in matches:
        print(f"{file}:{lineno}: {line}")
else:
    print("✅ Python 3.8과 호환되지 않는 타입 힌트 문법이 발견되지 않았습니다.")
