#!/usr/bin/env python3
import os


def split_file_by_percent(path, percents=(1, 5, 20, 50)):
    dirname = os.path.dirname(path)
    basename = os.path.basename(path)  # 예: wiki.en.txt
    prefix, ext = os.path.splitext(basename)

    # 1단계: 전체 라인 수 세기
    total = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            total += 1
    print(f"[INFO] {basename}: total {total} lines")

    # 2단계: percents별로 목표 라인 수 계산
    targets = {p: max(1, int(total * p / 100)) for p in percents}
    written = {p: 0 for p in percents}
    files = {p: open(f"{prefix}.{p}%.txt", "w", encoding="utf-8") for p in percents}

    # 3단계: 파일 다시 읽으면서 쓰기
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f, start=1):
            for p in percents:
                if written[p] < targets[p]:
                    files[p].write(line)
                    written[p] += 1
            # 모든 target을 다 채우면 종료
            if all(written[p] >= targets[p] for p in percents):
                break

    for f in files.values():
        f.close()

    for p in percents:
        print(f"[INFO] Wrote {prefix}.{p}%.txt with {written[p]} lines")


if __name__ == "__main__":
    split_file_by_percent("wiki.en.txt", percents=(1, 5, 20, 50))
