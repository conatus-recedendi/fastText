#!/usr/bin/env bash
set -euo pipefail

export LANGUAGE=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

NOW=$(date +"%Y%m%d")
ROOT="data/wikimedia/${NOW}"
mkdir -p "${ROOT}"
echo "[INFO] Saving data in ${ROOT}"

normalize_text() {
  sed -e "s/’/'/g" -e "s/′/'/g" -e "s/''/ /g" -e "s/'/ ' /g" -e "s/“/\"/g" -e "s/”/\"/g" \
      -e 's/"/ " /g' -e 's/\./ \. /g' -e 's/<br \/>/ /g' -e 's/, / , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' \
      -e 's/\?/ \? /g' -e 's/\;/ /g' -e 's/\:/ /g' -e 's/-/ - /g' -e 's/=/ /g' -e 's/*/ /g' -e 's/|/ /g' \
      -e 's/«/ /g' | tr 0-9 " "
}

# .gz/평문 모두 지원하는 안전 스트리머
stream_files() {
  local any=0
  for f in "$@"; do
    [[ -e "$f" ]] || continue
    any=1
    case "$f" in
      *.gz) gzip -cd -- "$f" ;;
      *)    cat -- "$f" ;;
    esac
  done
  # 아무 파일도 없으면 빈 스트림
  if [[ $any -eq 0 ]]; then
    return 0
  fi
}

# 보통 구조: ${ROOT}/training/ (혹은 training-monolingual/)
TRAIN_DIR="${ROOT}/training"
TRAIN_DIR_MONO="${ROOT}/training-monolingual"
shopt -s nullglob

langs=(de en es fr)

for L in "${langs[@]}"; do
  echo "[INFO] Processing language=${L}"

  # 가능한 패턴들을 모두 나열 (압축/비압축 모두)
  europarl_files=( \
    "${TRAIN_DIR}/europarl-v7.${L}.gz" \
    "${TRAIN_DIR}/europarl-v7.${L}" \
    "${TRAIN_DIR_MONO}/europarl-v7.${L}.gz" \
    "${TRAIN_DIR_MONO}/europarl-v7.${L}" \
  )
  nc_files=( \
    "${TRAIN_DIR}/news-commentary-v8.${L}.gz" \
    "${TRAIN_DIR}/news-commentary-v8.${L}" \
    "${TRAIN_DIR_MONO}/news-commentary-v8.${L}.gz" \
    "${TRAIN_DIR_MONO}/news-commentary-v8.${L}" \
  )
  news2012_files=( \
    "${TRAIN_DIR}/news.2012.${L}.shuffled.gz" \
    "${TRAIN_DIR}/news.2012.${L}.shuffled" \
    "${TRAIN_DIR_MONO}/news.2012.${L}.shuffled.gz" \
    "${TRAIN_DIR_MONO}/news.2012.${L}.shuffled" \
  )

  # 존재 검사 로그
  echo "  [DBG] Europarl candidates:"
  printf '    - %s\n' "${europarl_files[@]}" | sed '/- $/d' || true
  echo "  [DBG] NC v8 candidates:"
  printf '    - %s\n' "${nc_files[@]}" | sed '/- $/d' || true
  echo "  [DBG] News2012 candidates:"
  printf '    - %s\n' "${news2012_files[@]}" | sed '/- $/d' || true

  # ---- 1) Europarl v7 + News-Commentary v8 합본 ----
  out_combo="${ROOT}/wiki.${L}.europal_ncv8.txt"
  echo "[INFO]   Europarl+NC → ${out_combo}"
  {
    stream_files "${europarl_files[@]}"
    stream_files "${nc_files[@]}"
  } \
  | awk '{print tolower($0)}' \
  | normalize_text \
  | awk 'NF>1' \
  | tr -s " " \
  | tee >(wc -l >&2 >/dev/null) \
  | shuf \
  > "${out_combo}"

  # ---- 2) News 2012 단독 ----
  out_news2012="${ROOT}/wiki.${L}.news2012.txt"
  echo "[INFO]   News2012 → ${out_news2012}"
  stream_files "${news2012_files[@]}" \
  | awk '{print tolower($0)}' \
  | normalize_text \
  | awk 'NF>1' \
  | tr -s " " \
  | tee >(wc -l >&2 >/dev/null) \
  | shuf \
  > "${out_news2012}"

  # 크기 확인
  for f in "${out_combo}" "${out_news2012}"; do
    if [[ -s "$f" ]]; then
      echo "  [OK] $(basename "$f"): $(wc -l < "$f") lines"
    else
      echo "  [WARN] $(basename "$f") is empty. 입력 파일 패턴/경로를 확인하세요."
    fi
  done
done

echo "[INFO] Done."
