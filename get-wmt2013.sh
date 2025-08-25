#!/usr/bin/env bash
set -euo pipefail

# ===== locale =====
export LANGUAGE=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

# ===== output root =====
NOW=$(date +"%Y%m%d")
ROOT="data/wikimedia/${NOW}"
mkdir -p "${ROOT}"
echo "[INFO] Saving data in ${ROOT}"

# ===== normalizer =====
normalize_text() {
  sed -e "s/’/'/g" -e "s/′/'/g" -e "s/''/ /g" -e "s/'/ ' /g" -e "s/“/\"/g" -e "s/”/\"/g" \
      -e 's/"/ " /g' -e 's/\./ \. /g' -e 's/<br \/>/ /g' -e 's/, / , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' \
      -e 's/\?/ \? /g' -e 's/\;/ /g' -e 's/\:/ /g' -e 's/-/ - /g' -e 's/=/ /g' -e 's/*/ /g' -e 's/|/ /g' \
      -e 's/«/ /g' | tr 0-9 " "
}

# ===== download =====
echo "[INFO] Downloading archives (resume enabled)..."
wget -c https://www.statmt.org/wmt13/training-monolingual-europarl-v7.tgz    -P "${ROOT}"
wget -c https://www.statmt.org/wmt13/training-monolingual-nc-v8.tgz          -P "${ROOT}"
wget -c https://www.statmt.org/wmt13/training-monolingual-news-2012.tgz      -P "${ROOT}"

# ===== extract =====
echo "[INFO] Extracting..."
tar -xzf "${ROOT}/training-monolingual-europarl-v7.tgz"   -C "${ROOT}"
tar -xzf "${ROOT}/training-monolingual-nc-v8.tgz"         -C "${ROOT}"
tar -xzf "${ROOT}/training-monolingual-news-2012.tgz"     -C "${ROOT}"

# 보통 ${ROOT}/training/ 아래로 풀립니다.
TRAIN_DIR="${ROOT}/training"
if [[ ! -d "${TRAIN_DIR}" ]]; then
  echo "[WARN] Expected ${ROOT}/training directory not found. Checking fallback..."
  TRAIN_DIR="${ROOT}/training-monolingual"
  # 일부 배포는 다른 폴더명을 쓸 수 있어 필요시 수정하세요.
fi

# ===== helper: zcat 여러 파일을 안전하게 합치기 =====
zcat_many() {
  # 인자: 파일 리스트
  # 파일이 하나도 없으면 빈 스트림을 리턴
  if [[ $# -eq 0 ]]; then
    return 0
  fi
  zcat -- "$@" 2>/dev/null || true
}

shopt -s nullglob
langs=(de en es fr)

for L in "${langs[@]}"; do
  echo "[INFO] Processing language=${L}"

  # ---- 경로/패턴 (WMT13 일반적 파일명 규칙) ----
  europarl_files=( "${TRAIN_DIR}/europarl-v7.${L}.gz" )
  # NC v8: 여러 해가 섞여 있음: news.2007.L.shuffled.gz ~ news.2011.L.shuffled.gz 등
  # news-commentary-v8.ar
  nc_files=( "${TRAIN_DIR}"/news-commentary-v8.${L}.gz )
  # News 2012:
  news2012_files=( "${TRAIN_DIR}"/news.2012.${L}.shuffled.gz )

  # ---- 1) Europarl v7 + NC v8 합본 ----
  out_combo="${ROOT}/wiki.${L}.europal_ncv8.txt"
  echo "[INFO]   Europarl+NC → ${out_combo}"

  # 유효한 입력이 있는지 체크
  if [[ ${#europarl_files[@]} -eq 0 && ${#nc_files[@]} -eq 0 ]]; then
    echo "[WARN]   No Europarl v7 or NC v8 files for ${L}. Skipping combo."
  else
    {
      zcat_many "${europarl_files[@]}"
      zcat_many "${nc_files[@]}"
    } \
    | awk '{print tolower($0)}' \
    | normalize_text \
    | awk '{if (NF>1) print;}' \
    | tr -s " " \
    | shuf \
    > "${out_combo}"
  fi

  # ---- 2) News 2012 단독 ----
  out_news2012="${ROOT}/wiki.${L}.news2012.txt"
  echo "[INFO]   News2012 → ${out_news2012}"

  if [[ ${#news2012_files[@]} -eq 0 ]]; then
    echo "[WARN]   No News 2012 files for ${L}. Skipping news2012."
  else
    zcat_many "${news2012_files[@]}" \
    | awk '{print tolower($0)}' \
    | normalize_text \
    | awk '{if (NF>1) print;}' \
    | tr -s " " \
    | shuf \
    > "${out_news2012}"
  fi
done

echo "[INFO] Done."
echo "[INFO] Outputs:"
for L in "${langs[@]}"; do
  [[ -f "${ROOT}/wiki.${L}.europal_ncv8.txt" ]] && echo " - ${ROOT}/wiki.${L}.europal_ncv8.txt"
  [[ -f "${ROOT}/wiki.${L}.news2012.txt"     ]] && echo " - ${ROOT}/wiki.${L}.news2012.txt"
done
