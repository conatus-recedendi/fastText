#!/usr/bin/env bash
set -e

# 기본 환경 설정
export LANGUAGE=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

NOW=$(date +"%Y%m%d")
ROOT="data/wikimedia/${NOW}"
mkdir -p "${ROOT}"
echo "Saving data in ${ROOT}"

normalize_text() {
    sed -e "s/’/'/g" -e "s/′/'/g" -e "s/''/ /g" -e "s/'/ ' /g" -e "s/“/\"/g" -e "s/”/\"/g" \
        -e 's/"/ " /g' -e 's/\./ \. /g' -e 's/<br \/>/ /g' -e 's/, / , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' \
        -e 's/\?/ \? /g' -e 's/\;/ /g' -e 's/\:/ /g' -e 's/-/ - /g' -e 's/=/ /g' -e 's/*/ /g' -e 's/|/ /g' \
        -e 's/«/ /g' | tr 0-9 " "
}

# -------- 데이터 다운로드 --------
echo "Downloading Europarl v7..."
wget -c https://www.statmt.org/wmt13/training-monolingual-europarl-v7.tgz -P "${ROOT}"
echo "Downloading News Crawl v8..."
wget -c https://www.statmt.org/wmt13/training-monolingual-nc-v8.tgz -P "${ROOT}"
echo "Downloading News 2012..."
wget -c https://www.statmt.org/wmt13/training-monolingual-news-2012.tgz -P "${ROOT}"

# -------- 압축 해제 --------
echo "Extracting datasets..."
tar -xvzf "${ROOT}/training-monolingual-europarl-v7.tgz" -C "${ROOT}"
tar -xvzf "${ROOT}/training-monolingual-nc-v8.tgz" -C "${ROOT}"
tar -xvzf "${ROOT}/training-monolingual-news-2012.tgz" -C "${ROOT}"

# -------- 전처리 --------
# Europarl v7 + NC v8 → 하나의 파일로
echo "Preprocessing Europarl v7 + NC v8..."
cat "${ROOT}"/training/europarl-v7.*.gz \
    "${ROOT}"/training/news.20??.shuffled.gz \
    | gunzip -c \
    | awk '{print tolower($0)}' \
    | normalize_text \
    | awk '{if (NF>1) print;}' \
    | tr -s " " \
    | shuf \
    > "${ROOT}/wiki.europal_ncv8.txt"

# News 2012 → 단독 파일
echo "Preprocessing News 2012..."
cat "${ROOT}"/training/news.2012.*.gz \
    | gunzip -c \
    | awk '{print tolower($0)}' \
    | normalize_text \
    | awk '{if (NF>1) print;}' \
    | tr -s " " \
    | shuf \
    > "${ROOT}/wiki.news2012.txt"

echo "Done!"
echo "Output files:"
echo " - ${ROOT}/wiki.europal_ncv8.txt"
echo " - ${ROOT}/wiki.news2012.txt"
