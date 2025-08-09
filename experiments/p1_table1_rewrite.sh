#!/usr/bin/env bash


# 개발 시 debugging 용!

# bin/fastText -train ../data/ag_news.train     -output ../output/p1_test/ag_news.bin -size 10 -lr 0.25 -ngram 2     -min-count-vocab 1 -min-count-label 1 -bucket 10000000 -iter 5 -thread 4 -save-vocab ../output/p1_test/ag_news.vec -hs 1
# bin/fastText -train ../data/ag_news.train -output ../result/p1_test/ag_news.bin -size 10 -lr 0.25 -ngram 2 -min-count-vocab 100 -min-count-label 100 -bucket 100000000 -iter 5 -thread 1 -hs 0
# bin/fastText -train ../data/sogou_news.train     -output ../output/p1_test/sogou_news.bin -size 10 -lr 0.25 -ngram 2     -min-count-vocab 1 -min-count-label 1 -bucket 10000000 -iter 5 -thread 1 -save-vocab ../output/p1_test/sogou_news.vec -hs 1

# bin/test -load-model ../output/p1_test/ag_news.bin  -test-file ../data/ag_news.test -topk 1 -answer-threshold 0.0

# 로그 함수 정의
log_time() {
        logfile="$1"
        shift
        echo "Running: $*" | tee -a "$logfile"
        start=$(date +%s)
        "$@" 2>&1 | tee /dev/tty | awk 'index($0, "\r") == 0' >> "$logfile"
        end=$(date +%s)
        echo "Time elapsed: $((end - start))s" | tee -a "$logfile"
        echo "" | tee -a "$logfile"
}

EXPERIMENT_ID="p1_test"
TIMESTAMP=$(date +"%Y%m%d_%H%M")

DATADIR=../data
LOGDIR=../logs
RESULTDIR=../output/${EXPERIMENT_ID}

mkdir -p "${RESULTDIR}"
mkdir -p "${DATADIR}"
mkdir -p "${LOGDIR}"


LOG_FILE="${LOGDIR}/${EXPERIMENT_ID}.log"


myshuf() {
  perl -MList::Util=shuffle -e 'print shuffle(<>);' "$@";
}

normalize_text() {
  tr '[:upper:]' '[:lower:]' | sed -e 's/^/__label__/g' | \
    sed -e "s/'/ ' /g" -e 's/"//g' -e 's/\./ \. /g' -e 's/<br \/>/ /g' \
        -e 's/,/ , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' \
        -e 's/\?/ \? /g' -e 's/\;/ /g' -e 's/\:/ /g' | tr -s " " | myshuf
}

DATASET=(
  ag_news
  sogou_news
  dbpedia
  yelp_review_polarity
  yelp_review_full
  yahoo_answers
  amazon_review_full
  amazon_review_polarity
)

ID=(
  0Bz8a_Dbh9QhbUDNpeUdjb0wxRms # ag_news
  0Bz8a_Dbh9QhbUkVqNEszd0pHaFE # sogou_news
  0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k # dbpedia
  0Bz8a_Dbh9QhbNUpYQ2N3SGlFaDg # yelp_review_polarity
  0Bz8a_Dbh9QhbZlU4dXhHTFhZQU0 # yelp_review_full
  0Bz8a_Dbh9Qhbd2JNdDBsQUdocVU # yahoo_answers
  0Bz8a_Dbh9QhbZVhsUnRWRDhETzA # amazon_review_full
  0Bz8a_Dbh9QhbaW12WVVZS2drcnM # amazon_review_polarity
)

# These learning rates were chosen by validation on a subset of the training set.
LR=( 0.1 0.1 0.1 0.1 0.1 0.1 0.05 0.05 )

# Small datasets first

for i in {0..0}
do
  echo "Downloading dataset ${DATASET[i]}"
  if [ ! -f "${DATADIR}/${DATASET[i]}.train" ]
  then
    wget -c "https://drive.google.com/uc?export=download&id=${ID[i]}" -O "${DATADIR}/${DATASET[i]}_csv.tar.gz"
    tar -xzvf "${DATADIR}/${DATASET[i]}_csv.tar.gz" -C "${DATADIR}"
    cat "${DATADIR}/${DATASET[i]}_csv/train.csv" | normalize_text > "${DATADIR}/${DATASET[i]}.train"
    cat "${DATADIR}/${DATASET[i]}_csv/test.csv" | normalize_text > "${DATADIR}/${DATASET[i]}.test"
  fi
done

# Large datasets require a bit more work due to the extra request page

for i in {1..7}
do
  echo "Downloading dataset ${DATASET[i]}"
  if [ ! -f "${DATADIR}/${DATASET[i]}.train" ]
  then
    # curl -c /tmp/cookies "${ID[i]}" > /tmp/intermezzo.html
    # echo "https://drive.google.com$(cat /tmp/intermezzo.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')"
    # curl -L -b /tmp/cookies "https://drive.google.com$(cat /tmp/intermezzo.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')" > "${DATADIR}/${DATASET[i]}_csv.tar.gz"
    curl -L -b /tmp/cookies "https://drive.usercontent.google.com/download?id=${ID[i]}&export=download&authuser=0&confirm=t" > "${DATADIR}/${DATASET[i]}_csv.tar.gz"
    tar -xzvf "${DATADIR}/${DATASET[i]}_csv.tar.gz" -C "${DATADIR}"
    cat "${DATADIR}/${DATASET[i]}_csv/train.csv" | normalize_text > "${DATADIR}/${DATASET[i]}.train"
    cat "${DATADIR}/${DATASET[i]}_csv/test.csv" | normalize_text > "${DATADIR}/${DATASET[i]}.test"
  fi
done

make

THRESHOLD=0.08

for i in {0..7}
do
  log_time ${LOG_FILE} echo "Working on dataset ${DATASET[i]} for bigrams, threshold ${THRESHOLD}"
  # echo "../rewrite/bin/fastText -train "${DATADIR}/${DATASET[i]}.train" \
  #   -output "${RESULTDIR}/${DATASET[i]}" -size 10 -lr "${LR[i]}" -wordNgrams 1 \
  #   -min-count 1 -bucket 10000000 -iter 5 -thread 20"

  log_time ${LOG_FILE} ../rewrite/bin/fastText  -train "${DATADIR}/${DATASET[i]}.train" \
    -output "${RESULTDIR}/${DATASET[i]}_bi.bin" -size 10 -lr "${LR[i]}" -wordNgrams 2 \
    -min-count 1 -bucket 10000000 -iter 5 -thread 1 -save-vocab "${RESULTDIR}/${DATASET[i]}_bi.vec"

  # log_time ${LOG_FILE} ../rewrite/bin/fastText -train "${DATADIR}/${DATASET[i]}.train" \
  #   -output "${RESULTDIR}/${DATASET[i]}" -size 10 -lr "${LR[i]}" -wordNgrams 2 \
    # -min-count 1 -bucket 10000000 -iter 5 -thread 20 > /dev/null
  log_time ${LOG_FILE} ../rewrite/bin/test -load-model "${RESULTDIR}/${DATASET[i]}_bi.bin" -test-file "${DATADIR}/${DATASET[i]}.test" -topk 1 -answer-threshold ${THRESHOLD}
done


for i in {0..7}
do
  log_time ${LOG_FILE} echo "Working on dataset ${DATASET[i]} for 1grams, threshold ${THRESHOLD}"
  # echo "../rewrite/bin/fastText -train "${DATADIR}/${DATASET[i]}.train" \
  #   -output "${RESULTDIR}/${DATASET[i]}" -size 10 -lr "${LR[i]}" -wordNgrams 1 \
  #   -min-count 1 -bucket 10000000 -iter 5 -thread 20"

  log_time ${LOG_FILE} ../rewrite/bin/fastText  -train "${DATADIR}/${DATASET[i]}.train" \
    -output "${RESULTDIR}/${DATASET[i]}.bin" -size 10 -lr "${LR[i]}" -wordNgrams 1 \
    -min-count 1 -bucket 100000000 -iter 5 -thread 1 -save-vocab "${RESULTDIR}/${DATASET[i]}.vec"

  # log_time ${LOG_FILE} ../rewrite/bin/fastText -train "${DATADIR}/${DATASET[i]}.train" \
  #   -output "${RESULTDIR}/${DATASET[i]}" -size 10 -lr "${LR[i]}" -wordNgrams 2 \
    # -min-count 1 -bucket 10000000 -iter 5 -thread 20 > /dev/null
  log_time ${LOG_FILE} ../rewrite/bin/test -load-model "${RESULTDIR}/${DATASET[i]}.bin" -test-file "${DATADIR}/${DATASET[i]}.test" -topk 1 -answer-threshold ${THRESHOLD}
done

# for i in {0..7}
# do
#   log_time ${LOG_FILE} echo "Working on dataset ${DATASET[i]} for 1-grams"
#   log_time ${LOG_FILE} ../rewrite/bin/fastText -train "${DATADIR}/${DATASET[i]}.train" \
#     -output "${RESULTDIR}/${DATASET[i]}_1gram" -size 10 -lr "${LR[i]}" -wordNgrams 1 \
#     -min-count 1 -bucket 100000000 -iter 5 -thread 20 > /dev/null
#   log_time ${LOG_FILE} ../rewrite/bin/test "${RESULTDIR}/${DATASET[i]}_1gram.bin" \
#     "${DATADIR}/${DATASET[i]}.test"
# done
