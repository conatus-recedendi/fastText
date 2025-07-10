#!/bin/bash

# p1_table5

# 해당 로그에서 

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

EXPERIMENT_ID="p1_table5"
TIMESTAMP=$(date +"%Y%m%d_%H%M")
LOG_FILE="../logs/${EXPERIMENT_ID}_${TIMESTAMP}.log"

DATADIR=../data
LOGDIR=../logs
RESULTDIR=../output/${EXPERIMENT_ID}_${TIMESTAMP}

mkdir -p "${RESULTDIR}"
mkdir -p "${DATADIR}"
mkdir -p "${LOGDIR}"

LOG_FILE="${LOGDIR}/${EXPERIMENT_ID}_${TIMESTAMP}.log"

declare -A best_p1
declare -A best_lr


combinations=(
  "50 1 0.05"
  "50 2 0.05"
  "200 1 0.05"
  "200 2 0.05"

  "50 1 0.1"
  "50 2 0.1"
  "200 1 0.1"
  "200 2 0.1"

  "50 1 0.25"
  "50 2 0.25"
  "200 1 0.25"
  "200 2 0.25"

  "50 1 0.5"
  "50 2 0.5"
  "200 1 0.5"
  "200 2 0.5"
)

for combo in "${combinations[@]}";
do
read DIM GRAM LR <<< "$combo"

if [[ "$GRAM" == "2" ]]; then
  BUCKET=10000000  # 10M
else
  BUCKET=100000000 # 100M
fi

  OUTFILE="${RESULTDIR}/dim${DIM}_gram${GRAM}_lr${LR}"

  echo "Downloading dataset with dimensions ${DIM} and n-grams ${GRAM}"
  log_time "$LOG_FILE" ../fasttext supervised -input "${DATADIR}/YFCC100M/train-processing" \
    -output ${OUTFILE} -dim ${DIM} -lr ${LR} -wordNgrams ${GRAM} \
    -minCount 100 -minCountLabel 100 -bucket 10000000 -epoch 5 -thread 20 -loss hs > /dev/null

  # log_time "$LOG_FILE" echo "Testing on validation set"
  # OUTPUT=$(../fasttext test "${OUTFILE}.bin" "${DATADIR}/YFCC100M/valid-processing")
  # log_time "$LOG_FILE" echo "$OUTPUT"

  OUTPUT=$(log_time "$LOG_FILE" ../fasttext test "${OUTFILE}.bin" \
    "${DATADIR}/YFCC100M/valid-processing")

  # Extract P@1
  P1=$(echo "$OUTPUT" | awk '/P@1/ {print $2}')
  key="${DIM}_${GRAM}"
  if [[ -z "${best_p1[$key]}" || $(echo "$P1 > ${best_p1[$key]}" | bc -l) -eq 1 ]]; then
    best_p1[$key]=$P1
    best_lr[$key]=$LR
  fi
# log_time ${LOG_FILE} ../fasttext test "${RESULTDIR}/dim${DIM}_gram${GRAM}.bin" \
#     "${DATADIR}/YFCC100M/test-processing" 
# log_time ${LOG_FILE} ../fasttext predict "../output/p1_table5_20250703_0921/dim50_gram1.bin" \
#     "${DATADIR}/YFCC100M/test-processing" 1
  # log_time "$LOG_FILE" ../fasttext supervised -input "${DATADIR}/YFCC100M/train-processing" \
  #   -output "${RESULTDIR}/dim${DIM}_gram${GRAM}" -dim ${DIM} -lr 0.25 -wordNgrams ${GRAM} \
  #   -minCount 1 -minCountLabel 1 -bucket 100000000 -epoch 5 -thread 20 -loss ns> /dev/null
done


# Final result summary
log_time "$LOG_FILE"  echo -e "\n===== BEST RESULTS PER (DIM, GRAM) ====="
for key in "${!best_p1[@]}"; do

  log_time "$LOG_FILE" echo "DIM_GRAM: $key   Best P@1: ${best_p1[$key]}   LR: ${best_lr[$key]}"
  GRAM=$(echo "$key" | cut -d'_' -f2)
  DIM=$(echo "$key" | cut -d'_' -f1)

  OUTFILE="${RESULTDIR}/dim${DIM}_gram${GRAM}_lr${best_lr[$key]}"
  log_time "$LOG_FILE" ../fasttext test "${OUTFILE}.bin" \
    "${DATADIR}/YFCC100M/test-processing"
done | sort