#!/bin/bash

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

EXPERIMENT_ID="p2_table1"
TIMESTAMP=$(date +"%Y%m%d_%H%M")
LOG_FILE="../logs/${EXPERIMENT_ID}_${TIMESTAMP}.log"

mkdir -p "$BASE_OUTPUT_DIR"

# 데이터셋 분할
# bash ../scripts/split-dataset.sh "$DATASET" 6B

# 조합 리스트 (형식: "iter dim size model")
# WS353_EN.csv
combinations=(
  "1 300 /20250805/wiki.ar.txt /tasks/WS353_AR.csv skip-gram"
  "1 300 /20250805/wiki.ar.txt /tasks/WS353_AR.csv cbow"
)

# 반복 실행
for combo in "${combinations[@]}"; do
  read ITER DIM DATASET TASK MODEL <<< "$combo"
  
  INPUT_FILE="../data/wikimedia${DATASET}"
  if [ ! -f "$INPUT_FILE" ]; then
    echo "[SKIP] $INPUT_FILE not found."
    continue
  fi

  OUTPUT_FILE="${BASE_OUTPUT_DIR}/${MODEL}_${DATASET}_${TASK}_${DIM}d_iter${ITER}.bin"
  LOG_FILE="${BASE_OUTPUT_DIR}/${MODEL}_${DATASET}_${TASK}_${DIM}d_iter${ITER}.log"

  echo "▶ Training Word2Vec ($MODEL) on $INPUT_FILE with dim=$DIM, iter=$ITER..." | tee -a "$LOG_FILE"
  
  if [ "$MODEL" == "cbow" ]; then
    CBOW_FLAG=1
  else
    CBOW_FLAG=0
  fi

  log_time "$LOG_FILE" ../word2vec/bin/word2vec -train "$INPUT_FILE" -output "$OUTPUT_FILE" \
    -cbow "$CBOW_FLAG" -size "$DIM" -window 5 -negative 5 -hs 0 -sample 1e-4 \
    -threads 10 -binary 1 -iter "$ITER" -min-count 5

  echo "▶ Evaluating accuracy for $OUTPUT_FILE" | tee -a "$LOG_FILE"
  log_time "$LOG_FILE" ../word2vec/bin/compute-spearman "$OUTPUT_FILE" ../data/${TASK}

  echo "✔ Done: $OUTPUT_FILE"
  echo ""
done
