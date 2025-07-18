
# objective: p1_table5 실험 시, train 과정에서 오랜 시간이 걸림. NS로 prediction하면, softmax로 연산하게 됨
# 그러나, HS로 연산하면, HS loss로 연산하게 되어, 속도가 빨라질 것이라는 가설
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

EXPERIMENT_ID="p1_exp1"
TIMESTAMP=$(date +"%Y%m%d_%H%M")
LOG_FILE="../logs/${EXPERIMENT_ID}_${TIMESTAMP}.log"

DATADIR=../data
LOGDIR=../logs
RESULTDIR=../output/${EXPERIMENT_ID}_${TIMESTAMP}

mkdir -p "${RESULTDIR}"
mkdir -p "${DATADIR}"
mkdir -p "${LOGDIR}"

LOG_FILE="${LOGDIR}/${EXPERIMENT_ID}_${TIMESTAMP}.log"



combinations=(
  "50 1"
  "50 2"
  "200 1"
  "200 2"
)

for combo in "${combinations[@]}";
do
        read DIM GRAM <<< "$combo"


  echo "Downloading dataset with dimensions ${DIM} and n-grams ${GRAM}"
  log_time "$LOG_FILE" ../fasttext supervised -input "${DATADIR}/YFCC100M/train-processing" \
    -output "${RESULTDIR}/dim${DIM}_gram${GRAM}" -dim ${DIM} -lr 0.25 -wordNgrams ${GRAM} \
    -minCount 100 -minCountLabel 100 -bucket 100000000 -epoch 5 -thread 20 -loss hs > /dev/null
log_time ${LOG_FILE} ../fasttext test "${RESULTDIR}/dim${DIM}_gram${GRAM}.bin" \
    "${DATADIR}/YFCC100M/test-processing" 
# log_time ${LOG_FILE} ../fasttext predict "../output/p1_table5_20250703_0921/dim50_gram1.bin" \
#     "${DATADIR}/YFCC100M/test-processing" 1
  # log_time "$LOG_FILE" ../fasttext supervised -input "${DATADIR}/YFCC100M/train-processing" \
  #   -output "${RESULTDIR}/dim${DIM}_gram${GRAM}" -dim ${DIM} -lr 0.25 -wordNgrams ${GRAM} \
  #   -minCount 1 -minCountLabel 1 -bucket 100000000 -epoch 5 -thread 20 -loss ns> /dev/null
done