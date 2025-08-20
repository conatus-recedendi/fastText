#!/bin/zsh

# Hyperparameter values
learning_rates=(0.05 0.1 0.25 0.5)
ngrams=(1 2)
iters=(5)

# Paths
TRAIN_FILE="../data/sogou_news.train"
TEST_FILE="../data/sogou_news.test"
BIN_DIR="bin"
OUT_BASE="../output/p1_test"
VOCAB_FILE="$OUT_BASE/sogou_news.vec"
FASTTEXT="$BIN_DIR/fastText"
TEST="$BIN_DIR/test"
LOG_FILE="$OUT_BASE/results_p1_table1_sogou_news.log"

# Prepare log file
echo "lr,ngram,iter,Precision,Recall,F1,time" >> $LOG_FILE
# [INFO] Total time taken for training: 9.10 seconds

# Run combinations
for lr in "${learning_rates[@]}"; do
  for ng in "${ngrams[@]}"; do
    for it in "${iters[@]}"; do
      model_path="$OUT_BASE/ag_news.bin"
      echo "[INFO] Training with lr=$lr, ngram=$ng, iter=$it..."

      # train_result=$($FASTTEXT -train $TRAIN_FILE \
      #           -output "${model_path:r}" \
      #           -size 10 \
      #           -lr $lr \
      #           -ngram $ng \
      #           -min-count 1 \
      #           -bucket 10000000 \
      #           -iter $it \
      #           -thread 1 \
      #           -save-vocab $VOCAB_FILE \
      #           -hs 0
      # )
      train_result=$($FASTTEXT -train $TRAIN_FILE \
                     -output "${model_path:r}" \
                     -size 10 \
                     -lr $lr \
                     -ngram $ng \
                     -min-count 1 \
                     -bucket 10000000 \
                     -iter $it \
                     -thread 1 \
                     -save-vocab $VOCAB_FILE \
                     -hs 0)

      $train_result=$(echo "$train_result" | sed 's/\(\[INFO\]\)/\n\1/g')
      # [INFO] Total time taken for training: 9.10 seconds
      # [INFO] Total time taken for training: 9.10 seconds
      # training_time=$(echo "$train_result" | grep "[INFO] Total time taken for training")
      training_time=$(echo "$train_result" | grep "Total time taken for training" | grep -oE '[0-9]+\.[0-9]+')

      echo "[INFO] Training time: $training_time seconds"


      echo "[INFO] Testing..."
      result=$($TEST -load-model $model_path \
                     -test-file $TEST_FILE \
                     -topk 1 \
                     -answer-threshold 0.0)

      # Extract performance line
      perf_line=$(echo "$result" | grep "Precision@K")
      precision=$(echo $perf_line | awk '{print $3}' | tr -d ',')
      recall=$(echo $perf_line | awk '{print $5}' | tr -d ',')
      f1=$(echo $perf_line | awk '{print $7}' | tr -d ',')

      echo "$lr,$ng,$it,$precision,$recall,$f1,$training_time" >> $LOG_FILE
      echo "[INFO] Logged result: P=$precision R=$recall F1=$f1 Time=$training_time"
    done
  done
done
