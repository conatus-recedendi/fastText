#include "vocab_word.h"
#include <time.h>
#ifndef CONFIG_H
#define CONFIG_H

#define MAX_STRING 400
#define BULK_SIZE 16284 // 16KB
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 16000 // 한줄에 최대  몇개 단어거ㅏ 오느닞
#define MAX_LABELS 100 // 한 문제에 최대 몇 개의 label이 존재할 수 있는지
#define MAX_WORDS_PER_SENTENCE 4000 // MAX sentence length랑 동일

#define MAX_LINE_LEN 1000000
#define SOFTMAX_MIN -64.0f

typedef struct {
  long long layer1_size; // hidden size/
  char train_file[MAX_STRING];
  char output_file[MAX_STRING];
  char save_vocab_file[MAX_STRING];
  char read_vocab_file[MAX_STRING];
  int binary;
  int debug_mode;
  int cbow;
  int window;
  int min_count;
  int num_threads;
  int min_reduce;
  int hs;
  int negative;
  int iter;
  long long classes;
  float learning_rate;
  float learning_rate_decay; // 
  float sample;
  long long word_count_actual;
  long long file_size;
  long long train_words; // total number of words in the training file
  long long update_word_count;
  float loss;
  long long ngram;
  long long bucket_size; // size of the bucket for ngram
  long long pure_vocab_size;
  struct timespec start; // start time for tsraining

  // for vocab
  int *vocab_hash;
  long long vocab_hash_size; // size of the hash tvocab_max_sizeable for vocabulary
  vocab_word *vocab;
  vocab_word *labels; // labels for classification
  int *label_hash; // hash table for labels
  long long label_hash_size; // size of the hash table for labels
  long long label_size; // current label size
  long long label_max_size; // maximum label size
  long long vocab_size; // current vocabulary size
  long long vocab_max_size; // maximum vocabulary size


  // for learning
  float *layer1; // word embedding weights. vocab size * layer1_size
  float *layer2; // hidden weights. layer1_size * class_sizef
  float *output; // output (from softmax) weights.  class_size * 1

  long long total_lines;
  long long *start_offsets; // start offsets for each thread
  long long *end_offsets; // end offsets for each thread
  long long *start_line_by_thread; // actual offset for each thread
  long long total_learned_lines; // total lines learned by all threads
  long long *total_line_by_thread; // total lines by each thread

  

  // for test
  long long top_k; // top k for classification
  float answer_threshold; // threshold for answer classification
  char test_file[MAX_STRING]; // test file path
  char load_model_file[MAX_STRING]; // model file path .bin
  char read_vec_file[MAX_STRING]; // vector file path for inference
  long long min_count_label; // minimum count for labels
  long long min_count_vocab; // minimum count for vocabulary

} global_setting;


extern global_setting gs;
#endif