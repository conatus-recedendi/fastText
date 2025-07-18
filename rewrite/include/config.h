#include "vocab_word.h"
#include <time.h>
#ifndef CONFIG_H
#define CONFIG_H

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40
#define MAX_LABELS 1000
#define MAX_WORDS_PER_SENTENCE 1000
 
#define MAX_LINE_LEN 1000000

typedef struct {
  long long layer1_size; // hidden size/
  long long class_size; // number of classes
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
  clock_t start; // start time for training

  // for vocab
  int *vocab_hash;
  long long vocab_hash_size; // size of the hash table for vocabulary
  vocab_word *vocab;
  long long vocab_size; // current vocabulary size
  long long vocab_max_size; // maximum vocabulary size

  // for learning
  float *layer1; // word embedding weights. vocab size * layer1_size
  float *layer2; // hidden weights. layer1_size * class_sizef
  float *output; // output (from softmax) weights.  class_size * 1

  long long total_lines;
  long long *start_offsets;
  long long *end_offsets;

} global_setting;


extern global_setting gs;
#endif