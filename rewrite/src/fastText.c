
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <math.h>
#include <ctype.h>
#include "config.h"
#include "vocab.h"


typedef struct {
  long long id; // thread id
  global_setting *gs; // global settings
} thread_args;

/**
  * get arg_pos
  * idx = get_arg_pos("--size"), argc, argv)
 */
int get_arg_pos(char *str, int argc, char **argv) {
  int i = 0;

  for (i = 1; i < argc; i++) {
    if (!strcmp(str, argv[i])) {
      if (i == argc - 1) {
        printf("Argument missing for %s\n", str);
        exit(1);
      }
      return i;
    }
  }
  return -1;
}

void initialize_network(global_setting *gs) {
  printf("[INFO] Initializing network... %lld %lld \n", gs->vocab_size, gs->layer1_size);
  posix_memalign((void **)&(gs->layer1), 64, (long long)gs->vocab_size * gs->layer1_size * sizeof(float));
  if (gs->layer1 == NULL) {
    fprintf(stderr, "Memory allocation failed for layer1\n");
    exit(1);
  }
  printf("[INFO] Allocated memory for layer1 with size: %lld\n", gs->vocab_size * gs->layer1_size * sizeof(float));
  posix_memalign((void **)&(gs->layer2), 64, gs->layer1_size * gs->class_size * sizeof(float));
  if (gs->layer2 == NULL) {
    fprintf(stderr, "Memory allocation failed for layer2\n");
    exit(1);
  }
  printf("[INFO] Allocated memory for layer2 with size: %lld\n", gs->layer1_size * gs->class_size * sizeof(float));
  posix_memalign((void **)&(gs->output), 64, gs->class_size * sizeof(float));
  if (gs->output == NULL) {
    fprintf(stderr, "Memory allocation failed for output\n");
    exit(1);
  }
  printf("[INFO] Network initialized with layer1 size: %lld, class size: %lld\n", gs->layer1_size, gs->class_size);
  create_binary_tree(gs);
  return ;
}

void save_vector(char *output_file, global_setting *gs) {
  // Implement the logic to save the vector representations
  FILE *fo = fopen(output_file, "w");
  for (long long i = 0; i < gs->vocab_size; i++) {
    fprintf(fo, "%s ", gs->vocab[i].word);
    for (long long j = 0; j < gs->layer1_size; j++) {
      fprintf(fo, "%f ", gs->layer1[i * gs->layer1_size + j]);
    }
    fprintf(fo, "\n");
  }
}

void save_model(char *output_file, global_setting *gs) {
  // Implement the logic to save the model
  FILE *fo = fopen(output_file, "wb");
  if (fo == NULL) {
    printf("Error opening file %s for writing\n", output_file);
    exit(1);
  }
  // fwrite(gs, sizeof(global_setting), 1, fo);
  fclose(fo);
}

void *train_thread(thread_args *args) {

  long long thread_id = (long long)args->id;
  global_setting *gs = (global_setting *)args->gs;
  // Placeholder for thread-specific training logic

  long long file_size = gs->file_size;
  int num_threads = gs->num_threads;

  long long word_count = 0;
  long long last_word_count = 0;

  FILE *fi = fopen(gs->train_file, "rb");
  fseek(fi, file_size / (long long)num_threads * (long long)thread_id , SEEK_END);
  for (long long i = 0; i < gs->iter; i++) {
    // Placeholder for training logic
    printf("Thread %lld: Training iteration %lld\n", thread_id, i);
    
    if (word_count - last_word_count > gs->update_word_count) {
        // Update word count and print progress
      gs->word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if (gs->debug_mode > 1) {
        clock_t now = clock();
        printf("%lr: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ",
              13, gs->learning_rate_decay,
              gs->word_count_actual / (double)(gs->iter * gs->train_words + 1) * 100,
              gs->word_count_actual / ((double)(now - gs->start + 1) / (double)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }

      // Update the learning rate
      if (gs->learning_rate > 0) {
        gs->learning_rate_decay = gs->learning_rate * (1 - (double)gs->word_count_actual / (double)(gs->iter * gs->train_words + 1));
        if (gs->learning_rate_decay < 0) gs->learning_rate_decay = 0;
      }

    }
    // TODO
    // Implement the training logic here


  }


  fseek(fi, 0, SEEK_SET);
  // Implement the saving output here
  return NULL;
}



void train_model(global_setting *gs) {
  // Placeholder for training model logic
  printf("[INFO] Training model with layer size: %lld\n", gs->layer1_size);
  // Implement the training logic here
  char *read_vocab_file = gs->read_vocab_file;
  char *save_vocab_file = gs->save_vocab_file;
  char *output_file = gs->output_file;
  gs->learning_rate_decay = gs->learning_rate;

  printf("[INFO] Initializing threads...\n");
  pthread_t *pt = (pthread_t *)malloc(gs->num_threads * sizeof(pthread_t));

  printf("[INFO] read vocabulary...\n");

  if (read_vocab_file[0] != 0) {
    // Read vocabulary from file
    read_vocab(gs);
  } else {
    // Create vocabulary from training file
    create_vocab_from_train_file(gs);
  }

  printf("[INFO] save vocabulary...\n");

  if (save_vocab_file[0] != 0) {
    // Save vocabulary to file
    save_vocab(gs);
  }


  if (output_file[0] == 0) {
    printf("No output file specified. Exiting.\n");
    return;
  }

  printf("[INFO] Initializing network...\n");

  initialize_network(gs);

  // TODO: make unigram table

  gs->start = clock();

  printf("[INFO] Starting training threads...\n");

  for (int i = 0; i < gs->num_threads; i++) {
    // Create threads for training
    pthread_create(&pt[i], NULL, train_thread, (thread_args *){i, gs});

  }
  for (int i = 0; i < gs->num_threads; i++) {
    // Wait for threads to finish
    pthread_join(pt[i], NULL);
  }
  free(pt);


  // TODO: output 쓰기.
  // 1. embedding 논문
  // 2. classification 논문

  save_model(output_file, gs);
  save_vector(output_file, gs);
  // foramt
  // __label1__ __label2__ ... input
  // 2-1. input infeernece 
  // 2-2. softmax 구한 뒤 top-K(max) class 추출(threshold)
  // 2-3. 실제 정답인지 여부 확인 
  // 위 과정을 구하기 위해서 초기값에 줘야 하는 것들
  // .bin 파일 -> 모델 argument
  // feature matrix size: vocab * dim
  // hidden size: dim * class

  // test(line, topK, threshold, )
}

int main(int argc, char **argv) {
  int i;

  global_setting gs = {
    .layer1_size = 10, // Default layer size 
    .class_size = 10, // Default class size
    .binary = 0, // Default binary output
    .debug_mode = 2, // Default debug mode
    .cbow = 1, // Default CBOW model
    .window = 5, // Default window size
    .min_count = 5, // Default minimum count for words  
    .num_threads = 20, // Default number of threads
    .min_reduce = 1, // Default minimum reduce count
    .hs = 0, // Default hierarchical softmax
    .negative = 5, // Default negative sampling
    .iter = 5, // Default number of iterations
    .learning_rate = 0.05, // Default learning rate
    .learning_rate_decay = 0.05, // Default learning rate decay
    .sample = 1e-3, // Default subsampling rate
    .train_file = "", // Default training file
    .output_file = "", // Default output file
    .save_vocab_file = "", // Default vocabulary save file
    .read_vocab_file = "", // Default vocabulary read file
    .vocab_hash_size = 1000000, // Default vocabulary hash size

    .vocab_size = 0, // Default vocabulary size
    .vocab_max_size = 1000000, // Default maximum vocabulary size
    .vocab = NULL, // Vocabulary pointer
    .vocab_hash = NULL, // Vocabulary hash table
    .file_size = 0, // Default file size
    .train_words = 0, // Default number of words in training file
    .word_count_actual = 0, // Actual word count
    .start = 0, // Start time for training

    .update_word_count = 10000, // Update word count every 10,000 words
    .vocab = NULL, // Vocabulary pointer
    .layer1 = NULL, // Layer 1 weights
    .layer2 = NULL, // Layer 2 weights
    .output = NULL // Output weights
  };
  printf("[INFO] FastText training started.\n");

  if ((i = get_arg_pos((char *)"-size", argc, argv)) > 0) gs.layer1_size = atoi(argv[i + 1]);
  if ((i = get_arg_pos((char *)"-train", argc, argv)) > 0) strcpy(gs.train_file, argv[i + 1]);
  if ((i = get_arg_pos((char *)"-output", argc, argv)) > 0) strcpy(gs.output_file, argv[i + 1]);
  if ((i = get_arg_pos((char *)"-save-vocab", argc, argv)) > 0) strcpy(gs.save_vocab_file, argv[i + 1]);
  if ((i = get_arg_pos((char *)"-read-vocab", argc, argv)) > 0) strcpy(gs.read_vocab_file, argv[i + 1]);
  if ((i = get_arg_pos((char *)"-debug", argc, argv)) > 0) gs.debug_mode = atoi(argv[i + 1]);
  if ((i = get_arg_pos((char *)"-binary", argc, argv)) > 0) gs.binary = atoi(argv[i + 1]);
  if ((i = get_arg_pos((char *)"-cbow", argc, argv)) > 0) gs.cbow = atoi(argv[i + 1]);
  if ((i = get_arg_pos((char *)"-lr", argc, argv)) > 0) gs.learning_rate = atof(argv[i + 1]);
  if ((i = get_arg_pos((char *)"-window", argc, argv)) > 0) gs.window = atoi(argv[i + 1]);
  if ((i = get_arg_pos((char *)"-sample", argc, argv)) > 0) gs.sample = atof(argv[i + 1]);
  if ((i = get_arg_pos((char *)"-hs", argc, argv)) > 0) gs.hs = atoi(argv[i + 1]);
  if ((i = get_arg_pos((char *)"-negative", argc, argv)) > 0) gs.negative = atoi(argv[i + 1]);
  if ((i = get_arg_pos((char *)"-thread", argc, argv)) > 0) gs.num_threads = atoi(argv[i + 1]);
  if ((i = get_arg_pos((char *)"-iter", argc, argv)) > 0) gs.iter = atoi(argv[i + 1]);
  if ((i = get_arg_pos((char *)"-min-count", argc, argv)) > 0) gs.min_count = atoi(argv[i + 1]);
  if ((i = get_arg_pos((char *)"-classes", argc, argv)) > 0) gs.classes = atoi(argv[i + 1]);

  printf("[INFO] Argument parsing completed.\n");
  // bag of tricks for efficient text classification additional setting
  // lr
  // wordNgrams
  // bucket
  gs.vocab = (vocab_word *)calloc(gs.vocab_max_size, sizeof(vocab_word));
  for (int j = 0; j < gs.vocab_max_size; j++) {
    gs.vocab[j].word = (char *)calloc(MAX_STRING, sizeof(char));
    gs.vocab[j].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    gs.vocab[j].point = (int *)calloc(MAX_CODE_LENGTH + 1, sizeof(int));
  }
  gs.vocab_hash = (int *)calloc(gs.vocab_hash_size, sizeof(int));
  for (int j = 0; j < gs.vocab_hash_size; j++) {
    gs.vocab_hash[j] = -1; // Initialize the vocabulary hash table
  }

  train_model(&gs);

  printf("[INFO] FastText training started.\n");

  return 0;
}
