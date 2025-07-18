
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

int isNewLine(const char *word) {
  return strcmp(word, "\n") == 0;
}
int isClass(const char *word) {
  return word[0] == '__' && word[strlen(word) - 1] == '__';
}

int isWord(const char *word) {
  return !isNewLine(word) && !isClass(word);
}
void softmaxf(const float* input, float* output, int size) {
    float max = input[0];
    for (int i = 1; i < size; ++i) {
        if (input[i] > max) {
            max = input[i];
        }
    }

    // expf 계산 + 합 구하기 (overflow 방지 위해 max 빼기)
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        output[i] = expf(input[i] - max);
        sum += output[i];
    }

    // 정규화
    for (int i = 0; i < size; ++i) {
        output[i] /= sum;
    }
}

long count_lines(FILE *fp) {
    long lines = 0;
    int c;
    while ((c = fgetc(fp)) != EOF) {
        if (c == '\n') lines++;
    }
    rewind(fp);
    return lines;
}

// 시작 및 끝 오프셋 계산
void compute_thread_offsets(FILE *fp, int num_threads, long total_lines, long *start_offsets, long *end_offsets, long *total_offset) {
    char line[MAX_LINE_LEN];
    long current_line = 0;
    int current_thread = 0;


    long *start_lines = malloc(sizeof(long) * (num_threads + 1));
    if (!start_lines) {
        perror("malloc");
        exit(1);
    }

    for (int i = 0; i <= num_threads; i++) {
        start_lines[i] = total_lines * i / num_threads;
    }

    // 첫 스레드 시작은 항상 0
    rewind(fp);
    start_offsets[0] = 0;

    while (fgets(line, sizeof(line), fp)) {
        current_line++;

        // 다음 스레드의 시작 라인에 도달하면 offset 저장
        if (current_thread + 1 <= num_threads &&
            current_line == start_lines[current_thread + 1]) {
            end_offsets[current_thread] = ftell(fp);
            start_offsets[current_thread + 1] = ftell(fp);
            current_thread++;
        }
    }

    // 마지막 스레드의 끝 오프셋은 파일 끝
    fseek(fp, 0, SEEK_END);
    end_offsets[num_threads - 1] = ftell(fp);

    free(start_lines);
    printf("[INFO] Computed thread offsets:\n");
    for (int i = 0; i < num_threads; i++) {
        printf("Thread %d: Start: %ld, End: %ld\n", i, start_offsets[i], end_offsets[i]);
    }
    *total_offset = end_offsets[num_threads - 1] - start_offsets[0];
}


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
  // printf("[INFO] Initializing network... %lld %lld \n", gs->vocab_size, gs->layer1_size);
  posix_memalign((void **)&(gs->layer1), 64, (long long)gs->vocab_size * gs->layer1_size * sizeof(float));
  if (gs->layer1 == NULL) {
    fprintf(stderr, "Memory allocation failed for layer1\n");
    exit(1);
  }
  //xavier

  printf("[INFO] Allocated memory for layer1 with size: %lld\n", gs->vocab_size * gs->layer1_size * sizeof(float));

  for (long long i = 0; i < gs->vocab_size * gs->layer1_size; i++) {
    // Initialize layer1 with random values between -1 and 1
    // Using a uniform distribution for initialization
    // You can also use other initialization methods like Xavier or He initialization
    // Here we use a simple random initialization for demonstration purposes
    // gs->layer1[i] = (float)rand() / RAND_MAX * 2 - 1; // Initialize with random values between -1 and 1
    // Xavier initialization
    // https://en.wikipedia.org/wiki/Xavier_initialization
    gs->layer1[i] = (float)rand() / RAND_MAX * 2 - 1; // Initialize with random values between -1 and 1
  }


  

  // printf("[INFO] Allocated memory for layer1 with size: %lld\n", gs->vocab_size * gs->layer1_size * sizeof(float));
  posix_memalign((void **)&(gs->layer2), 64, gs->layer1_size * gs->class_size * sizeof(float));
  if (gs->layer2 == NULL) {
    fprintf(stderr, "Memory allocation failed for layer2\n");
    exit(1);
  }

  for (long long i = 0; i < gs->layer1_size * gs->class_size; i++) {
    gs->layer2[i] = (float)rand() / RAND_MAX * 2 - 1; // Initialize with random values between -1 and 1
  }
  // printf("[INFO] Allocated memory for layer2 with size: %lld\n", gs->layer1_size * gs->class_size * sizeof(float));
  posix_memalign((void **)&(gs->output), 64, gs->class_size * sizeof(float));
  if (gs->output == NULL) {
    fprintf(stderr, "Memory allocation failed for output\n");
    exit(1);
  }

  for (long long i = 0; i < gs->class_size; i++) {
    gs->output[i] = 0.0f; // Initialize output weights to zero
  }
  // printf("[INFO] Network initialized with layer1 size: %lld, class size: %lld\n", gs->layer1_size, gs->class_size);

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
  // Implement the training logic here

  long long thread_id = (long long)args->id;
  global_setting *gs = (global_setting *)args->gs;
  // Placeholder for thread-specific training logic

  long long file_size = gs->file_size;
  int num_threads = gs->num_threads;

  long long word_count = 0;
  long long last_word_count = 0;
  long long iter = gs->iter;

  long long sentence_length = 0;
  long long sentence_position = 0;
  long long sentence_start = 0;
  long long sentence_end = 0;
  long long sen[MAX_SENTENCE_LENGTH];

  // printf("[INFO] Thread %lld started training...\n", thread_id);
  

  FILE *fi = fopen(gs->train_file, "rb");
  fseek(fi, file_size / (long long)num_threads * (long long)thread_id , SEEK_END);

  // printf("[INFO] Thread %lld opened file %s\n", thread_id, gs->train_file);
  for (int iter = 0; iter < gs->iter; iter++) {
  

    // Reset sentence length and position for each iteration
    sentence_length = 0;
    sentence_position = 0;
    // Read the file line by line
    fseek(fi, gs->start_offsets[thread_id], SEEK_SET);
    

    char word[MAX_STRING];
    char sen[MAX_SENTENCE_LENGTH];
    long long labels[MAX_LABELS]; // [1, 0, 0, 1, 0 ...]
    long long words[MAX_WORDS_PER_SENTENCE]; // [0, 1, 2, 3, 4 ...]

    long long offset_length = gs->end_offsets[thread_id] - gs->start_offsets[thread_id] + 1;

    long long temp = 0;
    long long i = gs->start_offsets[thread_id];
    while (i <= gs->end_offsets[thread_id]) {
      long long word_length = read_word(word, fi);
      word_count++;
      i += word_length - 1; // Adjust for the length of the word read
      gs->word_count_actual++;
      gs->offset_actual += word_length;
      temp += word_length;

      gs->learning_rate_decay = gs->learning_rate * (1 - ((float)gs->word_count_actual / (gs->total_offset * gs->iter)));

      if (gs->debug_mode > 1 && temp % 100000 >= 0) {
        temp = 0;
        clock_t now = clock();
        printf("%clr: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  , %d",
              13, gs->learning_rate_decay,
              gs->offset_actual / (double)(gs->iter * gs->total_offset) * 100,
              gs->word_count_actual / ((double)(now - gs->start + 1) / (double)CLOCKS_PER_SEC * 1000) / gs->num_threads, gs->word_count_actual);
        fflush(stdout);
      }

      // if (isNewLine(word)) {
      //   // Forward 
      //   if (sentence_length > 0) {
      //     // Process the sentence
      //     float *layer1_avg = (float *)calloc(gs->layer1_size, sizeof(float));

      //     for (int j = 0; j < sentence_length; j++) {
      //       long long word_index = words[j];
      //       // Update the word vectors
      //       for (long long k = 0; k < gs->layer1_size; k++) {
      //         // gs->layer1[word_index * gs->layer1_size + k] += learning_rate * (1 - words[j]);
      //         layer1_avg[k] += gs->layer1[word_index * gs->layer1_size + k];
      //         // forward and backward pass logic
      //       }
      //     }
      //     for (long long j = 0; j < gs->layer1_size; j++) {
      //       layer1_avg[j] /= sentence_length;
      //     }

      //     // Dot product with layer1_avg and layer2
      //     for (long long j = 0; j < gs->class_size; j++) {
      //       gs->output[j] = 0;
      //       for (long long k = 0; k < gs->layer1_size; k++) {
      //         gs->output[j] += layer1_avg[k] * gs->layer2[k * gs->class_size + j];
      //       }
      //     }

      //     softmaxf(gs->output, gs->output, gs->class_size);  
      //     float loss = 0.0;
      //     for (long long j = 0; j < gs->class_size; j++)
      //     {
      //       if (labels[j] == 1) {
      //         loss -= logf(gs->output[j] + 1e-10); // Add small value to avoid log(0)
      //       }
      //     }
      //     loss /= sentence_length;
          
      //     // back propagation logic
      //     // Allocate gradients
      //     float *dL_dh = (float *)calloc(gs->layer1_size, sizeof(float)); // dL/dh
      //     float learning_rate = 0.01f;  // 예시용

      //     // (1) dL/dz = y_hat - y (output - labels)
      //     for (long long j = 0; j < gs->class_size; j++) {
      //         gs->output[j] -= labels[j];  // output[j] = y_hat_j - y_j
      //     }

      //     // (2) Update layer2 weights: W = W - lr * outer(h, dL/dz)
      //     for (long long j = 0; j < gs->class_size; j++) {
      //         for (long long k = 0; k < gs->layer1_size; k++) {
      //             float grad = layer1_avg[k] * gs->output[j];  // outer product
      //             gs->layer2[k * gs->class_size + j] -= gs->learning_rate_decay * grad;
      //             dL_dh[k] += gs->layer2[k * gs->class_size + j] * gs->output[j]; // accumulate dL/dh
      //         }
      //     }

      //     // (3) Backprop to word embeddings (A): average ⇒ distribute
      //     for (int j = 0; j < sentence_length; j++) {
      //         long long word_index = words[j];
      //         for (long long k = 0; k < gs->layer1_size; k++) {
      //             float grad = dL_dh[k] / sentence_length;  // distribute average
      //             gs->layer1[word_index * gs->layer1_size + k] -= gs->learning_rate_decay * grad;
      //         }
      //     }

      //     free(dL_dh);
      //     free(layer1_avg);

      //   }
      //   sentence_length = 0; // Reset for next sentence
      //   for (int j = 0; j < MAX_LABELS; j++) {
      //     labels[j] = 0; // Reset labels
      //   }
      //   for (int j = 0; j < MAX_WORDS_PER_SENTENCE; j++) {
      //     words[j] = -1; // Reset words
      //   }
      //   continue;
      // }

      // if (isClass(word)) {
      //   // it is class
      //   long long word_index = search_vocab(word, gs);
      //   if (word_index != -1) {
      //     labels[word_index] = 1;
      //   }
      // }
      // }

      // if (isClass(word)) {
      //   // it is class
      //   long long word_index = search_vocab(word, gs);
      //   if (word_index != -1) {
      //     labels[word_index] = 1;
      //   }
      // }

      // if (isWord(word)) {
      //   // it is word
      //   long long word_index = search_vocab(word, gs);
      //   if (word_index != -1) {
      //     words[sentence_length] = word_index;
      //     sentence_length++;
      //   } else {
      //     // Handle unknown word
      //     words[sentence_length] = -1; // or some other logic
      //   }
      // }
    }
  

  fclose(fi);

  // Implement the saving output here
  pthread_exit(NULL);
}



void train_model(global_setting *gs) {
  // Placeholder for training model logic
  printf("[INFO] Training model with layer size: %lld\n", gs->layer1_size);
  // Implement the training logic here
  char *read_vocab_file = gs->read_vocab_file;
  char *save_vocab_file = gs->save_vocab_file;
  char *output_file = gs->output_file;
  gs->learning_rate_decay = gs->learning_rate;

  // printf("[INFO] Initializing threads...\n");
  pthread_t *pt = (pthread_t *)malloc(gs->num_threads * sizeof(pthread_t));


  FILE *fp = fopen(gs->train_file, "r");
  if (!fp) {
      perror("fopen");
      return ;
  }

  gs->total_lines = count_lines(fp);
  gs->total_offset = 0;
  gs->start_offsets = malloc(sizeof(long long) * gs->num_threads);
  gs->end_offsets = malloc(sizeof(long long) * gs->num_threads);

  compute_thread_offsets(fp, gs->num_threads, gs->total_lines, gs->start_offsets, gs->end_offsets, &gs->total_offset);

  // printf("[INFO] read vocabulary...\n");

  if (read_vocab_file[0] != 0) {
    // Read vocabulary from file
    read_vocab(gs);
  } else {
    // Create vocabulary from training file
    create_vocab_from_train_file(gs);
  }

  // printf("[INFO] save vocabulary...\n");

  if (save_vocab_file[0] != 0) {
    // Save vocabulary to file
    // save_vocab(gs);
  }


  if (output_file[0] == 0) {
    printf("No output file specified. Exiting.\n");
    return;
  }

  // printf("[INFO] Initializing network...\n");

  initialize_network(gs);

  // TODO: make unigram table

  gs->start = clock();

  // printf("[INFO] Starting training threads...\n");

  for (int i = 0; i < gs->num_threads; i++) {
    // Create threads for training
    thread_args *args = (thread_args *)malloc(sizeof(thread_args));
    args->id = i;
    args->gs = gs;
    pthread_create(&pt[i], NULL, train_thread, (thread_args *)args);

  }
  // printf("[INFO] Training threads started.\n");
  for (int i = 0; i < gs->num_threads; i++) {
    // Wait for threads to finish
    pthread_join(pt[i], NULL);
  }
  free(pt);


  // TODO: output 쓰기.
  // 1. embedding 논문
  // 2. classification 논문

  save_model(output_file, gs);
  save_vector(save_vocab_file , gs);
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
    .output = NULL, // Output weights
    .offset_actual = 0, // Actual offset for training
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

  // printf("[INFO] Argument parsing completed.\n");
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
  gs.offset_actual = 0; // Initialize offset actual

  train_model(&gs);

  printf("[INFO] FastText training completed.\n");

  return 0;
}
