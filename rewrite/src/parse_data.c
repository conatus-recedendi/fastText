
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <math.h>
#include <ctype.h>
#include "parse_data_config.h"
#include "parse_data_vocab.h"
#include "parse_data_utils.h"

void process_file(const char *input_path, const char *output_path, global_setting *gs) {
    FILE *input_file = fopen(input_path, "r");
    if (!input_file) {
        perror("Error opening input file");
        return;
    }

    FILE *output_file = fopen(output_path, "w");
    if (!output_file) {
        perror("Error opening output file");
        fclose(input_file);
        return;
    }

    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    long long line_cnt = 0;

    while ((read = getline(&line, &len, input_file)) != -1) {
        line_cnt++;
        printf("%c progress: %.2f%", 13,  line_cnt / (double)gs->total_lines * 100);
        fflush(stdout);

        char *line_copy = strdup(line); // 원본 라인 보존
        if (!line_copy) {
            perror("Memory allocation failed");
            continue;
        }

        char *token;
        char *rest_of_line = line_copy;
        int has_valid_label = 0;
        int has_valid_description_word = 0;

        // 1. 레이블 파싱 및 유효성 검사
        while ((token = strtok_r(rest_of_line, " \n", &rest_of_line))) {
            if (strstr(token, "__label__") == token) {
                if (search_label(token, gs) != -1) {
                    has_valid_label = 1;
                }
            } else {
                // 레이블 파싱이 끝났으므로 description으로 넘어갑니다.
                break;
            }
        }
        
        // 2. description 파싱 및 유효성 검사
        char *description_section = rest_of_line;
        if (description_section) {
            char *desc_copy = strdup(description_section);
            if (!desc_copy) {
                perror("Memory allocation failed");
                free(line_copy);
                continue;
            }

            char *desc_token;
            char *desc_rest = desc_copy;
            while ((desc_token = strtok_r(desc_rest, " \n", &desc_rest))) {
                if (search_vocab(desc_token, gs) != -1) {
                    has_valid_description_word = 1;
                    break; 
                }
            }
            free(desc_copy);
        }

        // 3. 조건 만족 시, 원본 라인을 새 파일에 추가
        if (has_valid_label && has_valid_description_word) {
            fprintf(output_file, "%s", line);
        }
        
        free(line_copy);
    }

    free(line);
    fclose(input_file);
    fclose(output_file);
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

  printf("[INFO] Counting total lines in training file...\n");
  printf("[DEBUG] label_hash_size: %d, pointer: %p\n", gs->label_hash_size, gs->label_hash);
  gs->total_lines = count_lines(fp);
  printf("[INFO] Total lines in training file: %lld\n", gs->total_lines);
  printf("[DEBUG] label_hash_size: %d, pointer: %p\n", gs->label_hash_size, gs->label_hash);
  // lines을 thread 개수만큼 분리
  // 각 데이터의 start offset, end offset 저장. 각 thread에서 실행할 라인 수 계산
  // 스레드에서는 start offset으로 fseek하고, 각 thread에서 실행할 데이터만큼 학습
  
  // long long total_line;

  gs->start_offsets= malloc(sizeof(long long) * gs->num_threads);
  gs->end_offsets = malloc(sizeof(long long) * gs->num_threads);
  gs->start_line_by_thread = malloc(sizeof(long long) * gs->num_threads);
  gs->total_line_by_thread = malloc(sizeof(long long) * gs->num_threads);
  gs->label_size = 0;
  printf("[INFO] Computing thread offsets...\n");
  printf("[DEBUG] label_hash_size: %d, pointer: %p\n", gs->label_hash_size, gs->label_hash);

  // compute_thread_offsets(fp, gs);


  printf("[INFO] read vocabulary...\n");

  if (read_vocab_file[0] != 0) {
    // Read vocabulary from file
    read_vocab(gs);
  } else {
    // Create vocabulary from training file
    create_vocab_from_train_file(gs);
  }
  printf("[DEBUG] label_hash_size: %d, pointer: %p\n", gs->label_hash_size, gs->label_hash);

  printf("[INFO] save vocabulary...\n");


  process_file(gs->train_file, gs->parse_output, gs);
  
}

int main(int argc, char **argv) {
  int i;

  global_setting gs = {
    .layer1_size = 10, // Default layer size 
    .label_size = 10, // Default class size
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
    .sample = 1e-4, // Default subsampling rate
    .train_file = "", // Default training file
    .output_file = "", // Default output file
    .save_vocab_file = "", // Default vocabulary save file
    .read_vocab_file = "", // Default vocabulary read file
    .vocab_hash_size = 10000000, // Default vocabulary hash size
    .label_hash_size = 10000000, // Default label hash size

    .vocab_size = 0, // Default vocabulary size
    .vocab_max_size = 10000000, // Default maximum vocabulary size
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
    .total_lines = 0, // Total lines in training file
    .start_offsets = NULL, // Start offsets for each thread
    .end_offsets = NULL, // End offsets for each thread
    .start_line_by_thread = NULL, // Actual offset for each thread\
    .ngram = 1,
    .bucket_size = 0,
    .min_count_label = 1,
    .min_count_vocab = 1
  };
  printf("[INFO] FastText training started.\n");

  if ((i = get_arg_pos((char *)"-size", argc, argv)) > 0) gs.layer1_size = atoi(argv[i + 1]);
  if ((i = get_arg_pos((char *)"-parse-output", argc, argv)) > 0) strcpy(gs.parse_output, argv[i + 1]);
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
  if ((i = get_arg_pos((char *)"-bucket", argc, argv)) > 0) {
    gs.vocab_hash_size = atoi(argv[i + 1]);
    gs.vocab_max_size = gs.vocab_hash_size; // Set max size to double the hash size
    gs.label_hash_size = gs.vocab_hash_size; // Set label hash size to vocabulary hash size
  }
  if ((i = get_arg_pos((char *)"-hs", argc, argv)) > 0) gs.hs = atoi(argv[i + 1]);
  if ((i = get_arg_pos((char *)"-ngram", argc, argv)) > 0) gs.ngram = atoi(argv[i + 1]);
  if ((i = get_arg_pos((char *)"-min-count-label", argc, argv)) > 0) gs.min_count_label = atoi(argv[i + 1]);
  if ((i = get_arg_pos((char *)"-min-count-vocab", argc, argv)) > 0) gs.min_count_vocab = atoi(argv[i + 1]);

  // printf("[INFO] Argument parsing completed.\n");
  // bag of tricks for efficient text classification additional setting
  // lr
  // wordNgrams=
  // bucket
  gs.vocab = (vocab_word *)calloc(gs.vocab_max_size, sizeof(vocab_word));
  if (gs.vocab == NULL) {
    fprintf(stderr, "[ERROR] Memory allocation failed for vocabulary\n");
    exit(1);
  }
  gs.vocab_hash = (int *)calloc(gs.vocab_hash_size, sizeof(int));
  if (gs.vocab_hash == NULL) {
    fprintf(stderr, "[ERROR] Memory allocation failed for vocabulary hash table\n");
    exit(1);
  }
  for (int j = 0; j < gs.vocab_hash_size; j++) {
    gs.vocab_hash[j] = -1; // Initialize the vocabulary hash table
  }
  gs.labels = (vocab_word *)calloc(gs.label_hash_size, sizeof(vocab_word));
  if (gs.labels == NULL) {
    fprintf(stderr, "[ERROR] Memory allocation failed for labels\n");
    exit(1);
  }
  gs.label_hash = (int *)calloc(gs.label_hash_size, sizeof(int));
  if (gs.label_hash == NULL) {
    fprintf(stderr, "[ERROR] Memory allocation failed for label hash table\n");
    exit(1);
  }
  for (int j = 0; j < gs.label_hash_size; j++) {
    gs.label_hash[j] = -1; // Initialize the label hash table
  }

  printf("[DEBUG] label_hash_size: %d, pointer: %p\n", gs.label_hash_size, gs.label_hash);
  printf("[DEBUG] vocab_hash_size: %d, pointer: %p\n", gs.vocab_hash_size, gs.vocab_hash);
  printf("[DEBUG] vocab_max_size: %d, pointer: %p\n", gs.vocab_max_size, gs.vocab);
  printf("[DEBUG] label_max_size: %d, pointer: %p\n", gs.label_max_size, gs.labels);

  // printf("%lld\n", gs.vocab_hash[886005]);
  train_model(&gs);

  printf("[INFO] FastText training completed.\n");

  return 0;
}
  