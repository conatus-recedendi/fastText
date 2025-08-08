
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <math.h>
#include <ctype.h>
#include "config.h"
#include "vocab.h"
#include "utils.h"

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


  printf("[INFO] Allocated memory for layer1 with size: %lld\n", gs->vocab_size * gs->layer1_size * sizeof(float));
  posix_memalign((void **)&(gs->layer2), 64, gs->layer1_size * gs->label_size * sizeof(float));
  if (gs->layer2 == NULL) {
    fprintf(stderr, "Memory allocation failed for layer2\n");
    exit(1);
  }

  for (long long i = 0; i < gs->layer1_size * gs->label_size; i++) {
    gs->layer2[i] = (float)rand() / RAND_MAX * 2 - 1; // Initialize with random values between -1 and 1
  }
  // printf("[INFO] Allocated memory for layer2 with size: %lld\n", gs->layer1_size * gs->label_size * sizeof(float));
  posix_memalign((void **)&(gs->output), 64, gs->label_size * sizeof(float));
  if (gs->output == NULL) {
    fprintf(stderr, "Memory allocation failed for output\n");
    exit(1);
  }

  for (long long i = 0; i < gs->label_size; i++) {
    gs->output[i] = 0.0f; // Initialize output weights to zero
  }
  // printf("[INFO] Network initialized with layer1 size: %lld, class size: %lld\n", gs->layer1_size, gs->label_size);

  printf("[INFO] Network initialized with layer1 size: %lld, class size: %lld\n", gs->layer1_size, gs->label_size);
  // TODO: if classifation, gs->labels should be passed
  create_binary_tree(gs->vocab, gs->vocab_size);
  // create_binary_tree(gs->labels, gs->label_size);
  return ;
}

void save_vector(char *output_file, global_setting *gs) {
  // Implement the logic to save the vector representations
  FILE *fo = fopen(output_file, "w");
  fprintf(fo, "%lld %lld\n", gs->vocab_size, gs->layer1_size);
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

void load_model(char *load_model_file, global_setting *gs) {
  // Implement the logic to load the model
  FILE *fi = fopen(load_model_file, "rb");
  char read_vec_file[MAX_STRING];
  strncpy(read_vec_file, gs->read_vec_file, MAX_STRING);

  if (fi == NULL) {
    printf("Error opening file %s for reading\n", load_model_file);
    exit(1);  
  }
  printf("[INFO] Loading model from file: %s\n", load_model_file);
  fread(gs, sizeof(global_setting), 1, fi);
  printf("gs->ngrma_size: %lld\n", gs->ngram);

  printf("[INFO] Vocabhash table size: %lld\n", gs->vocab_hash_size);
  gs->vocab_hash = (int *)calloc(gs->vocab_hash_size, sizeof(int));

  if (gs->vocab_hash == NULL) {
    fprintf(stderr, "Memory allocation failed for vocab_hash\n");
    exit(1);
  }


  gs->vocab = (vocab_word *)calloc(gs->vocab_max_size, sizeof(vocab_word));
  if (gs->vocab == NULL) {
    fprintf(stderr, "Memory allocation failed for vocab\n");
    exit(1);
  }

  gs->label_hash = (int *)calloc(gs->label_hash_size, sizeof(int));
  if (gs->label_hash == NULL) {
    fprintf(stderr, "Memory allocation failed for label_hash\n");
    exit(1);
  }
  gs->labels = (vocab_word *)calloc(gs->label_max_size, sizeof(vocab_word));
  if (gs->labels == NULL) {
    fprintf(stderr, "Memory allocation failed for labels\n");
    exit(1);
  }
  printf("[INFO] Vocab and labels allocated with size: %lld, %lld\n", gs->vocab_max_size, gs->label_max_size);


  for (int j = 0; j < gs->label_max_size; j++) {
    // gs->vocab[j].word = (char *)calloc(MAX_STRING, sizeof(char));
    // gs->vocab[j].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    // gs->vocab[j].point = (int *)calloc(MAX_CODE_LENGTH + 1, sizeof(int));
    // printf("[DEBUG] labels[%d]: %s, cn: %lld\n", j, gs->labels[j].word, gs->labels[j].cn);
  }
// 
  // posix_memalign((void **)&(gs->layer1), 64, (long long)gs->vocab_size * gs->layer1_size * sizeof(float));
  fread(gs->vocab_hash, sizeof(int), gs->vocab_hash_size, fi);
  fread(gs->label_hash, sizeof(int), gs->label_hash_size, fi);
  fread(gs->vocab, sizeof(vocab_word), gs->vocab_max_size, fi);
  fread(gs->labels, sizeof(vocab_word), gs->label_max_size, fi);
  for (int i=0;i< 10; i++)  {
    // printf vocab
    printf("[DEBUG] labels[%d]: , word: %s, cn: %lld, codelen: %d\n", i, gs->labels[i].word, gs->labels[i].cn ,gs->labels[i].codelen);
  }

  // for (int i=0;i<100; i++)  {
  //   // printf vocab
  //   printf("[DEBUG] vocab[%d]: , cn: %lld\n", i, gs->vocab[i].cn);
  // }
  // printf("[INFO] Vocabulary and labels loaded from %lld\n", gs->label_hash[0]);
  // printf("[INFO] Vocabulary and labels loaded from %lld\n", gs->label_hash[427187]);

  printf("[INFO] Vocabulary loaded from %s\n", load_model_file);
  
  
  gs->layer1 = (float *)calloc(gs->vocab_size * gs->layer1_size, sizeof(float));
  if (gs->layer1 == NULL) {
    fprintf(stderr, "Memory allocation failed for layer1\n");
    exit(1);
  }
  posix_memalign((void **)&(gs->layer2), 64, gs->layer1_size * gs->label_size * sizeof(float));
  if (gs->layer2 == NULL) {
    fprintf(stderr, "Memory allocation failed for layer2\n");
    exit(1);
  }

  posix_memalign((void **)&(gs->output), 64, gs->label_size * sizeof(float));
  if (gs->output == NULL) {
    fprintf(stderr, "Memory allocation failed for output\n");
    exit(1);
  }

  // fread(gs->labels, sizeof(vocab_word), gs->label_size, fi);
  fread(gs->layer1, sizeof(float), gs->vocab_size * gs->layer1_size, fi);
  // printf("[INFO] Layer1 weights loaded from %s, read %zu elements\n", load_model_file, read);

  // for (long long i = 0; i < gs->vocab_size * gs->layer1_size; i++) {
  //   printf("[INFO] Layer1[%lld]: %f\n", i, gs->layer1[i]);
  // }

  fread(gs->layer2, sizeof(float), gs->layer1_size * gs->label_size, fi);
  // for (long long i = 0; i < gs->layer1_size * gs->label_size; i++) {
  //   printf("[INFO] Layer2[%lld]: %f\n", i, gs->layer2[i]);
  // }
  fread(gs->output, sizeof(float), gs->label_size, fi);
  gs->start_offsets= malloc(sizeof(long long) * gs->num_threads);
  gs->end_offsets = malloc(sizeof(long long) * gs->num_threads);
  gs->start_line_by_thread = malloc(sizeof(long long) * gs->num_threads);
  gs->total_line_by_thread = malloc(sizeof(long long) * gs->num_threads);

  fread(gs->start_offsets, sizeof(long long), gs->num_threads + 1, fi);

  printf("[INFO] Layer weights loaded from %s\n", load_model_file);
  fread(gs->end_offsets, sizeof(long long), gs->num_threads, fi);
  fread(gs->start_line_by_thread, sizeof(long long), gs->num_threads + 1, fi);
  fread(gs->total_line_by_thread, sizeof(long long), gs->num_threads + 1, fi);
  fclose(fi);


  if (read_vec_file[0]) {
    // Load the vector representations from the read_vec_file
    FILE *ff = fopen(read_vec_file, "r");
    // load_vector(gs);
    printf("[INFO] Loading vector representations from %s\n", read_vec_file);
    char line[MAX_WORDS_PER_SENTENCE];
    int line_num = 0;

    while (fgets(line, MAX_WORDS_PER_SENTENCE, ff)) {
      // printf("[DEBUG] Reading line %d: %s", line_num, line);
        line_num++;

        if (line_num == 1) continue;  // 첫 줄은 스킵

        // 줄 파싱 시작
        char *saveptr;
        char *token = strtok_r(line, "\t \n", &saveptr);  // 첫 단어
        if (!token) continue;

        int index = search_vocab(token, gs);
        if (index < 0 || index >= gs->vocab_size) {
          // fprintf(stderr, "%d", get_word_hash(token, gs));
            fprintf(stderr, "[WARN] Word '%s' not found in vocab (line %d), %d\n", token, line_num, index);
            continue;
        }

        // float 파싱
        for (int i = 0; i < gs->layer1_size; i++) {
            token = strtok_r(NULL, " \t\n", &saveptr);
            if (!token) {
                fprintf(stderr, "[ERROR] Not enough floats for word '%s' at line %d\n", token, line_num);
                break;
            }
            gs->layer1[index * gs->layer1_size + i] = strtof(token, NULL);
            // printf("[DEBUG] Layer1[%d][%d]: %f\n", index, i, gs->layer1[index * gs->layer1_size + i]);
        }
    }

    fclose(ff);
  } else {
    printf("[INFO] No read_vec_file specified, skipping vector loading.\n");
  }
  printf("Done loading model from %s\n", load_model_file);
}


void test_thread(global_setting *gs) {
  // Implement the test logic here
  long long thread_id = 0; // Assuming single thread for testing
  struct timespec start;
  clock_gettime(CLOCK_MONOTONIC, &start);
  // Placeholder for thread-specific test logic

  long long file_size = gs->file_size;
  int num_threads = gs->num_threads;

  long long word_count = 0;
  long long last_word_count = 0;
  long long iter = gs->iter;

  long long sentence_length = 0;
  long long sentence_position = 0;
  long long sentence_start = 0;
  long long sentence_end = 0;
  

  printf("[INFO] test_thread started testing...\n");
  

  FILE *fi = fopen(gs->test_file, "rb");
  printf("[DEBUG] Thread %lld opened file %s\n", thread_id, gs->test_file);

    // Reset sentence length and position for each iteration
  sentence_length = 0;
  sentence_position = 0;
  // Read the file line by line
  fseek(fi, 0, SEEK_SET);
  printf("[DEBUG] Thread %lld set file position to start\n", thread_id);
  

  // char word[MAX_SENTENCE_LENGTH];
  char sen[MAX_SENTENCE_LENGTH];
      char word[MAX_STRING];
    char prev_word[MAX_STRING]; // only support for ngram=2
    char concat_word[MAX_STRING];
  // long long labels[MAX_LABELS]; // [0, 3, -1, -1, -1 ...]
  long long *labels = (long long *)malloc(gs->label_size * sizeof(long long));
  // long long words[MAX_WORDS_PER_SENTENCE]; // [0, 1, 2, 3, 4 ...]
  long long *words = (long long *)malloc(MAX_WORDS_PER_SENTENCE * sizeof(long long));
  long long ngram_words[MAX_WORDS_PER_SENTENCE];


  long long temp = 0;
  // while (1) {
  //   fgets(word, MAX_STRING, fi);
    
  printf("[DEBUG] Starting to read sentences from file...\n");
  // }
  long long line = 0;
  long long max_line = count_lines(fi);
  printf("[DEBUG] Total lines in test file: %lld\n", max_line);

  long long correct_cnt = 0;
  long long total_cnt = 0;
  float precision = 0.0f;
  float recall = 0.0f;
  float f1_score = 0.0f;
  long long tp_cnt = 0;
  long long tn_cnt = 0;
  long long fp_cnt = 0;
  long long fn_cnt = 0;
  long long wrong_cnt = 0;


  float *neu1 = (float *)malloc(gs->layer1_size * sizeof(float));
  float *neu2 = (float *)malloc(gs->label_size * sizeof(float));
  long long avg_ngram =0;
  long long avg_failure_ngram = 0;
  long long avg_word =0;
  printf("[DEBUG] Starting to process sentences...\n");
  

  while (fgets(sen, MAX_SENTENCE_LENGTH, fi)) {
    line++;
    if (line % 1000 == 0) {
      // printf("%c[INFO] avg_ngram: %lld, avg_failrue_gram: %lld, avg_word: %lld, total: %lld/%lld\n", 13,avg_ngram / 1000, avg_failure_ngram / 1000, avg_word / 1000, line, (gs->train_words / gs->iter));
      // fflush(stdout);
      avg_ngram = 0;
      avg_failure_ngram = 0;
      avg_word = 0;
    }


    // printf("")
    printf("[DEBUG] Processing line %lld: %s", line, sen);
    // gs->total_learned_lines++;
    // word를 label, words로 분리.
    // 줄 끝 개행 문자 제거
    sen[strcspn(sen, "\n")] = 0;

    // 단어 분리
    char *token = strtok(sen, " ");
    long long sentence_length = 0;
    long long ngram_sentences_length = 0;
    long long label_length = 0;
    memset(labels, -1, gs->label_size * sizeof(long long)); // Initialize labels to 0
    memset(words, -1, MAX_WORDS_PER_SENTENCE * sizeof(long long)); // Initialize words to -1 (unknown word)
    memset(ngram_words, -1, sizeof(ngram_words)); // Initialize ngram_words to -1 (unknown word)

    while (token != NULL) {
      if (strlen(token) > MAX_STRING) {
        token = strtok(NULL, " ");
        continue; // Skip tokens that are too long
      }
      // printf("%s \n", token);
      if (strncmp(token, "__label__", 9) == 0) {

          memset(prev_word, 0, sizeof(prev_word)); // Reset previous word for ngram/ Reset previous word hash for ngram;
        // 라벨인 경우 __label_1__
          long long label_index = search_label(token, gs);



        // printf("[INFO] Found label: %s, index: %lld\n", token, label_index);
          if (label_index != -1) {
              labels[label_length++] = label_index;  // Set the label index to 1
          } else {
            // labels[label_length++] = -1; // unknown label
          
          }
        // printf("[DEBUG] Label Length: %lld, Labels: ", label_length);
      } else {

          // 일반 단어인 경우
          long long word_index = search_vocab(token, gs);
          // long long word_hash = get_word_hash(token, gs);
          // printf("[DEBUG] Token: %s, Word Hash: %lld, Word Index: %lld\n", token, word_hash, word_index);
          if (word_index != -1 && sentence_length < MAX_WORDS_PER_SENTENCE) {
            words[sentence_length++] = word_index;
            avg_word++;
            if (gs->ngram > 1) {

              if (prev_word[0] == 0) {
                strncpy(prev_word, token, sizeof(prev_word) - 1);
                // prev_word[sizeof(prev_word) - 1] = '\0'; // Ensure null termination
              } else {
                memset(concat_word, 0, sizeof(concat_word)); // Reset concat_word
                snprintf(concat_word, MAX_STRING, "%s-%s", prev_word, token);

                long long index = search_vocab(concat_word, gs);
                if (index == -1) {
                  // skip
                  // printf("[DEBUG] current line: %lld, Ngram word not found: %s\n", line, concat_word);
                  avg_failure_ngram++;
                  // getchar();
                } else {
                  avg_ngram++;
                  words[sentence_length++] = index; // ngram word
                }
              }
            }
          } 
          memset(prev_word, 0, sizeof(prev_word)); // Reset previous word for ngram
          strncpy(prev_word, token, MAX_STRING - 1); // Update previous word
          prev_word[MAX_STRING - 1] = '\0'; // Ensure null termination
      }
      token = strtok(NULL, " ");
    }
    printf("[DEBUG] Sentence Length: %lld, Label Length: %lld, Words: ", sentence_length, label_length);
    memcpy(prev_word, "", 1); // Reset previous word for next sentence
    // exit(1);
    gs->train_words += sentence_length; // Increment train words by the number of words in the sentence
    // gs->train_words += word_count(word);
    gs->learning_rate_decay = gs->learning_rate * (1 - (double)gs->total_learned_lines / (double)(gs->total_lines * gs->iter));
    
    if (gs->debug_mode > 1) {
      temp = 0;
      // clock_t now = clock();
      struct timespec ts;
      clock_gettime(CLOCK_MONOTONIC, &ts);

      // printf("%cProgress: %.2f%%  Line/sec: %.4fk, Time: %.2fs",
      //       13,
      //       line / (double)(max_line) * 100,
      //       line / ((double)(ts.tv_sec - start.tv_sec + 1) * (double)1000), 
      //     ((double)(ts.tv_sec - start.tv_sec)) + (double)(ts.tv_nsec - start.tv_nsec) / 1000000000.0);
      fflush(stdout);
    }
    
    // learning by line
    // logging by N lines
    // if (isNewLine(word)) {
    //   // Forward
    //   if (sentence_length > 0) {
    //     // Process the sentence
    //         total / (double)(gs->iter * gs->total_offset) * 100,
    //         gs->word_count_actual / ((double)(now - gs->start + 1) / (double)CLOCKS_PER_SEC * 1000) / gs->num_threads, gs->word_count_actual, i);
    //   fflush(stdout);
    //   }
    // }
    // learning by line
    // logging by N lines
    long long golden_label = -1;
    // printf("[DEBUG] Sentence Length: %lld, Label Length: %lld, Words: ", sentence_length, label_length);
    if (label_length==0 || sentence_length == 0) wrong_cnt++;

    if (sentence_length > 0 && label_length > 0) {
      // for (long long j = 0; j < ngram_sentences_length; j++) {
      //   // printf("ngram_words[%lld]: %lld ", j, ngram_words[j]);
      //   if (j + sentence_length > MAX_SENTENCE_LENGTH) {
      //     break ;
      //   }
      //   words[sentence_length++] = ngram_words[j];
      // }
    // words 안에 있는 단어들에 대한 임베딩을 가져와서 평균을 구함
      memset(neu1, 0, gs->layer1_size * sizeof(float));
      memset(neu2, 0, gs->label_size * sizeof(float));
      // memset(neu1err, 0, gs->layer1_size * sizeof(float));
      // memset(neu2err, 0, gs->label_size * sizeof(float));

      
      for (long long j = 0; j < sentence_length; j++) {
        if (words[j] != -1) {
          for (long long k = 0; k < gs->layer1_size; k++) {
            neu1[k] += gs->layer1[words[j] * gs->layer1_size + k];
          }

        }
      }
      for (long long j = 0; j < gs->layer1_size; j++) {
        neu1[j] /= sentence_length; // 평균을 구함
      }

      float *neu2_sorted = (float *)malloc(gs->label_size * sizeof(float));
      long long *index_sorted = (long long *)malloc(gs->label_size * sizeof(long long));

      printf("[DEBUG] Sentence Length: %lld, Label Length: %lld, Words: ", sentence_length, label_length);
      if (gs->hs)  {
         // MEMO: hierarchical softmax는 prec 값만 구함.
         // precision 외의 값을 구하기 위해서는 전체 label의 등장확률을 구해야하고 - softmax 보다 느림.
         // 즉, heirecal softmax로 얻은 max 값이 golden labels 내에 존재하기만 하면 됨

        long long *gold = (long long *)malloc(gs->label_size * sizeof(long long));
        long long local_tp_cnt = 0;
        long long local_fp_cnt = 0;
  
        long long gold_length = 0;
        for (long long j = 0; j < gs->label_size; j++) {
          if (labels[j] >= 0) {
            gold[gold_length++] = labels[j];
          }
        }

        // if (gold_length != 1) printf("[INFO] Gold length: %lld, Predicted length: %lld\n", gold_length, gs->top_k);
        int out_flag = 0;
        for (long long j = 0; j < gold_length; j++) {
          float prob = 1.0f;
          int flag = 0;
          // printf("labels[%lld].colden: %lld \n", gold[j], gs->labels[gold[j]].codelen);
          for (int k = 0; k < gs->labels[gold[j]].codelen; k++) {
            long long point = gs->labels[gold[j]].point[k];
            long long code = gs->labels[gold[j]].code[k];
            float dot = 0.0f;

            for (int l =0; l < gs->layer1_size; l++) {
              dot += neu1[l] * gs->layer2[l * gs->label_size + point];
            }

            float sigmoid = 1.0f / (1.0f + expf(-dot));
            prob *= (code == 0 ? logf(sigmoid + 1e-10) : logf(1.0f - sigmoid + 1e-10));
            if (code == 0 && sigmoid < 0.42) {
              // local_fp_cnt++;
              flag++;
              break ;
              // printf("[WARN] Hierarchical softmax: prob: %f, gold: %lld\n", prob, gold[j]);
            } else if (code == 1 && sigmoid > 0.58) {
              // local_fp_cnt++;
              flag++;
              break ;
            } else {


            }
            // printf("[DEBUG] Hierarchical softmax: point: %lld, code: %lld, dot: %f, sigmoid: %f, prob: %f\n", point, code, dot, sigmoid, prob);
          }
          if (flag == 0) {
            // local_tp_cnt++;
            out_flag = 1;
            break ;
          }
        }

        // dfs(gs, 2 * gs->label_size - 2, 0.0); // Traverse the binary tree to update output
        if (out_flag) {
          local_tp_cnt++;
        } else {
          local_fp_cnt++;
        }
        tp_cnt += local_tp_cnt;
        fp_cnt += local_fp_cnt;
        total_cnt += gold_length; // Total number of true labels
      } else {

        for (long long j = 0; j < gs->label_size; j++) {
          // neu2: 1 x c
          neu2[j] = 0.0f;
          for (long long k = 0; k < gs->layer1_size; k++) {
            neu2[j] += neu1[k] * gs->layer2[k * gs->label_size + j];
    
          }
          
        }
    
      
        float max = neu2[0];
        for (long long j = 0; j < gs->label_size; j++) {
          if (neu2[j] > max) max = neu2[j];
        }

        for (long long j = 0; j < gs->label_size; j++)
            neu2[j] = expf(neu2[j] - max);
    
        float sum = 0.0f;
        for (long long j = 0; j < gs->label_size; j++)
            sum += neu2[j];
    
        for (long long j = 0; j < gs->label_size; j++)
            neu2[j] /= sum;
      
        // for (long long j = 0; j < gs->label_size; j++) {
        //   printf("%f %f", neu2[j], sum);
        // }
        // printf("\n");
        // neu2 copy and sort by decreasign
        // but, there are remianing information to index to original neu2
    
        for (long long j = 0; j < gs->label_size; j++) {
          neu2_sorted[j] = neu2[j];
          index_sorted[j] = j;
        }
        for (long long j = 0; j < gs->label_size - 1; j++) {
          for (long long k = j + 1; k < gs->label_size; k++) {
            if (neu2_sorted[j] < neu2_sorted[k]) {
              // swap neu2_sorted
              float temp_value = neu2_sorted[j];
              neu2_sorted[j] = neu2_sorted[k];
              neu2_sorted[k] = temp_value;  
              // swap index_sorted
              long long temp_index = index_sorted[j];
              index_sorted[j] = index_sorted[k];  
              index_sorted[k] = temp_index;
            }
          }
        }
  
        // printf("[INFO] Sorted neu2: %f %lld", neu2_sorted[0], index_sorted[0]);
        // TODO:
        // for (long long j = 0; j < gs->label_size; j++) {
        //   printf("%f %lld ", neu2_sorted[j], index_sorted[j]);
        // }
        // printf("\n");
  
        long long *gold = (long long *)malloc(gs->label_size * sizeof(long long));
        long long *predicted = (long long *)malloc(gs->label_size * sizeof(long long));
  
        long long gold_length = 0;
        for (long long j = 0; j < gs->label_size; j++) {
          if (labels[j] >= 0 && gold_length < gs->top_k) {
            gold[gold_length++] = labels[j];
          }
        }
  
        // printf()
        // printf("[INFO] Gold length: %lld, Predicted length: %lld\n", gold_length, gs->top_k);
  
  
        long long predicted_length = 0;
        for (long long j = 0; j < gs->label_size; j++) {
          if (neu2_sorted[j] >= gs->answer_threshold) {
            predicted[predicted_length++] = index_sorted[j];
          }
          if (predicted_length >= gs->top_k) {
            break; // Stop if we have enough predictions
          }
        }
  
        long long local_tp_cnt = 0;
        for (long long j = 0; j < predicted_length; j++) {
          for (long long k = 0; k < gold_length; k++) {
            if (predicted[j] == gold[k]) {
              local_tp_cnt++;
              break;
            }
          }
        }
  
        long long local_fp_cnt = predicted_length - local_tp_cnt;
        long long local_fn_cnt = gold_length - local_tp_cnt;
        long long local_tn_cnt = gs->label_size - (local_tp_cnt + local_fp_cnt + local_fn_cnt);
        // printf("[INFO] TP: %lld, FP: %lld, Gold length: %lld, Predicted length: %lld\n", local_tp_cnt, local_fp_cnt, gold_length, predicted_length);
        // printf("[INFO] expected value: %lld,golden value: %lld\n", predicted[0], gold[0]);
  
        tp_cnt += local_tp_cnt;
        fp_cnt += local_fp_cnt;
        fn_cnt += local_fn_cnt;
        tn_cnt += local_tn_cnt;
        total_cnt += gold_length; // Total number of true labels
      }

      // neu1 dot layer2

      // sort neu2_sorted and index_sorted by neu2_sorted
    }

    printf("[DEBUG] Sentence Length: %lld, Label Length: %lld, Words: ", sentence_length, label_length);
    // get precision@K and recall@K
  
    if (tp_cnt + fp_cnt > 0) {
      precision = (float)tp_cnt / (tp_cnt + fp_cnt);
    } else {
      precision = 0.0f;
    }
    if (tp_cnt + fn_cnt > 0) {
      recall = (float)tp_cnt / (tp_cnt + fn_cnt);
    } else {
      recall = 0.0f;
    }
    // Reset sentence length and position for the next sentence
    sentence_length = 0;
    sentence_position = 0;
    sentence_start = 0;
    sentence_end = 0;   

    continue;
  }

  printf("\n[INFO] Precision@K: %.4f, Recall@K: %.4f, F1-Score: %.4f\n", precision, recall, 2 * precision * recall / (precision + recall));  
  printf("[INFO] Total TP: %lld, FP: %lld, FN: %lld, TN: %lld\n", tp_cnt, fp_cnt, fn_cnt, tn_cnt);
  printf("[INFO] Total: %lld, Total Gold: %lld\n", total_cnt);
  printf("[INFO] Total sentences processed: %lld\n", line);
  printf("[INFO] Total wrong predictions: %lld\n", wrong_cnt);
}





void test_model(global_setting *gs) {
  // Placeholder for training model logic
  printf("[INFO] Testing model with layer size: %lld\n", gs->layer1_size);
  // Implement the training logic here


  printf("[INFO] Loading model from file: %s\n", gs->load_model_file);


  printf("[INFO] Starting training threads...\n");
  printf("[INFO] Initializing network... %lld %lld\n", gs->layer1_size, gs->label_size);
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);
  test_thread(gs);
  clock_gettime(CLOCK_MONOTONIC, &end);

  printf("[INFO] All test threads finished.\n");
  // print precision@k, recall@k, f1-score
  double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
  printf("[INFO] Testing completed in %.3f seconds.\n", elapsed);


  // for (int i = 0; i < gs->num_threads; i++) {
  //   // Create threads for training
  //   thread_args *args = (thread_args *)malloc(sizeof(thread_args));
  //   args->id = i;
  //   args->gs = gs;
  //   pthread_create(&pt[i], NULL, train_thread, (thread_args *)args);

  // }
  // // printf("[INFO] Training threads started.\n");
  // for (int i = 0; i < gs->num_threads; i++) {
  //   // Wait for threads to finish
  //   pthread_join(pt[i], NULL);
  // }
  // printf("[INFO] All test threads finished.\n");
  // free(pt);


  // TODO: output 쓰기.
  // 1. embedding 논문
  // 2. classification 논문

  // save_model(output_file, gs);
  // save_vector(save_vocab_file , gs);
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
    .total_lines = 0, // Total lines in training file
    .start_offsets = NULL, // Start offsets for each thread
    .end_offsets = NULL, // End offsets for each thread
    .start_line_by_thread = NULL, // Actual offset for each thread
    .top_k = 1, // Default top K for classification
    .ngram = 1,
    .bucket_size = 0,
  };
  printf("[INFO] FastText test started.\n");

  if ((i = get_arg_pos((char *)"-load-model", argc, argv)) > 0) {
    strcpy(gs.load_model_file, argv[i + 1]);
    if (gs.load_model_file[0] == 0) {
      fprintf(stderr, "No model file specified for loading. Exiting.\n");
    }
    printf("[INFO] Model file specified for loading: %s\n", gs.load_model_file);
  } else {
    fprintf(stderr, "No model file specified for loading. Exiting.\n");
  }

  // /read_vec_file
  if ((i = get_arg_pos((char *)"-read-vec-file", argc, argv)) > 0) {
    // gs.read_vec_file = argv[i + 1];
    strcpy(gs.read_vec_file, argv[i + 1]);
    if (gs.read_vec_file[0] == 0) {
      fprintf(stderr, "No read vector file specified. Exiting.\n");
      return 1;
    }
    printf("[INFO] Read vector file specified: %s\n", gs.read_vec_file);
  } else {
    gs.read_vec_file[0] = 0; // No read vector file specified
    // gs.read_vec_file = NULL; // No read vector file specified
  }
  
  load_model(gs.load_model_file, &gs);
    // Save the model to file


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
  if ((i = get_arg_pos((char *)"-topk", argc, argv)) > 0) gs.top_k = atoi(argv[i + 1]);
  if ((i = get_arg_pos((char *)"-answer-threshold", argc, argv)) > 0) gs.answer_threshold = atof(argv[i + 1]);
  // test-file
  if ((i = get_arg_pos((char *)"-test-file", argc, argv)) > 0) {
    printf("[INFO] Test file specified: %s\n", argv[i + 1]);
    strcpy(gs.test_file, argv[i + 1]);
    if (gs.test_file[0] == 0) {
      fprintf(stderr, "No test file specified. Exiting.\n");
      return 1;
    }
  } else {
    fprintf(stderr, "No test file specified. Exiting.\n");
    return 1;
  }
  // printf("[INFO] Argument parsing completed.\n");
  // bag of tricks for efficient text classification additional setting
  // lr
  // wordNgrams
  // bucket
  // gs.vocab = (vocab_word *)calloc(gs.vocab_max_size, sizeof(vocab_word));
  // for (int j = 0; j < gs.vocab_max_size; j++) {
  //   gs.vocab[j].word = (char *)calloc(MAX_STRING, sizeof(char));
  //   gs.vocab[j].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
  //   gs.vocab[j].point = (int *)calloc(MAX_CODE_LENGTH + 1, sizeof(int));
  // }
  // gs.vocab_hash = (int *)calloc(gs.vocab_hash_size, sizeof(int));
  // for (int j = 0; j < gs.vocab_hash_size; j++) {
  //   gs.vocab_hash[j] = -1; // Initialize the vocabulary hash table
  // 

  printf("all information of global setting\n");
  printf("layer1_size: %lld\n", gs.layer1_size);
  printf("label_size: %lld\n", gs.label_size);
  printf("binary: %d\n", gs.binary);
  printf("debug_mode: %d\n", gs.debug_mode);
  printf("cbow: %d\n", gs.cbow);
  printf("window: %d\n", gs.window);
  printf("min_count: %d\n", gs.min_count);
  printf("num_threads: %d\n", gs.num_threads);
  printf("min_reduce: %d\n", gs.min_reduce);
  printf("hs: %d\n", gs.hs);
  printf("negative: %d\n", gs.negative);


  printf("iter: %d\n", gs.iter);
  printf("learning_rate: %f\n", gs.learning_rate);
  printf("learning_rate_decay: %f\n", gs.learning_rate_decay);
  printf("sample: %f\n", gs.sample);
  printf("train_file: %s\n", gs.train_file);

  printf("output_file: %s\n", gs.output_file);
  printf("save_vocab_file: %s\n", gs.save_vocab_file);
  printf("read_vocab_file: %s\n", gs.read_vocab_file);
  printf("vocab_hash_size: %lld\n", gs.vocab_hash_size);
  printf("vocab_size: %lld\n", gs.vocab_size);
  printf("vocab_max_size: %lld\n", gs.vocab_max_size);
  printf("test_file: %s\n", gs.test_file);
  printf("ngram: %d\n", gs.ngram);

  test_model(&gs);

  printf("[INFO] FastText test completed.\n");

  return 0;
}
  