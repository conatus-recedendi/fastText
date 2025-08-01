
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
  long long size = gs->vocab_size * gs->layer1_size;
  for (long long i = 0; i < size; i++) {
    // Initialize layer1 with random values between -1 and 1
    // Using a uniform distribution for initialization
    // You can also use other initialization methods like Xavier or He initialization
    // Here we use a simple random initialization for demonstration purposes
    // gs->layer1[i] = (float)rand() / RAND_MAX * 2 - 1; // Initialize with random values between -1 and 1
    // Xavier initialization
    // https://en.wikipedia.org/wiki/Xavier_initialization
    gs->layer1[i] = ((float)rand() / RAND_MAX * 2 - 1) / size; // Initialize with random values between -1 and 1
    // uniform
    // gs->layer1[i] = ((float)rand() / RAND_MAX * 2 - 1) / size;
  }


  printf("[INFO] Allocated memory for layer1 with size: %lld\n", gs->vocab_size * gs->layer1_size * sizeof(float));
  posix_memalign((void **)&(gs->layer2), 64, gs->layer1_size * gs->label_size * sizeof(float));
  if (gs->layer2 == NULL) {
    fprintf(stderr, "Memory allocation failed for layer2\n");
    exit(1);
  }

  size = gs->layer1_size * gs->label_size;
  for (long long i = 0; i < gs->layer1_size * gs->label_size; i++) {
    gs->layer2[i] = ((float)rand() / RAND_MAX * 2 - 1) / size; // Initialize with random values between -1 and 1
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
  // create_binary_tree(gs->vocab, gs->vocab_size);
  create_binary_tree(gs->labels, gs->label_size);
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

  // gs를 binary 형태로 전부 저장. 언제든지 불러올 수 있는 형태로.
  printf("[INFO] Saving model to %s\n", output_file);
  fwrite(gs, sizeof(global_setting), 1, fo);
  printf("[INFO] ngram %lld\n",  gs->ngram);
  printf("[INFO] Global settings saved to %s\n", output_file);
  // Save the layer1, layer2, and output weights
  fwrite(gs->vocab_hash, sizeof(int), gs->vocab_hash_size, fo);
  printf("[INFO] Vocabulary hash table saved to %s %lld, %lld\n", output_file, gs->vocab_size, sizeof(vocab_word));
  fwrite(gs->label_hash, sizeof(int), gs->label_hash_size, fo);
  fwrite(gs->vocab, sizeof(vocab_word), gs->vocab_max_size, fo);
  fwrite(gs->labels, sizeof(vocab_word), gs->label_max_size, fo);
  // fwrite(gs->labels, sizeof(vocab_word), gs->label_size, fo);
  // for (int i=0;i<100; i++)  {
  //   // printf vocab
  //   printf("[DEBUG] vocab[%d]: , cn: %lld\n", i, gs->vocab[i].cn);
  // }

  printf("[INFO] Vocabulary and labels saved to %s\n", output_file);
  fwrite(gs->layer1, sizeof(float), gs->vocab_size * gs->layer1_size, fo);
  // for (long long i = 0; i < gs->vocab_size * gs->layer1_size; i++) {
  //   printf("[INFO] Layer1[%lld]: %f\n", i, gs->layer1[i]);
  // }
  printf("[INFO] Layer1 weights saved to %s\n", output_file);
  fwrite(gs->layer2, sizeof(float), gs->layer1_size * gs->label_size, fo);
  // for (long long i = 0; i < gs->layer1_size * gs->label_size; i++) {
  //   printf("[INFO] Layer2[%lld]: %f\n", i, gs->layer2[i]);
  // }
  printf("[INFO] Layer2 weights saved to %s\n", output_file);
  fwrite(gs->output, sizeof(float), gs->label_size, fo);
  printf("[INFO] Layer weights saved to %s\n", output_file);
  fwrite(gs->start_offsets, sizeof(long long), gs->num_threads, fo);
  fwrite(gs->end_offsets, sizeof(long long), gs->num_threads, fo);
  printf("[INFO] Thread offsets saved to %s\n", output_file);
  fwrite(gs->start_line_by_thread, sizeof(long long), gs->num_threads , fo);
  fwrite(gs->total_line_by_thread, sizeof(long long), gs->num_threads, fo);
  printf("[INFO] Thread offsets saved to %s\n", output_file);
  // Save the vocabulary
  // Save the vocabulary hash table

  fclose(fo);
}

void *train_thread(thread_args *args) {
  // fflush(stdout);
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
    char prev_word[MAX_STRING]; // only support for ngram=2
    char sen[MAX_SENTENCE_LENGTH];
    char concat_word[MAX_STRING];
    long long offset = 0;
    // long long labels[MAX_LABELS]; // [0, 3, -1, -1, -1 ...]
    long long *labels = malloc(sizeof(long long) * gs->label_size);
    if (labels == NULL) {
        fprintf(stderr, "[ERROR] Memory allocation failed for labels\n");
        exit(1);
    }

    long long words[MAX_WORDS_PER_SENTENCE]; // [0, 1, 2, 3, 4 ...]
    long long ngram_words[MAX_WORDS_PER_SENTENCE];



    long long temp = 0;

    // while (1) {
    //   fgets(word, MAX_STRING, fi);
      
    // }
    long long line = 0;
    long long max_line = gs->total_line_by_thread[thread_id];
    

    float *neu1 = (float *)malloc(gs->layer1_size * sizeof(float));
    float *neu2 = (float *)malloc(gs->label_size * sizeof(float));
    float *neu1err = (float *)malloc(gs->layer1_size * sizeof(float));
    float *neu2err = (float *)malloc(gs->label_size * sizeof(float));

    long long avg_ngram = 0;
    long long avg_failure_ngram = 0;
    long long avg_word =0;

    if (neu1 == NULL || neu2 == NULL || neu1err == NULL || neu2err == NULL) {
        fprintf(stderr, "[ERROR] Memory allocation failed for neu1, neu2, neu1err, or neu2err\n");
        exit(1);
    }

    // printf("[INFO] Thread %lld started training with %lld lines\n", thread_id, max_line);

    // copy of layer1, layer 2
    // float *layer1 = (float *)malloc(gs->vocab_size * gs->layer1_size * sizeof(float));
    // float *layer2 = (float *)malloc(gs->layer1_size * gs->label_size * sizeof(float));
    // if (layer1 == NULL || layer2 == NULL) {
    //     fprintf(stderr, "[ERROR] Memory allocation failed for layer1 or layer2\n");
    //     exit(1);
    // } 
    // printf("start_offsets: %lld, end_offsets: %lld, total_lines: %lld\n", gs->start_offsets[thread_id], gs->end_offsets[thread_id], gs->total_line_by_thread[thread_id]);
  
    while ( fgets(sen, MAX_SENTENCE_LENGTH, fi) && line < max_line) {
      line++;
      temp++;
      gs->total_learned_lines++;
      if (line % 1000 == 0) {
        // printf("[INFO] avg_ngram: %lld, avg_failrue_gram: %lld, avg_word: %lld\n", avg_ngram / 1000, avg_failure_ngram / 1000, avg_word / 1000);
        avg_ngram = 0;
        avg_failure_ngram = 0;
        avg_word = 0;
      }

      // pritnf("[DEBUG] Thread %lld, line %lld, total learned lines: %lld\n", thread_id, line, gs->total_learned_lines);
      // if (line % 1000 == 0) {
      //   printf("[INFO] Thread %lld, line %lld, total learned lines: %lld\n", thread_id, line, gs->total_learned_lines);
      // }
      // word를 label, words로 분리.
      // 줄 끝 개행 문자 제거
      sen[strcspn(sen, "\n")] = 0;

      // 단어 분리
      char *token = strtok(sen, " ");

      long long sentence_length = 0;
      long long ngram_sentences_length = 0;
      long long label_length = 0;
      memset(labels, -1, sizeof(labels)); // Initialize labels to -1
      memset(words, -1, sizeof(words)); // Initialize words to -1 (unknown word
      memset(ngram_words, -1, sizeof(ngram_words)); // Initialize ngram_words to -1 (unknown word)

      // copy from gs->layer1 to laeyr1
      // memcpy(layer1, gs->layer1, gs->vocab_size * gs->layer1_size * sizeof(float));
      // memcpy(layer2, gs->layer2, gs->layer1_size * gs->label_size * sizeof(float));
      while (token != NULL) {
        if (strlen(token) > MAX_STRING) {
          token = strtok(NULL, " ");
          continue; // Skip tokens that are too long
        }
        
          if (strncmp(token, "__label__", 9) == 0) {
            memset(prev_word, 0, sizeof(prev_word)); // Reset previous word for ngram
            // 라벨인 경우 __label_1__
            long long label_index = search_label(token, gs);
            if (label_index != -1 && label_index < MAX_LABELS) {
                labels[label_length++] = label_index;  // Set the label index to 1
            } else {
              // labels[label_length++] = -1; // unknown label
            }
          } else {
            // printf("[DEBUG] Token: %s\n", token); 
              // 일반 단어인 경우
              
              long long word_hash = get_word_hash(token, gs);

              long long word_index = search_vocab(token, gs);

              if (word_index != -1 && sentence_length < MAX_WORDS_PER_SENTENCE) {
                words[sentence_length++] = word_index; // vocab[word_index] or layer1[word_index]
                avg_word++;
                if (gs->ngram > 1) {
                  if (prev_word[0] == 0) {
                    strncpy(prev_word, token, sizeof(prev_word) - 1);
                    
                  } else {
                    memset(concat_word, 0, sizeof(concat_word)); // Reset concat_word
                    // printf("%lld \n", sizeof(concat_word));
                    // strcpy_s(concat_word, MAX_STRING, prev_word);
                    // strcat_s(concat_word, MAX_STRING, "-");
                    // memcpy(concat_word, prev_word, MAX_STRING);

                    strncpy(concat_word, prev_word, strlen(prev_word));
                    // printf("1. concat_word: %s, prev_word: %s, token: %s\n", concat_word, prev_word, token);
                    concat_word[strlen(prev_word)] = 0; // Add hyphen

                    if(strlen(concat_word) < MAX_STRING) {
                      // printf("2.. concat_word: %s, prev_word: %s, token: %s\n", concat_word, prev_word, token);
                      memcpy(concat_word + strlen(prev_word), "-", 1); // 0 -> '-'
                      // printf("3. concat_word: %s, prev_word: %s, token: %s\n", concat_word, prev_word, token);
                      concat_word[strlen(prev_word) + 1] = '\0'; // Ensure null termination
                      if (strlen(concat_word) + strlen(token) < MAX_STRING) {
                        // strcat_s(concat_word, MAX_STRING, token);
                        memcpy(concat_word + strlen(prev_word) + 1, token, strlen(token) + 1);
                      }
                      // strcat_s(concat_word, MAX_STRING, token);
                      // memcpy(concat_word + strlen(prev_word) + 1, token, strlen(token) + 1);
                      // skip
                    }
                    concat_word[MAX_STRING - 1] = '\0'; // Ensure null termination

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
              } else if(word_index != -1) {
                  // words[sentence_length] = -1; // unknown word
              }
              memset(prev_word, 0, sizeof(prev_word)); // Reset previous word for ngram
              strncpy(prev_word, token, strlen(token)); // Update previous word

          }
          token = strtok(NULL, " ");
      }
      // strcpy_s(prev_word, MAX_STRING, ""); // Reset previous word for next sentence
      memcpy(prev_word, "", 1); // Reset previous word for next sentence
      // printf("\nlabels %p\n", labels);
      gs->train_words += sentence_length; // Increment train words by the number of words in the sentence
      // gs->train_words += word_count(word);
      gs->learning_rate_decay = gs->learning_rate * (1 - (double)gs->total_learned_lines / (double)(gs->total_lines * gs->iter));
      // gs->learning_rate_decay = gs->learning_rate;
      // if(thread_id == 1) {
      //   printf("\nftell: %lld, sentence: %s\n", ftell(fi), sen  );
      // }

      // printf("temp: %lld, line: %lld, max_line: %lld, sentence_length: %lld, label_length: %lld\n", temp, line, max_line, sentence_length, label_length);
      if (gs->debug_mode > 1) {
        // clock_t now = clock();
        struct timespec end_time;
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        printf("%clr: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  , loss: %f, Lines: %lld, setnence length: %lld, offset: %lld",
              13, gs->learning_rate_decay,
              gs->total_learned_lines / (double)(gs->iter * gs->total_lines) * 100,
              (gs->train_words / ((double)(end_time.tv_sec - gs->start.tv_sec + 1) * (double)1000)), gs->loss / gs->total_learned_lines, gs->total_learned_lines, sentence_length, offset
            );

        fflush(stdout);
      }

      long long golden_label = 0;
        
      if (sentence_length > 0) {
        // words 안에 있는 단어들에 대한 임베딩을 가져와서 평균을 구함
        memset(neu1, 0, gs->layer1_size * sizeof(float));
        memset(neu2, 0, gs->label_size * sizeof(float));
        for (long long j = 0; j <  sentence_length; j++) {
          if (words[j] != -1) {
            for (long long k = 0; k < gs->layer1_size; k++) {
              neu1[k] += gs->layer1[words[j] * gs->layer1_size + k];
            }
          }
        }
        

        for (long long j = 0; j < gs->layer1_size; j++) {
          //neu1: 1 x h
          neu1[j] /= sentence_length; // 평균을 구함
        }
        // implement Hiereical softmax
        if (gs->hs) {
          


          // for (long long j = 0; j < gs->label_size; j++) {
          //   for (long long k = 0; k < gs->layer1_size; k++) {
          //     neu2[j] += neu1[k] * gs->layer2[k * gs->label_size + j];
          //   }
          // }
          // Hierarchical softmax
          // Implement hierarchical softmax here
          // For now, we will just use the average of the words
          memset(neu1err, 0, gs->layer1_size * sizeof(float));
          for (int i = 0; i < label_length; i++) {
            // memset(neu1err, 0, gs->layer1_size * sizeof(float));
            if (labels[i] >= 0) {
              golden_label = labels[i];
            } else {
              continue ;
            }

            for (long long d=0;d<gs->labels[golden_label].codelen;d++) {
              
              float f = 0.0f;
              // layer1: vocab * hidden
              // layer2: hidden * label_size
              // neu1: 1 * hidden
              // neu1err: 1 * hidden
              // like neu2
              long long point = gs->labels[golden_label].point[d]; // label_size!c (1이면 1번째 lable을 가리키는 것
              long long l2 = point * gs->layer1_size;
              // printf("\n%lld gs->labels[golden_label].point[d]: %lld, code: %lld l2: %lld\n", d, gs->labels[golden_label].point[d],  gs->labels[golden_label].code[d], l2);
              for (long long j = 0; j < gs->layer1_size; j++) {
                // 1 * hidden * hidden * label_size
                // f += neu1[j] * gs->layer2[l2 + j];
                f += neu1[j] * gs->layer2[j * gs->label_size + point];
              }
              // if (f <= -6) {
              //   // neu2[d] = 0.0f;
              //   f = -6;
              //   continue;
              // } else if (f >= 6) {
              //   f = 6;
              //   continue ;
              // } else {
              // f = 1.0f / (1.0f + expf(-f));
              // }
              f = 1.0f / (1.0f + expf(-f)); // sigmoid function
              float g = gs->learning_rate_decay * (1 - gs->labels[golden_label].code[d] - f);
              if (g > 6) g = 6;
              if (g < -6) g = -6;
              for (long long j = 0; j < gs->layer1_size; j++) {
                neu1err[j] += g * gs->layer2[j * gs->label_size + point]; // to neu1
                gs->layer2[j * gs->label_size + point] += g * neu1[j]; // update layer2
              }
              // printf("%f ",f);
              // printf("\n");
              gs->loss += -logf(f + 1e-10f);
              if (isnan(gs->loss) || isinf(gs->loss)) {
                printf("[ERROR] Loss is NaN or Inf at line %lld, golden_label: %lld, f: %f\n", line, golden_label, f);
                getchar();
              }
            }
          }
          for (long long j = 0; j < sentence_length; j++) {
            if (words[j] != -1) {
              for (long long k = 0; k < gs->layer1_size; k++) {
                gs->layer1[words[j] * gs->layer1_size + k] += neu1err[k] / sentence_length; // Update layer1
              }
            }
          }

        } else { // if hierarchical softmax is not used
          // neu1 dot layer2
          for (long long j = 0; j < gs->label_size; j++) {
            for (long long k = 0; k < gs->layer1_size; k++) {
              neu2[j] += neu1[k] * gs->layer2[k * gs->label_size + j];
            }
          }

          float max = neu2[0];
          for (long long j = 1; j < gs->label_size; j++) {
            if (neu2[j] > max) max = neu2[j];
          }

          float sum = 0.0f;
          for (long long j = 0; j < gs->label_size; j++) {
              neu2[j] = expf(neu2[j] - max);
              sum += neu2[j];
          }
          for (long long j = 0; j < gs->label_size; j++)
              neu2[j] /= sum;

          float loss = 0.0f;

          memset(neu1err, 0, gs->layer1_size * sizeof(float));
          memset(neu2err, 0, gs->label_size * sizeof(float));
          for (int i = 0; i < label_length; i++) {
            if (labels[i] >= 0) {
              golden_label = labels[i];
            } else {
              break ;
            }

            float g = 0.0f;
            // multi answer 
            for (long long j = 0; j < gs->label_size; j++) {
              g = gs->learning_rate_decay* ((j == golden_label ? 1.0f : 0.0f) - neu2[j]);
              if (g > 6) g = 6;
              if (g < -6) g = -6;
              for (long long k = 0; k < gs->layer1_size; k++) {
                neu1err[k] += g * gs->layer2[k * gs->label_size + j]; // to neu1
                gs->layer2[k * gs->label_size + j] += g * neu1[k]; // update layer2
              } 
            }
            
            loss += -logf(neu2[golden_label] + 1e-10f);
            if (isnan(loss) || isinf(loss)) {
              getchar();
            }
          }
          // Update neu1err
          for (long long j = 0; j < sentence_length; j++) {
            if (words[j] != -1) {
              for (long long k = 0; k < gs->layer1_size; k++) {
                gs->layer1[words[j] * gs->layer1_size + k] += neu1err[k]; // Update layer1
              }
            }
          }

          if (label_length > 0) {
            loss /= label_length;
            gs->loss += loss;
          }
        }
      }
      
      sentence_length = 0;
      sentence_position = 0;
      sentence_start = 0;
      sentence_end = 0;   
      continue;
        
    }
    free(labels);
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
  // lines을 thread 개수만큼 분리
  // 각 데이터의 start offset, end offset 저장. 각 thread에서 실행할 라인 수 계산
  // 스레드에서는 start offset으로 fseek하고, 각 thread에서 실행할 데이터만큼 학습
  
  // long long total_line;
  gs->start_offsets= malloc(sizeof(long long) * gs->num_threads);
  gs->end_offsets = malloc(sizeof(long long) * gs->num_threads);
  gs->start_line_by_thread = malloc(sizeof(long long) * gs->num_threads);
  gs->total_line_by_thread = malloc(sizeof(long long) * gs->num_threads);
  gs->label_size = 0;

  //   // gs->total_offset = 0;
  // gs->start_offsets = malloc(sizeof(long long) * gs->num_threads);
  // gs->end_offsets = malloc(sizeof(long long) * gs->num_threads);
  // gs->offset_actual = malloc(sizeof(long long) * gs->num_threads);


  // printf("yes\n");
  compute_thread_offsets(fp, gs);
  // compute_thread_offsets(fp, gs->num_threads, gs->total_lines, gs->start_offsets, gs->end_offsets, &gs->total_offset, gs->offset_actual, gs->data_lines);
  // printf("[INFO] Total lines: %lld, Start offsets: ", gs->total_lines);

  // for (int i = 0; i < gs->num_threads; i++) {
  //   // fseek(fi, gs->start_offsets[i], SEEK_SET);
  //   long long line = count_lines_until_offset(gs->train_file, gs->end_offsets[i]);

  //   printf("%lld , what line: real: %lld, expected: %lld\n", gs->start_offsets[i], line, gs->total_line_by_thread[i]);
  // }
  // // given offset, what lines?


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
    // save_vocab(gs);
  }


  if (output_file[0] == 0) {
    printf("No output file specified. Exiting.\n");
    return;
  }

  printf("[INFO] Initializing network...\n");


  gs->pure_vocab_size = gs->vocab_size;
  gs->vocab_size += gs->bucket_size;
  initialize_network(gs);

  // printf("gs->vocab_size: %lld, gs->layer1_size: %lld, gs->label_size: %lld\n", 
  //        gs->vocab_size, gs->layer1_size, gs->label_size);

  // TODO: make unigram table

  for (int i = 0; i < gs->num_threads; i++) {
    // print start_offset, end offset, strat_line_by_thread, total_line_by_thread
    printf("[INFO] Thread %d: Start Offset: %lld, End Offset: %lld, Start Line: %lld, Total Lines: %lld\n",
           i, gs->start_offsets[i], gs->end_offsets[i], gs->start_line_by_thread[i], gs->total_line_by_thread[i]);
  }


  clock_gettime(CLOCK_MONOTONIC, &gs->start);
  // gs->start = clock();


  for (int i = 0; i < gs->num_threads; i++) {
    // Create threads for training
    thread_args *args = (thread_args *)malloc(sizeof(thread_args));
    args->id = i;
    args->gs = gs;
    // printf("[INFO] Creating thread %d with id %lld, global_setting: %p\n", i, args->id, (void *)args->gs);
    // printf("[INFO] %p, %p\n", &pt[i], args);
    // train_thread(args);

    pthread_create(&pt[i], NULL, train_thread, (thread_args *)args);

  }
  // printf("[INFO] Training threads started.\n");
  for (int i = 0; i < gs->num_threads; i++) {
    // Wait for threads to finish
    pthread_join(pt[i], NULL);
    printf("[INFO] Waiting for thread %d to finish : %lld...\n", i, gs->total_learned_lines);
  }
  printf("[INFO] All training threads finished.\n");
      struct timespec end_time;
    clock_gettime(CLOCK_MONOTONIC, &end_time);
  free(pt);

  printf("[INFO] Total time taken for training: %.2f seconds\n",
         (end_time.tv_sec - gs->start.tv_sec) +
         (end_time.tv_nsec - gs->start.tv_nsec) / 1e9);


  // TODO: output 쓰기.
  // 1. embedding 논문
  // 2. classification 논문

  printf("[INFO] Saving model...\n");

  save_model(output_file, gs);
  printf("[INFO] Model saved to %s\n", output_file);
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
    .min_count_vocab = 1,
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
  if ((i = get_arg_pos((char *)"-bucket", argc, argv)) > 0) {
    gs.vocab_hash_size = atoi(argv[i + 1]);
    gs.vocab_max_size = gs.vocab_hash_size; // Set max size to double the hash size
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

  // printf("%lld\n", gs.vocab_hash[886005]);
  train_model(&gs);

  printf("[INFO] FastText training completed.\n");

  return 0;
}
  