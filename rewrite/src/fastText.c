
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
    // gs->layer1[i] = (float)rand() / RAND_MAX * 2 - 1; // Initialize with random values between -1 and 1
    // uniform
    gs->layer1[i] = ((float)rand() / RAND_MAX * 2 - 1) / size;
  }


  printf("[INFO] Allocated memory for layer1 with size: %lld\n", gs->vocab_size * gs->layer1_size * sizeof(float));
  posix_memalign((void **)&(gs->layer2), 64, gs->layer1_size * gs->class_size * sizeof(float));
  if (gs->layer2 == NULL) {
    fprintf(stderr, "Memory allocation failed for layer2\n");
    exit(1);
  }

  size = gs->layer1_size * gs->class_size;
  for (long long i = 0; i < gs->layer1_size * gs->class_size; i++) {
    gs->layer2[i] = ((float)rand() / RAND_MAX * 2 - 1) / size; // Initialize with random values between -1 and 1
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
  // TODO: if classifation, gs->labels should be passed
  // create_binary_tree(gs->vocab, gs->vocab_size);
  create_binary_tree(gs->labels, gs->class_size);
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
  printf("[INFO] Global settings saved to %s\n", output_file);
  // Save the layer1, layer2, and output weights
  fwrite(gs->vocab_hash, sizeof(int), gs->vocab_hash_size, fo);
  printf("[INFO] Vocabulary hash table saved to %s %lld, %lld\n", output_file, gs->vocab_size, sizeof(vocab_word));
  fwrite(gs->vocab, sizeof(vocab_word), gs->vocab_max_size, fo);
  fwrite(gs->label_hash, sizeof(int), gs->label_hash_size, fo);
  fwrite(gs->labels, sizeof(vocab_word), gs->label_max_size, fo);
  // fwrite(gs->labels, sizeof(vocab_word), gs->class_size, fo);
  printf("[INFO] Vocabulary and labels saved to %s\n", output_file);
  fwrite(gs->layer1, sizeof(float), gs->vocab_size * gs->layer1_size, fo);
  // for (long long i = 0; i < gs->vocab_size * gs->layer1_size; i++) {
  //   printf("[INFO] Layer1[%lld]: %f\n", i, gs->layer1[i]);
  // }
  printf("[INFO] Layer1 weights saved to %s\n", output_file);
  fwrite(gs->layer2, sizeof(float), gs->layer1_size * gs->class_size, fo);
  for (long long i = 0; i < gs->layer1_size * gs->class_size; i++) {
    printf("[INFO] Layer2[%lld]: %f\n", i, gs->layer2[i]);
  }
  printf("[INFO] Layer2 weights saved to %s\n", output_file);
  fwrite(gs->output, sizeof(float), gs->class_size, fo);
  printf("[INFO] Layer weights saved to %s\n", output_file);
  fwrite(gs->start_offsets, sizeof(long long), gs->num_threads + 1, fo);
  fwrite(gs->end_offsets, sizeof(long long), gs->num_threads, fo);
  printf("[INFO] Thread offsets saved to %s\n", output_file);
  fwrite(gs->start_line_by_thread, sizeof(long long), gs->num_threads + 1, fo);
  fwrite(gs->total_line_by_thread, sizeof(long long), gs->num_threads + 1, fo);
  printf("[INFO] Thread offsets saved to %s\n", output_file);
  // Save the vocabulary
  // Save the vocabulary hash table

  fclose(fo);
}

void *train_thread(thread_args *args) {
  // printf("starting train thread %lld\n", args->id);
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
    char sen[MAX_SENTENCE_LENGTH];
    // long long labels[MAX_LABELS]; // [0, 3, -1, -1, -1 ...]
    long long *labels = malloc(sizeof(long long) * gs->class_size);
    if (labels == NULL) {
        fprintf(stderr, "[ERROR] Memory allocation failed for labels\n");
        exit(1);
    }

    long long words[MAX_WORDS_PER_SENTENCE]; // [0, 1, 2, 3, 4 ...]
          // printf("\nlabels %p\n", labels);


    long long offset_length = gs->end_offsets[thread_id] - gs->start_offsets[thread_id] + 1;

    long long temp = 0;
    long long i = gs->start_offsets[thread_id];
    // while (1) {
    //   fgets(word, MAX_STRING, fi);
      
    // }
    long long line = 0;
    long long max_line = gs->total_line_by_thread[thread_id];
    // printf("[INFO] Thread %lld started training with %lld lines\n", thread_id, max_line);
    

    float *neu1 = (float *)malloc(gs->layer1_size * sizeof(float));
    float *neu2 = (float *)malloc(gs->class_size * sizeof(float));
    float *neu1err = (float *)malloc(gs->layer1_size * sizeof(float));
    float *neu2err = (float *)malloc(gs->class_size * sizeof(float));

    // printf("[INFO] Thread %lld started training with %lld lines\n", thread_id, max_line);
  
    while (fgets(sen, MAX_SENTENCE_LENGTH, fi) && line < max_line) {

      line++;
      temp++;
      gs->total_learned_lines++;
      // word를 label, words로 분리.
      // 줄 끝 개행 문자 제거
      sen[strcspn(sen, "\n")] = 0;

      // 단어 분리
      char *token = strtok(sen, " ");
      long long sentence_length = 0;
      long long label_length = 0;
      memset(labels, -1, sizeof(labels)); // Initialize labels to -1

      memset(words, -1, sizeof(words)); // Initialize words to -1 (unknown word
      while (token != NULL) {
          if (strncmp(token, "__", 2) == 0) {
              // 라벨인 경우 __label_1__
              long long label_index = search_label(token, gs);
              if (label_index != -1 && label_index < MAX_LABELS) {
                  labels[label_length++] = label_index;  // Set the label index to 1
              } else {
                labels[label_length++] = -1; // unknown label
              }
          } else {
              // 일반 단어인 경우
              long long word_index = search_vocab(token, gs);
              if (word_index != -1 && sentence_length < MAX_WORDS_PER_SENTENCE) {
                  words[sentence_length++] = word_index;
              } else {
                  words[sentence_length++] = -1; // unknown word
              }
          }
          token = strtok(NULL, " ");
      }      
      // printf("\nlabels %p\n", labels);
      gs->train_words += sentence_length; // Increment train words by the number of words in the sentence
      // gs->train_words += word_count(word);
      gs->learning_rate_decay = gs->learning_rate * (1 - (double)gs->total_learned_lines / (double)(gs->total_lines * gs->iter));
      if (gs->debug_mode > 1 && gs->total_learned_lines % 1000 == 0) {
        // temp = 0;
        clock_t now = clock();
        printf("%clr: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  , loss: %f, Lines: %lld",
              13, gs->learning_rate_decay,
              gs->total_learned_lines / (double)(gs->iter * gs->total_lines) * 100,
              gs->train_words / ((double)(now - gs->start + 1) / (double)CLOCKS_PER_SEC * 1000) / gs->num_threads, gs->loss / gs->total_learned_lines, gs->total_learned_lines);

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
      long long golden_label = 0;
      // printf("\nlabels %p\n", labels);
      // 
        
      if (sentence_length > 0) {

        // printf("\nsentence length: %lld %lld %lld\n", sentence_length, gs->class_size, gs->layer1_size);
        

        // words 안에 있는 단어들에 대한 임베딩을 가져와서 평균을 구함
        memset(neu1, 0, gs->layer1_size * sizeof(float));
        memset(neu2, 0, gs->class_size * sizeof(float));
        


        
        for (long long j = 0; j < sentence_length; j++) {
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
        if (gs->hierarchical_softmax) {
          // Hierarchical softmax
          // Implement hierarchical softmax here
          // For now, we will just use the average of the words
          for (int i = 0; i < label_length; i++) {
            if (labels[i] >= 0) {
              golden_label = labels[i];
            } else {
              continue ;
            }
            for (long long d=0;d<gs->labels[golden_label].codelen;d++) {
              
              float f = 0.0f;
              long long l2 = gs->labels[golden_label].point[d] * gs->layer1_size;
              for (long long j = 0; j < gs->layer1_size; j++) {
                f += neu1[j] * gs->layer2[l2 + j];
              }
              if (f <= -6) {
                // neu2[d] = 0.0f;
                continue ;
              } else if (f >= 6) {
                continue ;
              } else {
                f = 1.0f / (1.0f + expf(-f));
              }
              float g = gs->learning_rate_decay * (1 - gs->labels[golden_label].code[d] - f);
              for (long long j = 0; j < gs->layer1_size; j++) {
                neu1err[j] += g * gs->layer2[l2 + j]; // to neu1
                gs->layer2[l2 + j] += g * neu1[j]; // update layer2
              }
              // printf("%f ",f);
              // printf("\n");
              gs->loss += -logf(f + 1e-10f);
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
          for (long long j = 0; j < gs->class_size; j++) {
            // neu2: 1 x c
            neu2[j] = 0.0f;
            for (long long k = 0; k < gs->layer1_size; k++) {
              neu2[j] += neu1[k] * gs->layer2[k * gs->class_size + j];
            }
          }

          
          // 
          float max = neu2[0];
          for (long long j = 1; j < gs->class_size; j++) {
            if (neu2[j] > max) max = neu2[j];
            // printf("%f ", neu2[j]);
          }

          // printf("\n");
          // softmax
          // softmax: 기존과 동일
          float sum = 0.0f;
          for (long long j = 0; j < gs->class_size; j++) {
              neu2[j] = expf(neu2[j] - max);
              sum += neu2[j];
          }
          for (long long j = 0; j < gs->class_size; j++)
              neu2[j] /= sum;

          float loss = 0.0f;
          
        
          for (int i = 0; i < label_length; i++) {
            if (labels[i] >= 0) {
              golden_label = labels[i];
            } else {
              break ;
            }
            memset(neu1err, 0, gs->layer1_size * sizeof(float));
            memset(neu2err, 0, gs->class_size * sizeof(float));

            float g = 0.0f;
            // multi answer 
            for (long long j = 0; j < gs->class_size; j++) {
              g = gs->learning_rate_decay* ((j == golden_label ? 1.0f : 0.0f) - neu2[j]);
              if (g > 6) g = 6;
              if (g < -6) g = -6;
              for (long long k = 0; k < gs->layer1_size; k++) {
                neu1err[k] += g * gs->layer2[k * gs->class_size + j]; // to neu1
                gs->layer2[k * gs->class_size + j] += g * neu1[k]; // update layer2
              } 
            }
            
            loss += -logf(neu2[golden_label] + 1e-10f);
            // gs->loss += 1;
            // printf("%f ",/ -logf(neu2[golden_label] + 1e-10f));


            // Update neu1err
            for (long long j = 0; j < sentence_length; j++) {
              if (words[j] != -1) {
                for (long long k = 0; k < gs->layer1_size; k++) {
                  gs->layer1[words[j] * gs->layer1_size + k] += neu1err[k] / sentence_length; // Update layer1
                }
              }
            }
          }
          if (label_length > 0) {
            loss /= label_length;
            gs->loss += loss;
          }
        }

        // free(neu1);
        // free(neu2);
        // free(neu1err);  
        // free(neu2err);
      }
        // Reset sentence length and position for the next sentence
      
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
  gs->class_size = 0;

  //   // gs->total_offset = 0;
  // gs->start_offsets = malloc(sizeof(long long) * gs->num_threads);
  // gs->end_offsets = malloc(sizeof(long long) * gs->num_threads);
  // gs->offset_actual = malloc(sizeof(long long) * gs->num_threads);


  // printf("yes\n");
  compute_thread_offsets(fp, gs);
  // compute_thread_offsets(fp, gs->num_threads, gs->total_lines, gs->start_offsets, gs->end_offsets, &gs->total_offset, gs->offset_actual, gs->data_lines);


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

  for (int i = 0; i < gs->num_threads; i++) {
    // print start_offset, end offset, strat_line_by_thread, total_line_by_thread
    printf("[INFO] Thread %d: Start Offset: %lld, End Offset: %lld, Start Line: %lld, Total Lines: %lld\n",
           i, gs->start_offsets[i], gs->end_offsets[i], gs->start_line_by_thread[i], gs->total_line_by_thread[i]);
  }

  gs->start = clock();

  for (int i = 0; i < gs->num_threads; i++) {
    // Create threads for training
    thread_args *args = (thread_args *)malloc(sizeof(thread_args));
    args->id = i;
    args->gs = gs;
    // printf("[INFO] Creating thread %d with id %lld, global_setting: %p\n", i, args->id, (void *)args->gs);
    // printf("[INFO] %p, %p\n", &pt[i], args);

    pthread_create(&pt[i], NULL, train_thread, (thread_args *)args);

  }
  // printf("[INFO] Training threads started.\n");
  for (int i = 0; i < gs->num_threads; i++) {
    // Wait for threads to finish
    pthread_join(pt[i], NULL);
  }
  printf("[INFO] All training threads finished.\n");
  free(pt);


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
    .label_hash_size = 1000000, // Default label hash size

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
    .hierarchical_softmax = 0, // Default hierarchical softmax
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
  if ((i = get_arg_pos((char *)"-bucket", argc, argv)) > 0) gs.vocab_hash_size = atoi(argv[i + 1]);
  if ((i = get_arg_pos((char *)"-hs", argc, argv)) > 0) gs.hierarchical_softmax = atoi(argv[i + 1]);

  // printf("[INFO] Argument parsing completed.\n");
  // bag of tricks for efficient text classification additional setting
  // lr
  // wordNgrams
  // bucket
  gs.vocab = (vocab_word *)calloc(gs.vocab_max_size, sizeof(vocab_word));
  gs.vocab_hash = (int *)calloc(gs.vocab_hash_size, sizeof(int));
  for (int j = 0; j < gs.vocab_hash_size; j++) {
    gs.vocab_hash[j] = -1; // Initialize the vocabulary hash table
  }
  gs.labels = (vocab_word *)calloc(gs.label_hash_size, sizeof(vocab_word));
  gs.label_hash = (int *)calloc(gs.label_hash_size, sizeof(int));
  for (int j = 0; j < gs.label_hash_size; j++) {
    gs.label_hash[j] = -1; // Initialize the label hash table
  }

  train_model(&gs);

  printf("[INFO] FastText training completed.\n");

  return 0;
}
  