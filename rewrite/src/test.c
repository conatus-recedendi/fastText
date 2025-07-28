
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
  // TODO: if classifation, gs->labels should be passed
  create_binary_tree(gs->vocab, gs->vocab_size);
  // create_binary_tree(gs->labels, gs->class_size);
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
  if (fi == NULL) {
    printf("Error opening file %s for reading\n", load_model_file);
    exit(1);  
  }
  printf("[INFO] Loading model from file: %s\n", load_model_file);
  fread(gs, sizeof(global_setting), 1, fi);

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



  for (int j = 0; j < gs->vocab_max_size; j++) {
    // gs->vocab[j].word = (char *)calloc(MAX_STRING, sizeof(char));
    // gs->vocab[j].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    // gs->vocab[j].point = (int *)calloc(MAX_CODE_LENGTH + 1, sizeof(int));
  }

  // posix_memalign((void **)&(gs->layer1), 64, (long long)gs->vocab_size * gs->layer1_size * sizeof(float));
  fread(gs->vocab_hash, sizeof(int), gs->vocab_hash_size, fi);
  fread(gs->vocab, sizeof(vocab_word), gs->vocab_max_size, fi);
  
  printf("[INFO] Vocabulary loaded from %s\n", load_model_file);
  
  
  gs->layer1 = (float *)calloc(gs->vocab_size * gs->layer1_size, sizeof(float));
  if (gs->layer1 == NULL) {
    fprintf(stderr, "Memory allocation failed for layer1\n");
    exit(1);
  }
  posix_memalign((void **)&(gs->layer2), 64, gs->layer1_size * gs->class_size * sizeof(float));
  if (gs->layer2 == NULL) {
    fprintf(stderr, "Memory allocation failed for layer2\n");
    exit(1);
  }

  posix_memalign((void **)&(gs->output), 64, gs->class_size * sizeof(float));
  if (gs->output == NULL) {
    fprintf(stderr, "Memory allocation failed for output\n");
    exit(1);
  }

  // fread(gs->labels, sizeof(vocab_word), gs->class_size, fi);
  fread(gs->layer1, sizeof(float), gs->vocab_size * gs->layer1_size, fi);
  // printf("[INFO] Layer1 weights loaded from %s, read %zu elements\n", load_model_file, read);

  // for (long long i = 0; i < gs->vocab_size * gs->layer1_size; i++) {
  //   printf("[INFO] Layer1[%lld]: %f\n", i, gs->layer1[i]);
  // }

  fread(gs->layer2, sizeof(float), gs->layer1_size * gs->class_size, fi);
  fread(gs->output, sizeof(float), gs->class_size, fi);
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
}


void test_thread(global_setting *gs) {
  // Implement the test logic here
  long long thread_id = 0; // Assuming single thread for testing
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
  

  // printf("[INFO] Thread %lld started testing...\n", thread_id);
  

  FILE *fi = fopen(gs->test_file, "rb");

    // Reset sentence length and position for each iteration
  sentence_length = 0;
  sentence_position = 0;
  // Read the file line by line
  fseek(fi, 0, SEEK_SET);
  

  // char word[MAX_SENTENCE_LENGTH];
  char sen[MAX_SENTENCE_LENGTH];
  long long labels[MAX_LABELS]; // [1, 0, 0, 1, 0 ...]
  long long words[MAX_WORDS_PER_SENTENCE]; // [0, 1, 2, 3, 4 ...]


  long long temp = 0;
  // while (1) {
  //   fgets(word, MAX_STRING, fi);
    
  // }
  long long line = 0;
  long long max_line = gs->total_lines;

  long long correct_cnt = 0;
  long long total_cnt = 0;
  float precision = 0.0f;
  float recall = 0.0f;
  float f1_score = 0.0f;
  long long tp_cnt = 0;
  long long tn_cnt = 0;
  long long fp_cnt = 0;
  long long fn_cnt = 0;


  float *neu1 = (float *)malloc(gs->layer1_size * sizeof(float));
  float *neu2 = (float *)malloc(gs->class_size * sizeof(float));


  while (fgets(sen, MAX_SENTENCE_LENGTH, fi) && line < max_line) {
    line++;
    // gs->total_learned_lines++;
    // word를 label, words로 분리.
    // 줄 끝 개행 문자 제거
    sen[strcspn(sen, "\n")] = 0;

    // 단어 분리
    char *token = strtok(sen, " ");
    long long sentence_length = 0;
    memset(labels, 0,  sizeof(labels)); // Initialize labels to 0
    memset(words, -1,  sizeof(words)); // Initialize words to -1 (unknown word)
    while (token != NULL) {
      if (strncmp(token, "__", 2) == 0) {
          
            // 라벨인 경우 __label_1__
            long long label_index = atoi(token + 9) - 1;
            if (label_index != -1 && label_index < MAX_LABELS) {
                labels[label_index] = 1;
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
    gs->train_words += sentence_length; // Increment train words by the number of words in the sentence
    // gs->train_words += word_count(word);
    gs->learning_rate_decay = gs->learning_rate * (1 - (double)gs->total_learned_lines / (double)(gs->total_lines * gs->iter));
    
    if (gs->debug_mode > 1) {
      temp = 0;
      clock_t now = clock();

      // printf("%clr: %f  Progress: %.2f%%  Words/thread/sec: %.2fk",
      //       13, gs->learning_rate_decay,
      //       gs->total_learned_lines / (double)(gs->iter * gs->total_lines) * 100,
      //       gs->train_words / ((double)(now - gs->start + 1) / (double)CLOCKS_PER_SEC * 1000) / gs->num_threads);
      // fflush(stdout);
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



    if (sentence_length > 0) {

      // words 안에 있는 단어들에 대한 임베딩을 가져와서 평균을 구함
      memset(neu1, 0, gs->layer1_size * sizeof(float));
      memset(neu2, 0, gs->class_size * sizeof(float));
      // memset(neu1err, 0, gs->layer1_size * sizeof(float));
      // memset(neu2err, 0, gs->class_size * sizeof(float));

      
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
      for (long long j = 0; j < gs->class_size; j++)
          neu2[j] = expf(neu2[j] - max);

      float sum = 0.0f;
      for (long long j = 0; j < gs->class_size; j++)
          sum += neu2[j];

      for (long long j = 0; j < gs->class_size; j++)
          neu2[j] /= sum;
    

      // neu2 copy and sort by decreasign
      // but, there are remianing information to index to original neu2

      float *neu2_sorted = (float *)malloc(gs->class_size * sizeof(float));
      long long *index_sorted = (long long *)malloc(gs->class_size * sizeof(long long));
      for (long long j = 0; j < gs->class_size; j++) {
        neu2_sorted[j] = neu2[j];
        index_sorted[j] = j;
      }

      // sort neu2_sorted and index_sorted by neu2_sorted
      for (long long j = 0; j < gs->class_size - 1; j++) {
        for (long long k = j + 1; k < gs->class_size; k++) {
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

      printf("[INFO] Sorted neu2: %f %lld", neu2_sorted[0], index_sorted[0]);
    
      long long *gold = (long long *)malloc(gs->class_size * sizeof(long long));
      long long *predicted = (long long *)malloc(gs->class_size * sizeof(long long));

      long long gold_length = 0;
      for (long long j = 0; j < gs->class_size; j++) {
        if (labels[j] == 1) {
          gold[gold_length++] = j;
        }
      }
      // printf("[INFO] Gold length: %lld, Predicted length: %lld\n", gold_length, gs->top_k);


      long long predicted_length = 0;
      for (long long j = 0; j < gs->class_size; j++) {
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
      long long local_tn_cnt = gs->class_size - (local_tp_cnt + local_fp_cnt + local_fn_cnt);
      printf("[INFO] TP: %lld, FP: %lld, Gold length: %lld, Predicted length: %lld\n", local_tp_cnt, local_fp_cnt, gold_length, predicted_length);

      tp_cnt += local_tp_cnt;
      fp_cnt += local_fp_cnt;
      fn_cnt += local_fn_cnt;
      tn_cnt += local_tn_cnt;
      total_cnt += gold_length; // Total number of true labels
    }

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
  printf("[INFO] Total sentences processed: %lld\n", line);
}





void test_model(global_setting *gs) {
  // Placeholder for training model logic
  printf("[INFO] Testing model with layer size: %lld\n", gs->layer1_size);
  // Implement the training logic here


  printf("[INFO] Loading model from file: %s\n", gs->load_model_file);


  printf("[INFO] Starting training threads...\n");
  printf("[INFO] Initializing network... %lld %lld\n", gs->layer1_size, gs->class_size);
  test_thread(gs);

  printf("[INFO] All test threads finished.\n");
  // print precision@k, recall@k, f1-score
  


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
    .total_lines = 0, // Total lines in training file
    .start_offsets = NULL, // Start offsets for each thread
    .end_offsets = NULL, // End offsets for each thread
    .start_line_by_thread = NULL, // Actual offset for each thread
    .top_k = 1, // Default top K for classification
  };
  printf("[INFO] FastText test started.\n");

    if ((i = get_arg_pos((char *)"-load-model", argc, argv)) > 0) {
    strcpy(gs.load_model_file, argv[i + 1]);
    if (gs.load_model_file[0] == 0) {
      fprintf(stderr, "No model file specified for loading. Exiting.\n");
      return 1;
    }
  } else {
    fprintf(stderr, "No model file specified for loading. Exiting.\n");
    return 1;
  }
  
  load_model(gs.load_model_file, &gs);

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
  printf("class_size: %lld\n", gs.class_size);
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

  test_model(&gs);

  printf("[INFO] FastText test completed.\n");

  return 0;
}
  