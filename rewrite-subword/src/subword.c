#include <locale.h>
#include <wchar.h>


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
  }

  posix_memalign((void **)&(gs->layer2), 64, (long long)gs->vocab_size * gs->layer1_size * sizeof(float));
  if (gs->layer2 == NULL) {
    fprintf(stderr, "Memory allocation failed for layer2\n");
    exit(1);
  }
  for (long long i = 0; i < size; i++) {
    gs->layer2[i] = ((float)rand() / RAND_MAX * 2 - 1) / size; // Initialize with random values between -1 and 1
  }    

  gs->random_table = (long long *)malloc(sizeof(long long) * gs->table_size);
  if (gs->random_table == NULL) {
    fprintf(stderr, "Memory allocation failed for random_table\n");
    exit(1);
  }

  // initUnigramTable(gs);
  double train_words_pow = 0.0;
  double d1, power = 0.75;
  for (long long i = 0; i < gs->vocab_size; i++) {
    train_words_pow += pow(gs->vocab[i].cn, power);
  }
  long long a = 0;
  d1=pow(gs->vocab[a].cn, power) / train_words_pow;
  for (long long i = 0; i < gs->table_size; i++) {
    gs->random_table[i] = a;
    if (i / (double)gs->table_size > d1) {
      a++;
      d1 += pow(gs->vocab[a].cn, power) / train_words_pow;
      if (a >= gs->vocab_size) {
        a = gs->vocab_size - 1; // Prevent going out of bounds
      }
    }
  }

  return ;
}

void save_vector(char *output_file, global_setting *gs) {
  // Implement the logic to save the vector representations
  printf("[INFO] Saving vector representations to %s\n", output_file);
  FILE *fo = fopen(output_file, "w");
  printf("[INFO] Writing vector representations to %s\n", output_file);
  fprintf(fo, "%lld %lld\n", gs->vocab_size, gs->layer1_size);
  for (long long i = 0; i < gs->vocab_size; i++) {
    fprintf(fo, "%s ", gs->vocab[i].word);
    for (long long j = 0; j < gs->layer1_size; j++) {
      fprintf(fo, "%f ", gs->layer1[i * gs->layer1_size + j]);
    }
    fprintf(fo, "\n");
  }
  printf("[INFO] Vector representations saved to %s\n", output_file);
  fclose(fo);
}

void save_model(char *output_file, global_setting *gs) {
  // Implement the logic to save the model
  FILE *fo = fopen(output_file, "wb");
  if (fo == NULL) {
    printf("Error opening file %s for writing\n", output_file);
    exit(1);
  }

  // gs를 binary 형태로 전부 저장. 언제든지 불러올 수 있는 형태로.
  printf("[INFO] Saving model to %s\n", output_file);
  fwrite(gs, sizeof(global_setting), 1, fo);
  printf("[INFO] ngram %lld\n",  gs->ngram);
  printf("[INFO] Global settings saved to %s\n", output_file);
  // Save the layer1, layer2, and output weights
  fwrite(gs->vocab_hash, sizeof(int), gs->vocab_hash_size, fo);
  printf("[INFO] Vocabulary hash table saved to %s %lld, %lld\n", output_file, gs->vocab_size, sizeof(vocab_word));
  fwrite(gs->vocab, sizeof(vocab_word), gs->vocab_max_size, fo);

  printf("[INFO] Vocabulary and labels saved to %s\n", output_file);
  fwrite(gs->layer1, sizeof(float), gs->vocab_size * gs->layer1_size, fo);

  printf("[INFO] Layer weights saved to %s\n", output_file);
  fwrite(gs->start_offsets, sizeof(long long), gs->num_threads+1, fo);
  fwrite(gs->end_offsets, sizeof(long long), gs->num_threads +1, fo);
  printf("[INFO] Thread offsets saved to %s\n", output_file);
  fwrite(gs->start_line_by_thread, sizeof(long long), gs->num_threads +1 , fo);
  fwrite(gs->total_line_by_thread, sizeof(long long), gs->num_threads+1, fo);
  printf("[INFO] Thread offsets saved to %s\n", output_file);



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
  
  
  

  FILE *fi = fopen(gs->train_file, "rb");
  
  // printf("[INFO] Thread %lld opened file %s\n", thread_id, gs->train_file);
  for (int iter = 0; iter < gs->iter; iter++) {
    
    
    // Reset sentence length and position for each iteration
    sentence_length = 0;
    sentence_position = 0;
    // Read the file line by line
    // fseek(fi, file_size / (long long)num_threads * (long long)thread_id , SEEK_END);
    fseeko(fi, gs->start_offsets[thread_id], SEEK_SET);
    // printf("[INFO] Thread %lld seeking to offset %lld\n", thread_id, gs->start_offsets[thread_id]);
    

    wchar_t word[MAX_STRING];
    wchar_t prev_word[MAX_STRING]; // only support for ngram=2
    wchar_t sen[MAX_SENTENCE_LENGTH];
    wchar_t concat_word[MAX_STRING];
    long long offset = 0;


    long long words[MAX_WORDS_PER_SENTENCE]; // [0, 1, 2, 3, 4 ...]
    long long ngram_words[MAX_WORDS_PER_SENTENCE];
    long long *subwords[MAX_WORDS_PER_SENTENCE]; // [0, 1, 2, 3, 4 ...]



    long long temp = 0;

    // while (1) {
    //   fgets(word, MAX_STRING, fi);
      
    // }
    long long line = 0;
    long long max_line = gs->total_line_by_thread[thread_id];
    

    float *neu1 = (float *)malloc(gs->layer1_size * sizeof(float));
    float *neu1err = (float *)malloc(gs->layer1_size * sizeof(float));


    long long avg_ngram = 0;
    long long avg_failure_ngram = 0;
    long long avg_word =0;

    if (neu1 == NULL || neu1err == NULL) {
        fprintf(stderr, "[ERROR] Memory allocation failed for neu1 or neu1err\n");
        exit(1);
    }
  
    const wchar_t *delim = L" "; // 공백 기준
    while ( fgetws(sen, MAX_SENTENCE_LENGTH, fi) && line < max_line) {

      memset(neu1, 0, gs->layer1_size * sizeof(float));
   
      line++;
      gs->total_learned_lines++;

      size_t len = wcslen(sen);
      if (len > 0 && sen[len - 1] == L'\n') {
          sen[len - 1] = L'\0';
      }

      // 단어 분리
      wchar_t *token = wcstok(sen, delim, NULL  );

      long long sentence_length = 0;
      long long ngram_sentences_length = 0;
      memset(words, -1, sizeof(words)); // Initialize words to -1 (unknown word
      memset(ngram_words, -1, sizeof(ngram_words)); // Initialize ngram_words to -1 (unknown word)

      while (token != NULL) {
        if (wcslen(token) >= MAX_STRING) {
          token = wcstok(NULL, delim, NULL);
          continue; // Skip tokens that are too long
        }

        swprintf(concat_word, sizeof(concat_word), L"<%s>", token);
        long long word_index = search_vocab(concat_word, gs);
        if (word_index != -1 && sentence_length < MAX_WORDS_PER_SENTENCE) {
            if (gs->sample > 0) {
            float ran = (sqrt(gs->vocab[word_index].cn / (gs->sample * gs->train_words)) + 1) * (gs->sample * gs->train_words) / gs->vocab[word_index].cn;
            double random_value = (double)rand() / ((double)RAND_MAX + 1.0); // Generate a random value between 0 and 1

            if (ran < random_value) {
              token = wcstok(NULL, delim, NULL);
              continue; // Skip this word
            }
          }
          if (gs->sisg > 0) {
            long long *subword_array = NULL;
            long long subword_array_length = search_subword(concat_word, gs, &subword_array); // Get subword for the word
            subwords[sentence_length] = subword_array; // Set the first subword index
          }
          words[sentence_length++] = word_index;
        } 
        token = wcstok(NULL, delim, NULL);
      }

      // skip-gram
      double loss = 0.0;
      for (int i = gs->window - 1; i < sentence_length - gs->window + 1; i++) {
        // i is center
        if ( i < 0 || i >= sentence_length) continue; // skip center word
        if (words[i] == -1) continue; // skip unknown word
        long long center_word = words[i]; // 중심 단어 구하고

        memset(neu1err, 0, gs->layer1_size * sizeof(float));

        memset(neu1, 0, gs->layer1_size * sizeof(float));

        for (long l = 0; l < gs->layer1_size; l++) {
          neu1[l] = gs->layer1[center_word * gs->layer1_size + l]; // 중심 단어의 vector representation position!
        }
        if (gs->sisg) {
          long long *subwords_array = subwords[i];
          long long l = 0;
          while (1) {
            // printf("[DEBUG] Subword index: %p, %lld\n", subwords_array, l);
            if (subwords_array[l] == -1) break; // end of subword array
            long long ngram_index = subwords_array[l] * gs->layer1_size;
            for (long long m = 0; m < gs->layer1_size; m++) {
              neu1[m] += gs->layer1[m + ngram_index]; // to neu1
            }
            l++;
          }
        }

        // 중심 단어로부터 맥락 단어 고름
        // printf("[DEBUG] Center word: %lld\n", center_word);

        for (int j = i - gs->window + 1; j < i + gs->window; j++) {
          // j is word!
          if (j == i || words[j] == -1) continue; // skip center word and unknown word
          if (j < 0 || j >= sentence_length) continue; // skip out of bounds
          // for (long long k = 0; k < gs->layer1_size; k++) {
          // 맥락 단어의 vector representation position!
          long long l1 = words[i] * gs->layer1_size;
          for (int d = 0; d < gs->negative + 1; d++) {
            // printf("[DEBUG] Negative sample %d for word: %lld\n", d, words[j]);
            long long label;
            long long target;
            if (d == 0) {
              target = words[j];
              label = 1;
            } else {
              // gs->vocab 에서 무작위로
              target = gs->random_table[rand() % gs->table_size]; // Randomly select a target word
              if (target == 0) target = rand() % (gs->vocab_size - 1) + 1; // Ensure target is not 0
              if (target == center_word) {
                // If target is the same as center word, skip it
                continue;
              }
              // target = get_negative_sample(gs, thread_id);
              label = 0;
            }


            long long l2 = target * gs->layer1_size;


            float f = 0.0f;
            for (long long k = 0; k < gs->layer1_size; k++) {
              f += neu1[k] * gs->layer2[k + l2];
            }
            float g = 0;

            if (f > 6) {
              f = 6;
              f = 1.0f / (1.0f + expf(-f)); // sigmoid function
              g = (label - 1) * gs->learning_rate_decay;
            }
            else if (f < -6) {
              f = -6;
              f = 1.0f / (1.0f + expf(-f)); // sigmoid function
              g = (label - 0) * gs->learning_rate_decay;
            }
            else {
              f = 1.0f / (1.0f + expf(-f)); // sigmoid function
              g = gs->learning_rate_decay * (label - f);
            }



            for (long long k = 0; k < gs->layer1_size; k++) {
              neu1err[k] += g * gs->layer2[k + l2]; // to neu1
              gs->layer2[l2 + k] += g * neu1[k]; // to layer2
            }

            if (label == 1)  {

              loss += -logf(f + 1e-10f); // log loss
            } else {
              loss += -logf(1 - f + 1e-10f); // log loss
            }
            if (isnan(loss) || isinf(loss)) {
              fprintf(stderr, "[ERROR] Loss is NaN at line %lld, word %lld, f: %f\n\n\n", line, i, f);
              exit(1);
            }
          }
          if (gs->sisg)  {
            long long *subwords_array = subwords[i];
            long long l = 0;
            long long subword_array_length = 0;
            while (1) { if(subwords_array[subword_array_length] == -1) break; subword_array_length++; }
            while (1) {
              // printf("[DEBUG] Subword index: %p, %lld\n", subwords_array, l);
              if (subwords_array[l] == -1) break; // end of subword array
              long long ngram_index = subwords_array[l] * gs->layer1_size;
              for (long long m = 0; m < gs->layer1_size; m++) {
                // neu1[m] += gs->layer2[m + ngram_index]; // to neu1
                gs->layer1[ngram_index + m] += neu1err[m] / ( 1 + subword_array_length); // update layer2 for subword
              }
              l++;
            } 
            for (long long k = 0; k < gs->layer1_size; k++) {
              gs->layer1[k + l1] += neu1err[k]  / ( 1 + subword_array_length);
            }
          } else {

            for (long long k = 0; k < gs->layer1_size; k++) {
              gs->layer1[k + l1] += neu1err[k];
            }
          }
        }

        

      }
      
    
      if (sentence_length > 0) {
        gs->loss += loss / ((gs->window * 2 - 1) * sentence_length); // average loss per word
      }

      gs->train_words += sentence_length; 
      gs->learning_rate_decay = gs->learning_rate * (1 - (double)gs->total_learned_lines / (double)(gs->total_lines * gs->iter));
      if (gs->debug_mode > 1 && temp % (gs->num_threads * 1000) == thread_id * 1000) {
        temp = 0;
        clock_t now = clock();
        struct timespec end_time;
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        float lines_sec = gs->total_learned_lines / ((float)(end_time.tv_sec - gs->start.tv_sec + 1));
        long long remain_lines = gs->total_lines * gs->iter - gs->total_learned_lines;
        long long eta_seconds = remain_lines / lines_sec;
        long long eta_hours = eta_seconds / 3600;
        long long eta_minutes = (eta_seconds % 3600) / 60;
        wprintf(L"%clr: %f  Progress: %.2f%%  Words/thread/sec: %.2fk, Lines/thread/sec: %.3fk, loss: %f, LossA: %f, Lines: %lld,  ETA: %lldH:%lldm:%llds",
               13, gs->learning_rate_decay,
              gs->total_learned_lines / (double)(gs->iter * gs->total_lines) * 100,
              (gs->train_words / ((double)(end_time.tv_sec - gs->start.tv_sec + 1) * (double)1000)), 
              (gs->total_learned_lines / ((double)(end_time.tv_sec - gs->start.tv_sec + 1) * (double)1000)),
               gs->loss / gs->total_learned_lines, gs->loss, gs->total_learned_lines,
               eta_hours, eta_minutes, eta_seconds % 60)
            ;

        fflush(stdout);
      }

      sentence_length = 0;
      sentence_position = 0;
      sentence_start = 0;
      sentence_end = 0;   
      continue;
        
    }
    free(neu1);
    free(neu1err);
  }
  fclose(fi);
  
  // Implement the saving output here
  pthread_exit(NULL);

}




void train_model(global_setting *gs) {
  // Placeholder for training model logic
  wprintf(L"[INFO] Training model with layer size: %lld\n", gs->layer1_size);
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

  gs->total_lines = count_lines_subword(fp);
  // lines을 thread 개수만큼 분리
  // 각 데이터의 start offset, end offset 저장. 각 thread에서 실행할 라인 수 계산
  // 스레드에서는 start offset으로 fseek하고, 각 thread에서 실행할 데이터만큼 학습
  
  // long long total_line;
  gs->start_offsets= malloc(sizeof(long long) * gs->num_threads);
  gs->end_offsets = malloc(sizeof(long long) * gs->num_threads);
  gs->start_line_by_thread = malloc(sizeof(long long) * gs->num_threads);
  gs->total_line_by_thread = malloc(sizeof(long long) * gs->num_threads);

  compute_thread_offsets_subword(fp, gs);


  wprintf(L"[INFO] read vocabulary...\n");

  if (read_vocab_file[0] != 0) {
    // Read vocabulary from file
    read_vocab(gs);
  } else {
    // Create vocabulary from training file
    create_vocab_from_train_file(gs);
  }

  wprintf(L"[INFO] save vocabulary...\n");

  if (save_vocab_file[0] != 0) {
    // Save vocabulary to file
    // save_vocab(gs);
  }


  if (output_file[0] == 0) {
    wprintf(L"No output file specified. Exiting.\n");
    return;
  }

  wprintf(L"[INFO] Initializing network...\n");


  gs->pure_vocab_size = gs->vocab_size;
  gs->vocab_size += gs->bucket_size;
  initialize_network(gs);

  for (int i = 0; i < gs->num_threads; i++) {
    wprintf(L"[INFO] Thread %d: Start Offset: %lld, End Offset: %lld, Start Line: %lld, Total Lines: %lld\n",
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
    wprintf(L"[INFO] Waiting for thread %d to finish : %lld...\n", i, gs->total_learned_lines);
  }
  wprintf(L"[INFO] All training threads finished.\n");
      struct timespec end_time;
    clock_gettime(CLOCK_MONOTONIC, &end_time);
  free(pt);

  wprintf(L"[INFO] Total time taken for training: %.2f seconds\n",
         (end_time.tv_sec - gs->start.tv_sec) +
         (end_time.tv_nsec - gs->start.tv_nsec) / 1e9);


  // TODO: output 쓰기.
  // 1. embedding 논문
  // 2. classification 논문

  wprintf(L"[INFO] Saving model...\n");

  save_model(output_file, gs);
  wprintf(L"[INFO] Model saved to %s\n", output_file);
  save_vector(save_vocab_file , gs);
  wprintf(L"[INFO] Vector Saved to %s\n", save_vocab_file);
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
  setlocale(LC_ALL, ""); // Set locale to support UTF-8
  fwide(stdout, 1); 
  int i;

  global_setting gs = {
    .layer1_size = 10, // Default layer size 
    .binary = 0, // Default binary output
    .debug_mode = 2, // Default debug mode
    .cbow = 0, // Default CBOW model
    .window = 5, // Default window size
    .min_count = 5, // Default minimum count for words  
    .num_threads = 20, // Default number of threads
    .min_reduce = 1, // Default minimum reduce count
    .hs = 0, // Default hierarchical softmax
    .negative = 5, // Default negative sampling
    .iter = 5, // Default number of iterations
    .learning_rate = 0.05, // Default learning rate
    .learning_rate_decay = 0.05, // Default learning rate decay
    .sample = 0.0001, // Default subsampling rate
    .train_file = "", // Default training file
    .output_file = "", // Default output file
    .save_vocab_file = "", // Default vocabulary save file
    .read_vocab_file = "", // Default vocabulary read file
    .vocab_hash_size = 10000000, // Default vocabulary hash size


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

    .total_lines = 0, // Total lines in training file
    .start_offsets = NULL, // Start offsets for each thread
    .end_offsets = NULL, // End offsets for each thread
    .start_line_by_thread = NULL, // Actual offset for each thread\
    .ngram = 1,
    .bucket_size = 0,
    .min_count_vocab = 5,
    .minx = 2,
    .maxx = 6,
    .sisg = 1,

    .table_size = 1e8, // Default size for unigram table
    

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


  // printf("%lld\n", gs.vocab_hash[886005]);
  train_model(&gs);

  printf("[INFO] FastText training completed.\n");

  return 0;
}
  