// 주어진 데이터로부터 vocab만들기
#include <wchar.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>   
#include "config.h"
#include "vocab.h"
// Reads a single word from a file, assuming space + tab + EOL to be word boundaries

void compute_thread_offsets_subword(FILE *fp, global_setting *gs) {
    const int T = gs->num_threads;
    if (T <= 0) { fprintf(stderr, "num_threads must be > 0\n"); exit(1); }

    long long total_lines = gs->total_lines;
    long long *start_offsets        = gs->start_offsets;
    long long *end_offsets          = gs->end_offsets;
    long long *start_line_by_thread = gs->start_line_by_thread;
    long long *total_line_by_thread = gs->total_line_by_thread;

    // 라인 분할 경계(라인 번호) 계산
    // target_line[0]=0, target_line[T]=total_lines
    long long *target_line = (long long *)malloc((T + 1) * sizeof(long long));
    if (!target_line) { perror("malloc target_line"); exit(1); }
    for (int i = 0; i <= T; i++) {
        target_line[i] = (total_lines * i) / T;
    }
    for (int i = 0; i < T; i++) {
        start_line_by_thread[i]   = target_line[i];
        total_line_by_thread[i]   = target_line[i + 1] - target_line[i];
    }
    

    // 파일 처음으로
    if (fseeko(fp, 0, SEEK_SET) != 0) { perror("fseeko"); exit(1); }


    // 스레드 0은 파일 처음에서 시작
    start_offsets[0] = 0;
    // 라인 단위로 순회하며 “해당 라인의 시작 바이트 오프셋”을 잡는다.
    // getline은 호출 시점의 파일 위치(=해당 라인 시작)에서 읽기 시작하므로,
    // "라인 시작 오프셋"은 호출 직전의 ftello(fp) 값.
    char *line = NULL;
    size_t cap = 0;
    long long curr_line = 0;       // 1-based 증가용 카운터
    int next_thread = 1;           // 1..T-1 까지 채울 예정

    while (1) {
      off_t pos_before = ftello(fp);              // 이 라인 시작 바이트 오프셋
      ssize_t nread = getline(&line, &cap, fp);   // 한 줄 읽기 (POSIX)

      if (nread < 0) break;                       // EOF

      curr_line++;

      // target_line[next_thread] 번째 라인의 "시작 위치"를 다음 스레드의 시작으로 삼음
      // curr_line 은 1-based, target_line[...]은 0-based 라인번호이므로
      // "현재 읽은 라인 번호(1-based)가 (target_line[next_thread]+1)"일 때가 그 라인의 시작.
      while (next_thread < T && curr_line == target_line[next_thread] + 1) {
          // 스레드 next_thread 의 시작 오프셋은 pos_before (이번 라인 시작)
          start_offsets[next_thread] = (long long)pos_before;
          // 이전 스레드의 끝 오프셋은 pos_before (반개구간 [start, end) 표준화)
          end_offsets[next_thread - 1] = (long long)pos_before;
          next_thread++;
      }
    }

    // 마지막 스레드의 끝 오프셋 = 파일 끝
    off_t file_end = ftello(fp);
    end_offsets[T - 1] = (long long)file_end;

    free(line);
    free(target_line);

    // (선택) 디버그 프린트

    for (int i = 0; i < T; i++) {
        fprintf(stderr, "[offsets] thr=%d lines=[%lld..%lld) cnt=%lld  bytes=[%lld..%lld)\n",
            i,
            start_line_by_thread[i],
            start_line_by_thread[i] + total_line_by_thread[i],
            total_line_by_thread[i],
            start_offsets[i],
            end_offsets[i]);
    }
}

long count_lines_subword(FILE *fp) {
    long lines = 0;
    wint_t c;
    while ((c = fgetwc(fp)) != WEOF) {
        if (c == L'\n') lines++;
    }
    rewind(fp);
    return lines;
}


// wchar_t 단어 읽기
long long read_word(wchar_t *word, FILE *f_in) {
  wint_t wc;
  size_t i = 0;

  // 1) 선행 공백 스킵
  while ((wc = fgetwc(f_in)) != WEOF) {
      if (!iswspace(wc)) break;
  }
  if (wc == WEOF) return 0;

  // 2) 첫 글자
  word[i++] = (wchar_t)wc;

  // 3) 공백 전까지 읽기
  while ((wc = fgetwc(f_in)) != WEOF) {
      if (iswspace(wc)) break;
      if (i < MAX_STRING - 1) {
          word[i++] = (wchar_t)wc;
      }
  }

  word[i] = L'\0';
  return 1;
}


int get_word_hash(wchar_t *word, global_setting *gs) {
  unsigned long long vocab_hash_size = gs->vocab_hash_size;
  vocab_word *vocab = gs->vocab;
  int *vocab_hash = gs->vocab_hash;
  unsigned long long hash = 0;
  for (unsigned long long a = 0; a < wcslen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

int get_subword_hash(wchar_t *word, global_setting *gs) {
  unsigned long long subword_hash_size = gs->subword_hash_size;
  vocab_word *subword_vocab = gs->subword_vocab;
  int *subword_hash = gs->subword_hash;
  unsigned long long hash = 0;
  for (unsigned long long a = 0; a < wcslen(word); a++) hash = hash * 257 + word[a];
  hash = hash % subword_hash_size;
  return hash;
}

long long create_subword(wchar_t *word, global_setting *gs) {
    // word는 "<abcde>" 와 같이 전체 문장이 옴!
  // serach_subword는 min_elngth~amx_length 길이로 단어를 분리하여 반환한다
  // minx = 2, maxx = 5  이면
  // x = 2 -> <a, ab, bc, cd, de, e>
  // x = 3 -> <abc, bcd, cde>
  // x = 4 -> <abcd, bcde>
  // x = 5 -> <abcde>
  // ret -> [<a, ab, bc, cd, de, e>, <abc, bcd, cde>, <abcd, bcde>, <abcde>]
  // 이때, <a 와 같은 값들은 hash index이고, 전체 배열은 malloc으로 할당 후 인자 subword에 반환해야 한다
  // 반환 값은 subword 배열의 길이이다.
  // subword_array 기본으로 Null을 가지고 있어야 한다. -> 여기서 malloc
  // 마지막은 -1로 끝나야 한다.
  long long min_length  = gs->minx; // default: 2 , min 2
  long long max_length = gs->maxx; // default: 5, max 6

  if (wcslen(word) < min_length) {
    return -1; // Return -1 if the word length is not within the specified range
  }
  long long subword_count = 0;
  long long word_length = wcslen(word);
  long long subword_size = (word_length - min_length + 1) * (max_length - min_length + 1);
  long long temp_subword_index = -1;

  for (long long length = min_length; length <= max_length; length++) {
    for (long long start = 0; start <= word_length - length; start++) {
      wchar_t subword[MAX_STRING];
      wcsncpy(subword, word + start, length);
      subword[length] = L'\0'; // Null-terminate the subword
      long long subword_index = search_vocab(subword, gs);
      if (subword_index == -1) {
        temp_subword_index = add_word_to_vocab(subword, gs);
        gs->vocab[temp_subword_index].cn = 1;
      } else {
        gs->vocab[subword_index].cn++;
      }
    }
  }

  return 0; // Return the number of subwords found
}

long long search_subword(wchar_t *word, global_setting *gs, long long **subword_array) {
  // word는 "<abcde>" 와 같이 전체 문장이 옴!
  // serach_subword는 min_elngth~amx_length 길이로 단어를 분리하여 반환한다
  // minx = 2, maxx = 5  이면
  // x = 2 -> <a, ab, bc, cd, de, e>
  // x = 3 -> <abc, bcd, cde>
  // x = 4 -> <abcd, bcde>
  // x = 5 -> <abcde>
  // ret -> [<a, ab, bc, cd, de, e>, <abc, bcd, cde>, <abcd, bcde>, <abcde>]
  // 이때, <a 와 같은 값들은 hash index이고, 전체 배열은 malloc으로 할당 후 인자 subword에 반환해야 한다
  // 반환 값은 subword 배열의 길이이다.
  // subword_array 기본으로 Null을 가지고 있어야 한다. -> 여기서 malloc
  // 마지막은 -1로 끝나야 한다.
  long long min_length  = gs->minx; // default: 2 , min 2
  long long max_length = gs->maxx; // default: 5, max 6

  if (*subword_array != NULL) {
    free(*subword_array); // Free previously allocated memory if any
  }
  if (wcslen(word) < min_length) {
    return -1; // Return -1 if the word length is not within the specified range
  }
  long long subword_count = 0;
  long long word_length = wcslen(word);
  long long subword_size = (word_length - min_length + 1) * (max_length - min_length + 1);
  *subword_array = (long long *)malloc(subword_size * sizeof(long long));
  if (*subword_array == NULL) {
    fprintf(stderr, "Memory allocation failed for subword array\n");
    exit(1);
  }
  for (long long length = min_length; length <= max_length; length++) {
    for (long long start = 0; start <= word_length - length; start++) {
      wchar_t subword[MAX_STRING];
      wcsncpy(subword, word + start, length);
      subword[length] = L'\0'; // Null-terminate the subword
      long long subword_index = search_vocab(subword, gs);
      // printf("[DEBUG] Searching subword: %s, result: %lld\n", subword, subword_index);
      if (subword_index != -1) {
        (*subword_array)[subword_count++] = subword_index;
      }
    }
  }
  (*subword_array)[subword_count++] = -1; // Null-terminate the array
  // Resize the array to the actual number of subwords found
  *subword_array = (long long *)realloc(*subword_array, subword_count * sizeof(long long));
  if (*subword_array == NULL) {
    fprintf(stderr, "Memory allocation failed for resized subword array\n");
    exit(1);
  }
  return subword_count; // Return the number of subwords found
}

int search_vocab(wchar_t *word, global_setting *gs) {
  unsigned int hash = get_word_hash(word, gs);
  vocab_word *vocab = gs->vocab;
  int *vocab_hash = gs->vocab_hash;
  long long vocab_hash_size = gs->vocab_hash_size;
  long long too_long = 0;
  while (1) {
    too_long++;
    if (vocab_hash[hash] == -1) return -1;
    if (!wcscmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

int add_word_to_vocab(wchar_t *word, global_setting *gs) {
  unsigned int hash, length = wcslen(word) + 1;
  vocab_word *vocab = gs->vocab;
  long long *vocab_size = &gs->vocab_size;
  long long *vocab_max_size = &gs->vocab_max_size;
  int *vocab_hash = gs->vocab_hash;
  long long vocab_hash_size = gs->vocab_hash_size;


  if (length > MAX_STRING) length = MAX_STRING;

  wcscpy(vocab[*vocab_size].word, word);
  vocab[*vocab_size].cn = 0; // Initialize count to zero
  (*vocab_size)++;
  

  if (*vocab_size + 2 >= *vocab_max_size) {
    printf("debug: vocab size: %lld, max size: %lld\n", *vocab_size, *vocab_max_size);
    *vocab_max_size += 1000;
    // *vocab = (vocab_word *)realloc(*vocab, *vocab_max_size * sizeof(vocab_word));
    vocab = (vocab_word *)realloc(vocab, *vocab_max_size * sizeof(vocab_word));
    gs->vocab = vocab;
  }
  // printf("[INFO] Resizing vocab to max size: %lld\n", *vocab_max_size);
  hash = get_word_hash(word, gs);
  // printf("[INFO] Hash for new word '%s': %u\n", word, hash);
  while (vocab_hash[hash] != -1) {
    hash = (hash + 1) % vocab_hash_size;
  }

  // printf("[INFO] Inserting word '%s' at hash index %u\n", word, hash);
  vocab_hash[hash] = *vocab_size - 1; // Store the index
  
  return *vocab_size - 1; // Return the index of the new word
}

int compare_vocab(const void *a, const void *b) {
  return ((vocab_word *)b)->cn - ((vocab_word *)a)->cn; // Sort by count in descending order
}

void sort_vocab(global_setting *gs) {
  vocab_word *vocab = gs->vocab;
  long long vocab_size = gs->vocab_size;
  int *vocab_hash = gs->vocab_hash;
  long long vocab_hash_size = gs->vocab_hash_size;

  qsort(vocab, vocab_size, sizeof(vocab_word), compare_vocab);
  
  // Rebuild the hash table
  for (int i = 0; i < vocab_hash_size; i++) {
    vocab_hash[i] = -1;
  }
  for (int i = 0; i < vocab_size; i++) {
    unsigned int hash = get_word_hash(vocab[i].word, gs);
    while (vocab_hash[hash] != -1) {
      hash = (hash + 1) % vocab_hash_size; // Linear probing
    }
    vocab_hash[hash] = i;
  }
}

void reduce_vocab(global_setting *gs, long long min_reduce) {
  int a, b = 0;
  unsigned int hash;
  vocab_word *vocab = gs->vocab;
  long long *vocab_size = &gs->vocab_size;
  int *vocab_hash = gs->vocab_hash;
  long long vocab_hash_size = gs->vocab_hash_size;
  long long min_count = gs->min_count;

  for (a = 0; a < *vocab_size; a++) if (vocab[a].cn >= min_reduce) {
    vocab[b].cn = vocab[a].cn;
    for (int c = 0; c < MAX_STRING; c++) {
      vocab[b].word[c] = vocab[a].word[c]; // Copy the word
    }
    // vocab[b].word = vocab[a].word;
    b++;
  } //else free(vocab[a].word);
  *vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < *vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = get_word_hash(vocab[a].word, gs);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  gs->min_count_vocab++;
}



void create_vocab_from_train_file(global_setting *gs) {
  wchar_t word[MAX_STRING];
  wchar_t prev_word[MAX_STRING]; // only support for ngram=2
  wchar_t concat_word[MAX_STRING];
  long long temp_vocab, temp_vocab_index;
  long long debug_mode = gs->debug_mode;
  long long train_words = gs->train_words;
  long long vocab_size = gs->vocab_size;
  long long vocab_hash_size = gs->vocab_hash_size;
  vocab_word *vocab = gs->vocab;
  char *train_file = gs->train_file;



  // printf("[INFO] Creating vocabulary from training file...\n");

  
  FILE *f_in = fopen(train_file, "rb");
  if (f_in == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }

  printf("[INFO] Reading words from training file...\n");

  clock_t search_vocab_time;
  clock_t accum_search_vocab_time = 0;
  clock_t add_word_to_vocab_time;
  clock_t accum_add_word_to_vocab_time = 0;
  long long create_ngram = 0;
  long long prev_word_hash = -1; // for ngram
  while(1) {
    read_word(word, f_in);
    if (feof(f_in)) break;

    train_words++;
    // wprintf(L"\n[DEBUG] Read word: %ls\n", word);
    if ((debug_mode > 1) && (train_words % 1000 == 0)) {
      wprintf(L"%lldK, search_vocab_time: %lld, add_word_to_vocab_time: %lld, vocab_length :%lld%c", train_words / 1000, accum_search_vocab_time, accum_add_word_to_vocab_time, gs->vocab_size, 13);
      accum_search_vocab_time = 0;
      accum_add_word_to_vocab_time = 0;
      fflush(stdout);
    }
    swprintf(concat_word, MAX_STRING, L"<%ls>", word);
    temp_vocab_index = search_vocab(concat_word, gs);
    
    if (temp_vocab_index == -1) {
      if (train_words < 100) {
        wprintf(L"[DEBUG] Adding new word: %ls\n", concat_word);
      }
      temp_vocab_index = add_word_to_vocab(concat_word, gs);
      gs->vocab[temp_vocab_index].cn = 1; // Initialize count to 1
      if (train_words < 100) {
        wprintf(L"[DEBUG] New word added: %ls at index %lld, new: %ls\n", concat_word, temp_vocab_index, gs->vocab[temp_vocab_index].word);
      }
      if (gs->sisg > 0 ) {
        create_subword(concat_word, gs); // Create subwords for the new word
      }

    } else {
      gs->vocab[temp_vocab_index].cn++; // Increment count for existing word
    }
    

    if (gs->vocab_size >= vocab_hash_size * 0.7) {
      wprintf(L"[INFO] Vocabulary reduced. Current size: %lld\n", gs->vocab_size);
      reduce_vocab(gs, gs->min_count_vocab);
      wprintf(L"[INFO] Vocabulary size after reduction: %lld\n", gs->vocab_size);
    }

  }
  reduce_vocab(gs, gs->min_count_vocab);
  sort_vocab(gs);
  gs->total_words = train_words;
  gs->file_size = ftell(f_in);
  fclose(f_in);
}

void save_vocab(global_setting *gs) {
  long long i;
  long long vocab_size = gs->vocab_size;
  char *save_vocab_file = gs->save_vocab_file;
  vocab_word *vocab = gs->vocab;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void read_vocab(global_setting *gs) {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  
  FILE *fin = fopen(gs->read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < gs->vocab_hash_size; a++) gs->vocab_hash[a] = -1;
  gs->vocab_size = 0;
  while (1) {
    read_word(word, fin);
    if (feof(fin)) break;
    if (i < 100) {
      wprintf(L"[DEBUG] Read word: %ls\n", word);
    }
    a = add_word_to_vocab(word, gs);
    fscanf(fin, "%lld%c", &(gs->vocab[a].cn), &c);
    i++;
  }
  sort_vocab(gs);
  if (gs->debug_mode > 0) {
    wprintf(L"Vocab size: %lld\n", gs->vocab_size);
    wprintf(L"Words in train file: %lld\n", gs->train_words);
  }
  fin = fopen(gs->train_file, "rb");
  if (fin == NULL) {
    wprintf(L"ERROR: training data file not found!\n");
    exit(1);
  }
  fseeko(fin, 0, SEEK_END);
  gs->file_size = ftell(fin);
  fclose(fin);
}

// oriignal binary tree is indicate same key value 
void create_binary_tree(vocab_word *_vocab, long long *left_node, long long *right_node, long long size) {
  long long a, b, i, min1i, min2i, pos1, pos2, point[VOCAB_MAX_CODE_LENGTH];
  char code[VOCAB_MAX_CODE_LENGTH];
  long long vocab_size = size;
  vocab_word *vocab = _vocab;
  wprintf(L"[INFO] Allocate memory space for temporar values...\n");
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  pos1 = vocab_size - 1;
  pos2 = vocab_size;

  for (a = 0; a < vocab_size * 2 - 1; a++) {
    left_node[a] = -1;
    right_node[a] = -1;
  }

  wprintf(L"[INFO] Creating binary tree for vocabulary...\n");
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < vocab_size - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    binary[min2i] = 1;

    left_node[vocab_size + a] = min1i;
    right_node[vocab_size + a] = min2i;
  }
  wprintf(L"[INFO] Finished creating binary tree for vocabulary.\n");
  // Now assign binary code to each vocabulary word
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == vocab_size * 2 - 2) break;
    }
    vocab[a].codelen = i;
    vocab[a].point[0] = vocab_size - 2;
    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b];
      vocab[a].point[i - b] = point[b] - vocab_size;
    }
  }
  wprintf(L"[INFO] Binary tree created with %lld nodes.\n", vocab_size * 2 - 1);
  free(count);
  free(binary);
  free(parent_node);
}