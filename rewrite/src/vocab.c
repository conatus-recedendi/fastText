// 주어진 데이터로부터 vocab만들기

#include <stdio.h>
#include <stdlib.h>
#include <string.h>   
#include "config.h"
#include "vocab.h"
void push_word(char *prev_word, const char *new_word) {
    if (strlen(prev_word) > 0) {
        strcat(prev_word, " ");  // Add space before new word
    }
    strcat(prev_word, new_word);
}

void pop_first_word(char *prev_word) {
    char *space = strchr(prev_word, ' ');
    if (space != NULL) {
        // Shift everything after first space (including space) to beginning
        memmove(prev_word, space + 1, strlen(space + 1) + 1);
    } else {
        // Only one word in prev_word
        prev_word[0] = '\0';
    }
}


int get_label_hash(char *word, global_setting *gs) {
  unsigned long long label_hash_size = gs->label_hash_size;
  vocab_word *labels = gs->labels;
  int *label_hash = gs->label_hash;
  unsigned long long hash = 0;
  for (unsigned long long a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % label_hash_size;
  return hash;
}

int get_word_hash(char *word, global_setting *gs) {
  unsigned long long vocab_hash_size = gs->vocab_hash_size;
  vocab_word *vocab = gs->vocab;
  int *vocab_hash = gs->vocab_hash;
  unsigned long long hash = 0;
  for (unsigned long long a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
long long read_word(char *word, FILE *fin) {
  long long a = 0, ch;
  while (!feof(fin)) { 
    ch = fgetc(fin);
    if (ch == 13) continue; // Skip carriage return
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      continue ;
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return ftell(fin);
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
  return ftell(fin);
}

int search_label(char *word, global_setting *gs) {
  // printf("[INFO] Searching for label: %s\n", word);
  unsigned int hash = get_label_hash(word, gs);
  // printf("[INFO] Hash for label '%s': %u\n", word, hash);
  vocab_word *labels = gs->labels;
  int *label_hash = gs->label_hash;
  long long label_hash_size = gs->label_hash_size;
  // printf("[INFO] Searching in label hash table... label_hash_size: %lld\n", label_hash_size);
  
  while (1) {
    // printf("[INFO] Checking label hash index %u: %d\n", hash, label_hash[hash]);
    // getchar();
    if (label_hash[hash] == -1) return -1; // Not found
    // printf("%s vs %s, %d\n", word, labels[label_hash[hash]].word, strcmp(word, labels[label_hash[hash]].word));
    if (!strcmp(word, labels[label_hash[hash]].word)) {
      return label_hash[hash]; // Found
    }
    hash = (hash + 1) % label_hash_size; // Linear probing
  }
  return -1; // Should not reach here
}

int search_vocab(char *word, global_setting *gs) {
  // printf("[INFO] Searching for word: %s\n", word);
  unsigned int hash = get_word_hash(word, gs);
  // printf("[INFO] Hash for word '%s': %u\n", word, hash);
  vocab_word *vocab = gs->vocab;
  int *vocab_hash = gs->vocab_hash;
  long long vocab_hash_size = gs->vocab_hash_size;
  // printf("[INFO] Searching in vocab hash table... vocab_hash_size: %lld\n", vocab_hash_size);
  // printf("[INFO] Starting search in vocab hash table... \n");
  long long too_long = 0;
  while (1) {
    too_long++;
    // printf("[INFO] Checking hash index %u: %d\n", hash, vocab_hash[hash]);
    // printf("[INFO] Current word: %lld\n", vocab[vocab_hash[hash]].cn);
    if (vocab_hash[hash] == -1) return -1;
    // printf("[INFO] Found hash index %u: %d \"%s\"\n", hash, vocab_hash[hash], vocab[vocab_hash[hash]].word);
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    // if (too_long > 10) {
    //   printf("[INFO] Hash for word '%s' vs '%s': %u %lld\n", vocab[vocab_hash[hash]].word, word, hash, vocab_hash_size);
    // }
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

int read_word_index(FILE *fin, global_setting *gs) {
  char word[MAX_STRING];
  read_word(word, fin);
  if (feof(fin)) return -1;
  return search_vocab(word, gs);
}

int add_label_to_vocab(char *word, global_setting *gs) {
  unsigned int hash, length = strlen(word) + 1;
  vocab_word *labels = gs->labels;
  long long *label_size = &gs->label_size;
  long long *label_max_size = &gs->label_max_size;
  int *label_hash = gs->label_hash;
  long long label_hash_size = gs->label_hash_size;

  if (length > MAX_STRING) length = MAX_STRING;
  
  strcpy(labels[*label_size].word, word);
  labels[*label_size].cn = 0; // Initialize count to zero
  (*label_size)++;
  
  if (*label_size + 2 >= *label_max_size) {
    *label_max_size += 1000;
    labels = (vocab_word *)realloc(labels, *label_max_size * sizeof(vocab_word));
    if (labels == NULL) {
      fprintf(stderr, "Memory allocation failed for labels\n");
      exit(1);
    }
    gs->labels = labels; // Update global setting
  }
  
  hash = get_label_hash(word, gs);
  
  while (label_hash[hash] != -1) {
    hash = (hash + 1) % label_hash_size;
  }
  
  label_hash[hash] = *label_size - 1; // Store the index
  
  return *label_size - 1; // Return the index of the new label
}

int add_word_to_vocab(char *word, global_setting *gs) {
  unsigned int hash, length = strlen(word) + 1;
  vocab_word *vocab = gs->vocab;
  long long *vocab_size = &gs->vocab_size;
  long long *vocab_max_size = &gs->vocab_max_size;
  int *vocab_hash = gs->vocab_hash;
  long long vocab_hash_size = gs->vocab_hash_size;

  // printf("[INFO] Adding word: %s\n", word);
  if (length > MAX_STRING) length = MAX_STRING;
  // printf("[INFO] Current vocab size: %lld, max size: %lld\n", *vocab_size, *vocab_max_size);
  // vocab[*vocab_size]->word = (char *)calloc(length, sizeof(char));
  // vocab[*vocab_size] = (vocab_word *)malloc(sizeof(vocab_word));

  // vocab[*vocab_size].word = (char *)calloc(length, sizeof(char));



  strcpy(vocab[*vocab_size].word, word);
  vocab[*vocab_size].cn = 0; // Initialize count to zero
  (*vocab_size)++;
  
  // printf("[INFO] Current vocab size: %lld, max size: %lld\n", *vocab_size, *vocab_max_size);
  // Hashing logic here
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

void reduce_label(global_setting *gs, long long min_reduce) {
  int a, b = 0;
  unsigned int hash;
  vocab_word *vocab = gs->labels;
  long long *vocab_size = &gs->label_size;
  int *vocab_hash = gs->label_hash;
  long long vocab_hash_size = gs->label_hash_size;
  long long min_count = gs->min_count;

  for (a = 0; a < *vocab_size; a++) if (vocab[a].cn > min_reduce) {
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
  gs->min_count_label++;
}



void create_vocab_from_train_file(global_setting *gs) {
  char word[MAX_STRING];
  char prev_word[MAX_STRING]; // only support for ngram=2
  char concat_word[MAX_STRING];
  long long temp_vocab, temp_vocab_index;
  long long temp_label, temp_label_hash;
  long long debug_mode = gs->debug_mode;
  long long train_words = gs->train_words;
  long long vocab_size = gs->vocab_size;
  long long vocab_hash_size = gs->vocab_hash_size;
  long long label_size = gs->label_size;
  long long label_hash_size = gs->label_hash_size;
  vocab_word *vocab = gs->vocab;
  char *train_file = gs->train_file;



  // printf("[INFO] Creating vocabulary from training file...\n");

  
  FILE *f_in = fopen(train_file, "rb");
  if (f_in == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }

  // printf("[INFO] Reading words from training file...\n");

  clock_t search_label_time;
  clock_t accum_search_label_time = 0;
  clock_t add_label_to_vocab_time;
  clock_t accum_add_label_to_vocab_time = 0;
  clock_t search_vocab_time;
  clock_t accum_search_vocab_time = 0;
  clock_t add_word_to_vocab_time;
  clock_t accum_add_word_to_vocab_time = 0;
  long long create_ngram = 0;
  // char prev_word[MAX_STRING];
  long long prev_word_hash = -1; // for ngram
  while(1) {
    read_word(word, f_in);
    if (feof(f_in)) break;
    if (strncmp(word, "__label__", 9) == 0) {
      prev_word_hash = -1;
      temp_label_hash = search_label(word, gs);
      if (temp_label_hash == -1) {
        temp_label = add_label_to_vocab(word, gs);
        gs->labels[temp_label].cn = 1; // Initialize count to 1
      } else {
        gs->labels[temp_label_hash].cn++; 
      }
      continue;
    }

    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK, search_label_time: %lld, add_label_to_vocab_time: %lld, search_vocab_time: %lld, add_word_to_vocab_time: %lld, vocab_length :%lld, label_length: %lld%c", train_words / 1000, accum_search_label_time, accum_add_label_to_vocab_time, accum_search_vocab_time, accum_add_word_to_vocab_time, gs->vocab_size, gs->label_size, 13);
      accum_search_label_time = 0;
      accum_add_label_to_vocab_time = 0;
      accum_search_vocab_time = 0;
      accum_add_word_to_vocab_time = 0;
      fflush(stdout);
    }
    temp_vocab_index = search_vocab(word, gs);
    if (temp_vocab_index == -1) {
      temp_vocab_index = add_word_to_vocab(word, gs);
      gs->vocab[temp_vocab_index].cn = 1; // Initialize count to
    } else {
      vocab[temp_vocab_index].cn++; // Increment count for existing word
    }
    if (gs->ngram > 1) {
      if (prev_word[0] == 0) {
        strncpy(prev_word, word, sizeof(prev_word) - 1);
      } else {
        memset(concat_word, 0, sizeof(concat_word)); // Reset concat_word
        snprintf(concat_word, MAX_STRING, "%s-%s", prev_word, word);
        long long index = search_vocab(concat_word, gs);
        if (index == -1) {
          index = add_word_to_vocab(concat_word, gs);
          vocab[index].cn = 1; // Initialize count to 1
          create_ngram++;
        } else {
          vocab[index].cn++; // Increment count for existing ngram word
        }
      }
      memset(prev_word, 0, sizeof(prev_word)); // Reset previous word for ngram
      strncpy(prev_word, word, strlen(word)); // Update previous word
      prev_word[MAX_STRING - 1] = '\0'; // Ensure null termination
    }
    if (gs->vocab_size >= vocab_hash_size * 0.7) {
      printf("[INFO] Vocabulary reduced. Current size: %lld\n", gs->vocab_size);
      reduce_vocab(gs, gs->min_count_vocab);
      printf("[INFO] Vocabulary size after reduction: %lld\n", gs->vocab_size);
    }
    if (gs->label_size >= label_hash_size * 0.7) {
      printf("[INFO] Label size: %lld\n", gs->label_size);
      reduce_label(gs, gs->min_count_label);
      printf("[INFO] Label size after reduction: %lld\n", gs->label_size);
    }
  }
  reduce_vocab(gs, gs->min_count_vocab);
  reduce_label(gs, gs->min_count_label);
  sort_vocab(gs);
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
    a = add_word_to_vocab(word, gs);
    fscanf(fin, "%lld%c", &(gs->vocab[a].cn), &c);
    i++;
  }
  sort_vocab(gs);
  if (gs->debug_mode > 0) {
    printf("Vocab size: %lld\n", gs->vocab_size);
    printf("Words in train file: %lld\n", gs->train_words);
  }
  fin = fopen(gs->train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  gs->file_size = ftell(fin);
  fclose(fin);
}

// oriignal binary tree is indicate same key value 
void create_binary_tree(vocab_word *_vocab, long long *left_node, long long *right_node, long long size) {
  long long a, b, i, min1i, min2i, pos1, pos2, point[VOCAB_MAX_CODE_LENGTH];
  char code[VOCAB_MAX_CODE_LENGTH];
  long long vocab_size = size;
  vocab_word *vocab = _vocab;
  printf("[INFO] Allocate memory space for temporar values...\n");
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

  printf("[INFO] Creating binary tree for vocabulary...\n");
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

  for (int i = vocab_size; i< vocab_size * 2 - 1; i++) {
    printf("%lld ", point[i]);
  }
  printf("\n");

  printf("[INFO] Finished creating binary tree for vocabulary. %lld\n", vocab_size * 2 - 1);
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


  printf("[INFO] Binary tree created with %lld nodes.\n", vocab_size * 2 - 1);
  free(count);
  free(binary);
  free(parent_node);
}