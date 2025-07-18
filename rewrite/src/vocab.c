// 주어진 데이터로부터 vocab만들기

#include <stdio.h>
#include <stdlib.h>
#include <string.h>   
#include "config.h"
#include "vocab.h"


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
void read_word(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) { 
    ch = fgetc(fin);
    if (ch == 13) continue; // Skip carriage return
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

int search_vocab(char *word, global_setting *gs) {
  printf("[INFO] Searching for word: %s\n", word);
  unsigned int hash = get_word_hash(word, gs);
  printf("[INFO] Hash for word '%s': %u\n", word, hash);
  vocab_word *vocab = gs->vocab;
  int *vocab_hash = gs->vocab_hash;
  long long vocab_hash_size = gs->vocab_hash_size;
  printf("[INFO] Starting search in vocab hash table... \n");
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
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


int add_word_to_vocab(char *word, global_setting *gs) {
  unsigned int hash, length = strlen(word) + 1;
  vocab_word **vocab = &gs->vocab;
  long long *vocab_size = &gs->vocab_size;
  long long *vocab_max_size = &gs->vocab_max_size;
  int *vocab_hash = gs->vocab_hash;
  long long vocab_hash_size = gs->vocab_hash_size;

  printf("[INFO] Adding word: %s\n", word);
  if (length > MAX_STRING) length = MAX_STRING;
  printf("[INFO] Current vocab size: %lld, max size: %lld\n", *vocab_size, *vocab_max_size);
  // vocab[*vocab_size]->word = (char *)calloc(length, sizeof(char));
  vocab[*vocab_size] = (vocab_word *)malloc(sizeof(vocab_word));
  if (vocab[*vocab_size] == NULL) {
    fprintf(stderr, "Memory allocation failed for vocab word\n");
    exit(1);
  }
  vocab[*vocab_size]->word = (char *)calloc(length, sizeof(char));

  printf("[INFO] Allocating memory for word: %s %d\n", word, *vocab_size,);

  strcpy(vocab[*vocab_size]->word, word);
  printf("[INFO] Word '%s' added to vocab at index %lld\n", word, *vocab_size);
  vocab[*vocab_size]->cn = 0; // Initialize count to zero
  (*vocab_size)++;
  
  printf("[INFO] Current vocab size: %lld, max size: %lld\n", *vocab_size, *vocab_max_size);
  // Hashing logic here
  if (*vocab_size + 2 >= *vocab_max_size) {
    *vocab_max_size += 1000;
    *vocab = (vocab_word *)realloc(*vocab, *vocab_max_size * sizeof(vocab_word));
  }
  printf("[INFO] Resizing vocab to max size: %lld\n", *vocab_max_size);
  hash = get_word_hash(word, gs);
  printf("[INFO] Hash for new word '%s': %u\n", word, hash);
  while (vocab_hash[hash] != -1) {
    hash = (hash + 1) % vocab_hash_size;
  }

  printf("[INFO] Inserting word '%s' at hash index %u\n", word, hash);
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

void reduce_vocab(global_setting *gs) {
  int a, b = 0;
  long long min_reduce = 1; // Minimum count to keep a word in the vocabulary
  unsigned int hash;
  vocab_word *vocab = gs->vocab;
  long long vocab_size = gs->vocab_size;
  int *vocab_hash = gs->vocab_hash;
  long long vocab_hash_size = gs->vocab_hash_size;
  long long min_count = gs->min_count;

  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else free(vocab[a].word);
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = get_word_hash(vocab[a].word, gs);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}


void create_vocab_from_train_file(global_setting *gs) {
  char word[MAX_STRING];
  long long temp_vocab, temp_vocab_hash;
  long long debug_mode = gs->debug_mode;
  long long train_words = gs->train_words;
  long long vocab_size = gs->vocab_size;
  long long vocab_hash_size = gs->vocab_hash_size;
  vocab_word *vocab = gs->vocab;
  char *train_file = gs->train_file;

  printf("[INFO] Creating vocabulary from training file...\n");

  
  FILE *f_in = fopen(train_file, "rb");
  if (f_in == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }

  printf("[INFO] Reading words from training file...\n");

  while(1) {
    read_word(word, f_in);
    printf("[INFO] Read word: %s\n", word);
    if (feof(f_in)) break;
    train_words++;
    printf("[INFO] Current word count: %lld\n", train_words);
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout); 
    }
    printf("[INFO] Searching for word in vocabulary...\n");
    temp_vocab_hash = search_vocab(word, gs);
    printf("[INFO] Search result: %lld\n", temp_vocab_hash);
    if (temp_vocab_hash == -1) {
      temp_vocab = add_word_to_vocab(word, gs);
      vocab[temp_vocab].cn = 1; // Initialize count to 1
    } else {
      vocab[temp_vocab_hash].cn++; // Increment count for existing word
    }
    printf("[INFO] Added word: %s, count: %lld\n", vocab[temp_vocab].word, vocab[temp_vocab].cn);
    if (vocab_size >= vocab_hash_size * 0.7) {
      reduce_vocab(gs);
    }
  }
  printf("\n[INFO] Finished reading words from training file.\n");
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

void create_binary_tree(global_setting *gs) {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  long long vocab_size = gs->vocab_size;
  vocab_word *vocab = gs->vocab;
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
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
  }
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
  free(count);
  free(binary);
  free(parent_node);
}