
#ifndef H_VOCAB_WORD_H
#define H_VOCAB_WORD_H

typedef struct {
  long long cn; // Count of word occurrences
  int *point; // Points to the nodes in the Huffman tree
  char *word, *code, codelen; // Word string, Huffman code, and its length
} vocab_word;
#endif