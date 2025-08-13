
#ifndef H_VOCAB_WORD_H
#define H_VOCAB_WORD_H

  #define VOCAB_MAX_STRING 400
  #define VOCAB_MAX_CODE_LENGTH 40
typedef struct {
  long long cn; // Count of word occurrences
  char codelen; // Word string, Huffman code, and its length
  char word[VOCAB_MAX_STRING]; // Word string
  char code[VOCAB_MAX_CODE_LENGTH]; // Huffman code
  char point[VOCAB_MAX_CODE_LENGTH + 1]; // Points to the nodes in the Huffman tree
} vocab_word;
#endif