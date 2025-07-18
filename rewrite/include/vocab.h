#include "vocab_word.h"
#ifndef VOCAB_H
#define VOCAB_H


int get_word_hash(char *word, global_setting *gs);
long long read_word(char *word, FILE *fin);
int search_vocab(char *word, global_setting *gs);
int read_word_index(FILE *fin, global_setting *gs);
int add_word_to_vocab(char *word, global_setting *gs);
void sort_vocab(global_setting *gs);
void reduce_vocab(global_setting *gs);
void create_vocab_from_train_file(global_setting *gs);
void create_binary_tree(global_setting *gs);
void read_vocab(global_setting *gs);
void save_vocab(global_setting *gs);

#endif

