#include "vocab_word.h"
#ifndef VOCAB_H
#define VOCAB_H


int get_word_hash(char *word, global_setting *gs);
long long read_word(char *word, FILE *fin);
long long read_buffer(char *word, char *bulk, long long start_idx);
int search_vocab(char *word, global_setting *gs);
int read_word_index(FILE *fin, global_setting *gs);
int add_word_to_vocab(char *word, global_setting *gs);
void sort_vocab(global_setting *gs);
void reduce_vocab(global_setting *gs, long long min_reduce);
void reduce_label(global_setting *gs, long long min_reduce);
void create_vocab_from_train_file(global_setting *gs);
void create_binary_tree(vocab_word *vocab, long long *left_node, long long *right_node, long long size);
void read_vocab(global_setting *gs);
void save_vocab(global_setting *gs);
int add_label_to_vocab(char *word, global_setting *gs);
int get_label_hash(char *word, global_setting *gs);
int search_label(char *word, global_setting *gs);
long long search_subword(char *word, global_setting *gs, long long **subword_array);


#endif

