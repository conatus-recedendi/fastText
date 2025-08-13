#include <wchar.h>
#include "vocab_word.h"
#ifndef VOCAB_H
#define VOCAB_H


int search_vocab(wchar_t *word, global_setting *gs);
int read_word_index(FILE *fin, global_setting *gs);
int add_word_to_vocab(wchar_t *word, global_setting *gs);
void sort_vocab(global_setting *gs);
void reduce_vocab(global_setting *gs, long long min_reduce);
void create_vocab_from_train_file(global_setting *gs);
void create_binary_tree(vocab_word *vocab, long long *left_node, long long *right_node, long long size);
void read_vocab(global_setting *gs);
void save_vocab(global_setting *gs);
long long search_subword(wchar_t *word, global_setting *gs, long long **subword_array);
void compute_thread_offsets_subword(FILE *fp, global_setting *gs);
long count_lines_subword(FILE *fp);


#endif

