
#include "config.h"
#include "vocab.h"
#ifndef UTILS_H
#define UTILS_H

typedef struct {
  long long id; // thread id
  global_setting *gs; // global settings
} thread_args;

int isNewLine(const char *word);
int isClass(const char *word);
int isWord(const char *word);
void softmaxf(const float* input, float* output, int size);
long count_lines(FILE *fp);
void compute_thread_offsets(FILE *fp, global_setting *gs);    
int get_arg_pos(char *str, int argc, char **argv);

#endif