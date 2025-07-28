
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <math.h>
#include <ctype.h>
#include "config.h"
#include "vocab.h"

void softmaxf(const float* input, float* output, int size) {
    float max = input[0];
    for (int i = 1; i < size; ++i) {
        if (input[i] > max) {
            max = input[i];
        }
    }

    // expf 계산 + 합 구하기 (overflow 방지 위해 max 빼기)
    // float sum = 0.0f;
    // for (int i = 0; i < size; ++i) {
    //     output[i] = expf(input[i] - max);
    //     sum += output[i];
    // }
    float sum = 0.0f;

    for (int i = 0; i < size; ++i) {
        float z = input[i] - max;
        if (z < SOFTMAX_MIN) z = SOFTMAX_MIN;  // Prevent exp underflow
        output[i] = expf(z);
        sum += output[i];
    }

    if (sum == 0.0f) {
        // 모든 값이 0인 경우, 1로 나누기
        for (int i = 0; i < size; ++i) {
            output[i] = 1.0f / size;
        }
        return;
    }

    // 정규화
    for (int i = 0; i < size; ++i) {
        output[i] /= sum;
    }
    
}

long count_lines(FILE *fp) {
    long lines = 0;
    int c;
    while ((c = fgetc(fp)) != EOF) {
        if (c == '\n') lines++;
    }
    rewind(fp);
    return lines;
}

// 시작 및 끝 오프셋 계산
void compute_thread_offsets(FILE *fp, global_setting *gs) {
    char line[MAX_LINE_LEN];
    long current_line = 0;
    int current_thread = 0;



    long *start_offsets = gs->start_offsets;
    long *end_offsets = gs->end_offsets;
    long *start_line_by_thread = gs->start_line_by_thread;
    long *total_line_by_thread = gs->total_line_by_thread;

    for (int i = 0; i <= gs->num_threads; i++) {
        start_line_by_thread[i] = gs->total_lines * i / gs->num_threads;
        total_line_by_thread[i] = 0;
    }

    // 첫 스레드 시작은 항상 0
    rewind(fp);
  
    start_offsets[0] = 0;
    while (fgets(line, sizeof(line), fp)) {
        current_line++;

        
        // 다음 스레드의 시작 라인에 도달하면 offset 저장
        if (current_thread + 1 <= gs->num_threads &&
            current_line == start_line_by_thread[current_thread + 1]) {
            end_offsets[current_thread] = ftell(fp) - 1;
            start_offsets[current_thread + 1] = ftell(fp);
            total_line_by_thread[current_thread] = current_line - start_line_by_thread[current_thread];

            current_thread++;
        }
    }
    total_line_by_thread[current_thread] = current_line - start_line_by_thread[current_thread];
    end_offsets[current_thread] = ftell(fp);

    // 마지막 스레드의 끝 오프셋은 파일 끝
    fseek(fp, 0, SEEK_END);

}


/**
  * get arg_pos
  * idx = get_arg_pos("--size"), argc, argv)
 */
int get_arg_pos(char *str, int argc, char **argv) {
  int i = 0;

  for (i = 1; i < argc; i++) {
    if (!strcmp(str, argv[i])) {
      if (i == argc - 1) {
        printf("Argument missing for %s\n", str);
        exit(1);
      }
      return i;
    }
  }
  return -1;
}