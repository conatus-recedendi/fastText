// bulk_word_reader.h
#pragma once
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifndef MAX_STRING
#define MAX_STRING 1024
#endif

typedef struct {
    FILE *f;
    unsigned char *buf;
    size_t cap;     // buffer capacity
    size_t i;       // current index into buf
    size_t n;       // valid bytes in buf
    long long file_pos; // logical ftell-equivalent (bytes consumed from FILE)
    int eof;        // EOF reached on underlying FILE
    int pending_eos; // need to emit </s> at next call (due to pushback '\n')
} BufReader;

static inline void br_init(BufReader *br, FILE *f, size_t cap) {
    br->f = f;
    br->buf = (unsigned char*)malloc(cap);
    br->cap = cap;
    br->i = br->n = 0;
    br->file_pos = 0;
    br->eof = 0;
    br->pending_eos = 0;
}

static inline void br_free(BufReader *br) {
    free(br->buf);
    br->buf = NULL;
    br->cap = br->i = br->n = 0;
}

static inline size_t br_fill(BufReader *br) {
    if (br->eof) return 0;
    br->i = 0;
    br->n = fread(br->buf, 1, br->cap, br->f);
    if (br->n == 0) {
        br->eof = 1;
    }
    return br->n;
}

// Peek next byte without consuming. Returns -1 on EOF.
static inline int br_peek(BufReader *br) {
    if (br->i >= br->n) {
        if (!br_fill(br)) return -1;
    }
    return br->buf[br->i];
}

// Get next byte and consume. Returns -1 on EOF.
static inline int br_getc(BufReader *br) {
    if (br->i >= br->n) {
        if (!br_fill(br)) return -1;
    }
    int c = br->buf[br->i++];
    br->file_pos++;
    return c;
}

// Unget 1 byte previously taken with br_getc (only supports single-step pushback).
static inline void br_ungetc(BufReader *br) {
    if (br->i > 0) {
        br->i--;
        br->file_pos--;
    } else {
        // If buffer is empty at i==0 and we need a pushback across fill(),
        // a full-featured ring-buffer is needed. For our use (only pushing back '\n'
        // we just saw), this path won't occur because we only push back immediately.
        // Could be extended if required.
    }
}

/**
 * Buffered read_word:
 * - Boundaries: ' ', '\t', '\n'
 * - CR(13) is skipped
 * - If newline occurs with no accumulated chars, returns "</s>"
 * - If newline ends a word, it is "pushed back" so the next call will return "</s>"
 * - Truncates tokens longer than MAX_STRING-1
 * - Returns ftell-like byte position (bytes consumed from start of file) after this call
 *   (same 의미로 사용 가능)
 */
static inline long long br_read_word(char *word, BufReader *br) {
    long long a = 0;
    int ch;

    // If we deferred a newline from previous call, emit </s> now and consume the '\n'.
    if (br->pending_eos) {
        // Consume exactly one '\n' if present; if file ended, we still emit </s>
        int pk = br_peek(br);
        if (pk == '\n') (void)br_getc(br);
        br->pending_eos = 0;
        strcpy(word, "</s>");
        return br->file_pos;
    }

    // Main scan
    for (;;) {
        ch = br_getc(br);
        if (ch == -1) break;           // EOF

        if (ch == 13) continue;        // skip CR

        if (ch == ' ' || ch == '\t' || ch == '\n') {
            if (a > 0) {
                if (ch == '\n') {
                    // push back NL so next call returns </s>
                    br_ungetc(br);
                    br->pending_eos = 1;
                }
                break; // end of current word
            } else {
                if (ch == '\n') {
                    // newline alone => return </s> immediately (already consumed)
                    strcpy(word, "</s>");
                    return br->file_pos;
                } else {
                    // skip leading spaces/tabs
                    continue;
                }
            }
        }

        if (a < MAX_STRING - 1) {
            word[a++] = (char)ch;
        } else {
            // truncate too long words: keep reading but don't append
            // (same as original a-- trick, but clearer)
            // just ignore additional chars until boundary
        }
    }

    // finalize token
    word[a] = '\0';
    return br->file_pos;
}