#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

#define MAX_PAIRS 1000
#define MAX_STRING 200
#define MAX_W 50

const long long max_size = 2000;         // max length of strings
const long long max_w = MAX_W;           // max length of vocabulary entries

// 구조체: 단어쌍과 점수 보관
typedef struct {
    char word1[MAX_STRING];
    char word2[MAX_STRING];
    float gt_score;
    float model_score;
    int gt_rank;
    int model_rank;
} Pair;

void to_upper_english_only(char *str) {
    for (int i = 0; str[i] != '\0' || i < MAX_STRING; i++) {
        // 영어 알파벳 (ASCII)만 대문자로 변환
        if ((unsigned char)str[i] < 128 && isalpha(str[i])) {
            str[i] = toupper((unsigned char)str[i]);
        }
    }
}


// 코사인 유사도
float cosine_similarity(float *vec1, float *vec2, int size) {
    float dot = 0.0, norm1 = 0.0, norm2 = 0.0;
    for (int i = 0; i < size; i++) {
        dot += vec1[i] * vec2[i];
        norm1 += vec1[i] * vec1[i];
        norm2 += vec2[i] * vec2[i];
    }
    return dot / (sqrt(norm1) * sqrt(norm2) + 1e-8);
}

// 비교 함수: 내림차순 (gt)
int compare_gt(const void *a, const void *b) {
    return (((Pair *)b)->gt_score > ((Pair *)a)->gt_score) - (((Pair *)b)->gt_score < ((Pair *)a)->gt_score);
}
// 비교 함수: 내림차순 (model)
int compare_model(const void *a, const void *b) {
    return (((Pair *)b)->model_score > ((Pair *)a)->model_score) - (((Pair *)b)->model_score < ((Pair *)a)->model_score);
}

// 순위 부여
void assign_rank(Pair *pairs, int n, int use_gt) {
    if (use_gt) qsort(pairs, n, sizeof(Pair), compare_gt);
    else qsort(pairs, n, sizeof(Pair), compare_model);

    for (int i = 0; i < n; i++) {
        if (use_gt) pairs[i].gt_rank = i + 1;
        else pairs[i].model_rank = i + 1;
    }
}

// 스피어만 상관계수 계산
float compute_spearman(Pair *pairs, int n) {
    float d_sum = 0.0;
    for (int i = 0; i < n; i++) {
        float d = pairs[i].gt_rank - pairs[i].model_rank;
        d_sum += d * d;
    }
    float rho = 1.0 - (6.0 * d_sum) / (n * (n * n - 1));
    return rho;
}

// 단어 인덱스 찾기
int find_word_index(char *word, char *vocab, long long words) {
    char *vocab_element = (char *)malloc(max_w);

    if (vocab_element == NULL) {
        printf("Memory allocation failed for vocab_element\n");
        return -1;
    }

    
    for (int i = 0; i < words; i++) {
        printf("[DEBUG] Checking word: %s\n", &vocab[i * max_w]);
        memcpy(vocab_element, &vocab[i * max_w], max_w);
        vocab_element[max_w - 1] = '\0'; // Ensure null-termination
        // 대소문자 구분 없이 비교
        printf("[DEBUG] Comparing with: %s\n", vocab_element);
        for (int j = 0; j < max_w && vocab_element[j] != '\0'; j++) {
            vocab_element[j] = toupper(vocab_element[j]);
        }
        if (strcmp(word, vocab_element) == 0) {
            free(vocab_element);
            return i;
        }
    }
    free(vocab_element);
    return -1;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: ./compute-spearman <embeddings.bin> <word_pairs.csv>\n");
        return 1;
    }

    // 모델 로딩
    FILE *f = fopen(argv[1], "rb");
    if (!f) { printf("Cannot open embedding file\n"); return 1; }

    long long words, size;
    fscanf(f, "%lld %lld", &words, &size);
    printf("[INFO] Loaded %lld words with vector size %lld\n", words, size);

    char *vocab = (char *)malloc(words * max_w);
    float *M = (float *)malloc(words * size * sizeof(float));
    for (int i = 0; i < words; i++) {
        fscanf(f, "%s ", &vocab[i * max_w]);
        if (i < 10) {
            printf("[DEBUG] Word %d: %s;\n", i, &vocab[i * max_w]);
        }
        for (int j = 0; j < size; j++) {
            fread(&M[i * size + j], sizeof(float), 1, f);
        }
        float len = 0;
        for (int j = 0; j < size; j++) len += M[i * size + j] * M[i * size + j];
        len = sqrt(len);
        for (int j = 0; j < size; j++) M[i * size + j] /= len;
    }
    fclose(f);

    // CSV 로딩
    f = fopen(argv[2], "r");
    if (!f) { printf("Cannot open CSV file\n"); return 1; }

    Pair pairs[MAX_PAIRS];
    int n = 0;
    while (fscanf(f, "%[^,],%[^,],%f\n", pairs[n].word1, pairs[n].word2, &pairs[n].gt_score) == 3) {
        // to_upper_english_only(pairs[n].word1);
        // to_upper_english_only(pairs[n].word2);
        // 대문자 변환
        for (int k = 0; k < strlen(pairs[n].word1); k++) pairs[n].word1[k] = toupper(pairs[n].word1[k]);
        for (int k = 0; k < strlen(pairs[n].word2); k++) pairs[n].word2[k] = toupper(pairs[n].word2[k]);

        int idx1 = find_word_index(pairs[n].word1, vocab, words);
        int idx2 = find_word_index(pairs[n].word2, vocab, words);
        if (idx1 == -1 || idx2 == -1) {
            printf("Word not found: %s or %s\n", pairs[n].word1, pairs[n].word2);
            continue;
        }
        pairs[n].model_score = cosine_similarity(&M[idx1 * size], &M[idx2 * size], size);
        n++;
    }
    fclose(f);

    // 순위 지정
    assign_rank(pairs, n, 1); // GT 순위
    assign_rank(pairs, n, 0); // Model 순위

    // 스피어만 계산
    float rho = compute_spearman(pairs, n);
    printf("[INFO] Total pairs: %d\n", n);
    printf("[INFO] Spearman correlation: %.4f\n", rho);

    // 디버깅용 출력
    for (int i = 0; i < n; i++) {
        printf("%s,%s | GT: %.2f (Rank %d), Model: %.4f (Rank %d)\n",
            pairs[i].word1, pairs[i].word2,
            pairs[i].gt_score, pairs[i].gt_rank,
            pairs[i].model_score, pairs[i].model_rank);
    }

    free(vocab);
    free(M);
    return 0;
}