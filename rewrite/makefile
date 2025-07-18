# 디렉토리 경로 설정
SRC_DIR := src
INCLUDE_DIR := include
BUILD_DIR := build
BIN_DIR := bin

# 컴파일러 및 플래그
CC := gcc
CFLAGS := -O3 -march=native -Wall -funroll-loops -Wno-unused-result -lm -pthread
INC_FLAGS := -I$(INCLUDE_DIR)

# 소스 파일 및 오브젝트 파일
SRCS := $(wildcard $(SRC_DIR)/*.c) $(wildcard $(SRC_DIR)/vocab/*.c)
OBJS := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRCS))
OBJS := $(patsubst $(VOCAB_DIR)/%.c,$(BUILD_DIR)/vocab/%.o,$(OBJS))

# 실행 파일
TARGET := $(BIN_DIR)/fastText

# 기본 타겟
all: $(TARGET)

# 실행 파일 빌드
$(TARGET): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(CC) $^ -o $@ $(CFLAGS)

# .o 파일 빌드 (src/ 디렉토리 소스용)
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) -c $< -o $@ $(CFLAGS) $(INC_FLAGS)

# .o 파일 빌드 (src/vocab/ 디렉토리 소스용)
$(BUILD_DIR)/vocab/%.o: $(VOCAB_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) -c $< -o $@ $(CFLAGS) $(INC_FLAGS)

# 정리
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)
