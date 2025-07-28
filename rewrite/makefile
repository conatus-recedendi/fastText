# 디렉토리 설정
SRC_DIR     := src
INCLUDE_DIR := include
BUILD_DIR   := build
BIN_DIR     := bin

# 컴파일러 설정
CC      := gcc
CFLAGS  := -O3 -march=native -Wall -funroll-loops -Wno-unused-result -lm -pthread
INC     := -I$(INCLUDE_DIR)

# 공통 소스 및 오브젝트 (⚠️ 이걸 먼저 정의)
COMMON_SRCS := $(SRC_DIR)/utils.c $(SRC_DIR)/vocab.c
COMMON_OBJS := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(COMMON_SRCS))

# 실행 파일 목록 (여기만 고치면 됨)
TARGETS := fastText test

# main 파일 매핑 (타겟명 -> main 파일)
define MAIN_SRC
$(SRC_DIR)/$(1).c
endef

# main 오브젝트 매핑
define MAIN_OBJ
$(BUILD_DIR)/$(1).o
endef

# 각 타겟의 바이너리 경로
BINARIES := $(addprefix $(BIN_DIR)/,$(TARGETS))

# 기본 빌드 타겟
all: $(BINARIES)

# 실행 파일 빌드 규칙 (이제는 COMMON_OBJS가 정의된 상태)
$(BIN_DIR)/%: $(BUILD_DIR)/%.o $(COMMON_OBJS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -o $@

# 전체 오브젝트 파일 빌드 규칙
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) -c $< -o $@ $(CFLAGS) $(INC)

# 정리
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)
