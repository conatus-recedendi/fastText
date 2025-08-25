#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from fasttext import load_model
import numpy as np
from scipy import stats
import os
import math
import argparse


import re

_WS_RE = re.compile(r"\s+")


def normalize_token(token: str) -> str:
    """쉘의 normalize_text와 최대한 유사하게 단일 '단어 키'를 만들기 위한 정규화.
    - 소문자화
    - 특수 따옴표/큰따옴표 등 치환
    - 여러 기호(#, @, «, ♯, 전각 콜론 등) → 공백
    - 숫자 → 공백
    - 공백 압축/트림
    - 공백이 생겨 다단어가 되면 '첫 토큰'만 사용
    """
    if not token:
        return ""

    s = token.lower()

    # 따옴표류 통일
    s = s.replace("’", "'").replace("′", "'").replace("“", '"').replace("”", '"')

    # 쉘 sed에서 공백으로 치환하던 기호들
    for ch in ["«", "♯", "#", "@", "：", ",", "،", "=", "*", "|", "»", "ː"]:
        s = s.replace(ch, " ")

    # 문장부호 주변에 공백을 넣는 대신, 단어 키를 만들 목적이므로
    # 여기서는 그냥 제거/공백화만 수행(토큰 키가 쪼개지지 않도록)
    for ch in [".", "(", ")", "!", "?", "-", '"', "'"]:
        s = s.replace(ch, " ")

    # 숫자 → 공백 (sed 의 `tr 0-9 " "` 대응)
    s = re.sub(r"[0-9]", "", s)

    # 공백 압축 및 트림
    s = _WS_RE.sub(" ", s).strip()

    # 다단어로 쪼개졌다면 첫 단어만 사용(키 충돌 방지 목적)
    if " " in s:
        s = s.split(" ", 1)[0]

    return s


def get_nearest_vector(vector, vectors, topk, golden_word, args):
    # find nearest vector in vectors
    nearest = None
    min_dist = float("inf")
    for word, vec in vectors.items():
        # if np.array_equal(vector, vec):
        #     return word

        # print vector, vec size
        # print(word, vec)
        # print("vector size:", vector.shape, "vec size:", vec.shape)
        d = vector - vec
        dist = np.dot(d, d)
        if dist < min_dist:
            min_dist = dist
            nearest = word

    if args.sisg:
        golden_word = golden_word.lower()
        if golden_word not in vectors:
            golden_subword_vec = get_subword_average(
                golden_word, vectors, args.minn, args.maxn
            )
            if min_dist > np.dot(
                vector - golden_subword_vec, vector - golden_subword_vec
            ):
                nearest = golden_word

    return nearest


def get_subword_average(word, vectors, minn, maxn):
    """
    Get the average vector of subwords for a given word.
    """
    word = "<" + word + ">"  # Add < and > to the word
    subword_vectors = []
    if word in vectors:
        subword_vectors.append(vectors[word])  # Include the word itself
    for i in range(minn, maxn + 1):
        for j in range(len(word) - i + 1):
            subword = word[j : j + i]
            # print("word:", word, "i:", i, "j:", j)
            # print("subword:", subword)
            if subword in vectors:
                subword_vectors.append(vectors[subword])
    if not subword_vectors:
        return np.zeros_like(next(iter(vectors.values())))
    return np.mean(subword_vectors, axis=0)
    # return np.sum(subword_vectors, axis=0)


def compat_splitting(line):
    # split by ,
    return line.decode("utf8").split()


def compat_splitting_by_comma(line):
    # split by comma
    return line.decode("utf8").strip().split(",")


def similarity(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    return np.dot(v1, v2) / n1 / n2


parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument(
    "--model",
    "-m",
    dest="modelPath",
    action="store",
    required=True,
    help="path to model",
)

parser.add_argument(
    "--minn",
    "-mn",
    dest="minn",
    action="store",
    type=int,
    default=3,
    help="minimum length of subword",
)

parser.add_argument(
    "--maxn",
    "-mx",
    dest="maxn",
    action="store",
    type=int,
    default=6,
    help="maximum length of subword",
)

parser.add_argument(
    "--data", "-d", dest="dataPath", action="store", required=True, help="path to data"
)
parser.add_argument(
    "--sisg",
    "-sisg",
    dest="sisg",
    action="store_true",
    help="use SISG (Subword Information and Similarity Graph) for evaluation",
)
parser.add_argument(
    "--topk",
    "-k",
    dest="topk",
    action="store",
    type=int,
    default=300000,
)
args = parser.parse_args()

vectors = {}
fin = open(args.modelPath, "rb")

f = load_model(args.modelPath)
words, counts = f.get_words(include_freq=True)


# for _, line in enumerate(fin):
#     try:
#         tab = compat_splitting(line)
#         if tab is None or len(tab) < 2:
#             continue

#         if len(tab) != 301:
#             continue
#         vec = np.array(tab[1:], dtype=float)

#         word = tab[0]
#         word = word.lower()
#         word = normalize_token(word)
#         # word = word.lstrip("<").rstrip(">")
#         if np.linalg.norm(vec) == 0:
#             continue
#         if not word in vectors:
#             vectors[word] = vec
#     except ValueError:
#         continue
#     except UnicodeDecodeError:
#         continue

for w in words[: args.topk]:
    print(w)

    w = w.lower()
    w = normalize_token(w)
    vec = f.get_word_vector(w)
    if np.linalg.norm(vec) == 0:
        continue
    if w not in vectors:
        vectors[w] = vec


print("Loaded {0:} words.".format(len(vectors)))
words = []
W_list = []
for w, v in vectors.items():
    words.append(w)
    W_list.append(v.astype(np.float32))
W = np.vstack(W_list)  # (N, D)
# 이미 위에서 정규화해두면 생략 가능
W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-12
idx_of = {w: i for i, w in enumerate(words)}  # ← 추가
fin.close()


semantic = []
syntactic = []

drop = 0.0
nwords = 0.0


fin = open(args.dataPath, "rb")
print("Evaluating on data in {0:}".format(args.dataPath))
# len of fin

flag = "semantic"
for line in fin:

    tline = compat_splitting(line)
    if tline[0].startswith(":"):
        # 이 다음 줄부터는 syntactic
        if tline[1].startswith("gram"):
            flag = "syntactic"
        continue

    word1 = tline[0].lower()
    word2 = tline[1].lower()
    word3 = tline[2].lower()
    word4 = tline[3].lower()

    nwords = nwords + 1.0

    # word
    # 1 - 2 = 3 - 4
    # 4 = 3 - 2 + 1
    # find nearest 3 - 2 + 1

    if (word1 in vectors) and (word2 in vectors) and (word3 in vectors):
        v1 = vectors[word1]
        v2 = vectors[word2]
        v3 = vectors[word3]
        # v1 - v2 = v3 - q
        q = v3 + v2 - v1
        q /= np.linalg.norm(q) + 1e-12
        best_idx = -1
        best_sim = -1.0
        B = 100_000
        mask = np.zeros(W.shape[0], dtype=bool)
        for ww in (word1, word2, word3):
            i = idx_of.get(ww, None)
            if i is not None:
                mask[i] = True

        sim = W @ q  # (N,)
        sim[mask] = -np.inf  # 제외 적용
        best_idx = int(np.argmax(sim))
        nearest = words[best_idx]
        # for i in range(0, W.shape[0], B):
        #     sim = W[i : i + B] @ q  # (B,)
        #     j = np.argmax(sim)
        #     if sim[j] > best_sim and words[i + j] not in (word1, word2, word3):
        #         best_sim = float(sim[j])
        #         best_idx = i + j
        nearest = words[best_idx]
        if nearest == word4:
            d = 1.0
        else:
            d = 0.0
        if flag == "semantic":
            semantic.append(d)
        else:
            syntactic.append(d)
        print(
            "\r Semantic Accuarcy: {0:.2f}%, Syntactic Accuracy: {1:.2f}%, Analogy: '{2}' is to '{3}' as '{4}' is to '{5}' (predicted: '{6}')".format(
                np.mean(semantic) * 100,
                np.mean(syntactic) * 100,
                word1,
                word2,
                word3,
                word4,
                nearest,
            ),
            end="",
            flush=True,
        )


print("Semantic accuracy: {0:.2f} %".format(np.mean(semantic) * 100))
print("Syntactic accuracy: {0:.2f} %".format(np.mean(syntactic) * 100))
print("Total accuracy: {0:.2f} %".format(np.mean(semantic + syntactic) * 100))
