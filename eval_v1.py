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

f = load_model(args.modelPath)
words, counts = f.get_words(include_freq=True)

vectors = {}
# fin = open(args.modelPath, "rb")
# for _, line in enumerate(fin):
#     try:
#         tab = compat_splitting(line)
#         if tab is None or len(tab) < 2:
#             continue
#         vec = np.array(tab[1:], dtype=float)

#         word = tab[0]
#         word = word.lower()
#         # word = word.lstrip("<").rstrip(">")
#         if np.linalg.norm(vec) == 0:
#             continue
#         if not word in vectors:
#             vectors[word] = vec
#     except ValueError:
#         continue
#     except UnicodeDecodeError:
#         continue
# fin.close()

for w in words[: args.topk]:

    w = w.lower()
    w = normalize_token(w)
    vec = f.get_word_vector(w)
    if np.linalg.norm(vec) == 0:
        continue
    if w not in vectors:
        vectors[w] = vec


subword_vectors = {}
for w in words[: args.topk]:  # subwords is a tuple (list of sub
    # words, list of indices)
    subwords = f.get_subwords(w)[0]
    for sw in subwords:
        sw = sw.lower()
        vec = f.get_word_vector(sw)
        if np.linalg.norm(vec) == 0:
            continue
        if sw not in subword_vectors:
            subword_vectors[sw] = vec

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

mysim = []
mysim_sisg = []
gold = []
gold_sisg = []

drop = 0.0
nwords = 0.0


fin = open(args.dataPath, "rb")
print("Evaluating on data in {0:}".format(args.dataPath))
# len of fin
# print("Total lines:", sum(1 for line in fin))
for line in fin:
    tline = compat_splitting_by_comma(line)
    # show tline infor
    # print("Processing:", tline)
    word1 = tline[0].lower()
    # word1 = "<" + word1 + ">"  # Add < and > to the word
    word2 = tline[1].lower()
    # word2 = "<" + word2 + ">"  # Add < and > to
    nwords = nwords + 1.0

    # print("Comparing words: '{0}' and '{1}'".format(word1, word2))

    if (word1 in vectors) and (word2 in vectors):
        # v1 = vectors[word1]
        # v1 = get_subword_average(word1, vectors, args.minn, args.maxn)
        # v2 = get_subword_average(word2, vectors, args.minn, args.maxn)
        v1 = vectors[word1]
        v2 = vectors[word2]
        d = similarity(v1, v2)
        mysim.append(d)
        gold.append(float(tline[2]))
        mysim_sisg.append(d)
        gold_sisg.append(float(tline[2]))
    elif word1 in vectors and not word2 in vectors:
        drop = drop + 1.0
        if args.sisg:
            # SISG 이면 word1, word2 둘 중 하나가 없음! 이렇게 구하면 dim  안맞음
            # v1 = get_subword_average(word1, vectors, args.minn, args.maxn)

            v1 = vectors[word1]
            v2 = get_subword_average(word2, subword_vectors, args.minn, args.maxn)
            if np.linalg.norm(v2) == 0:
                # as null vector
                continue

            d = similarity(v1, v2)
            # print similairty
            print(
                "Similarity (SISG) between '{0}' and '{1}': {2:.4f}, {3:.4f}".format(
                    word1, word2, d, float(tline[2])
                )
            )
            mysim.append(0)
            gold.append(float(tline[2]))
            mysim_sisg.append(d)
            gold_sisg.append(float(tline[2]))

    elif word2 in vectors and not word1 in vectors:
        drop = drop + 1.0
        if args.sisg:
            # SISG 이면 word1, word2 둘 중 하나가 없음! 이렇게 구하면 dim  안맞음
            v1 = get_subword_average(word1, subword_vectors, args.minn, args.maxn)
            # v2 = get_subword_average(word2, vectors, args.minn, args.maxn)
            v2 = vectors[word2]
            if np.linalg.norm(v1) == 0:
                # as null vector
                continue
            d = similarity(v1, v2)
            print(
                "Similarity (SISG) between '{0}' and '{1}': {2:.4f}, {3:.4f}".format(
                    word1, word2, d, float(tline[2])
                )
            )
            mysim.append(0)
            gold.append(float(tline[2]))
            mysim_sisg.append(d)
            gold_sisg.append(float(tline[2]))
    else:
        drop = drop + 1.0
        if args.sisg:
            v1 = get_subword_average(word1, subword_vectors, args.minn, args.maxn)
            v2 = get_subword_average(word2, subword_vectors, args.minn, args.maxn)
            if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
                # as null vector
                continue
            d = similarity(v1, v2)

            print(
                "Similarity (SISG) between '{0}' and '{1}': {2:.4f}, {3:.4f}".format(
                    word1, word2, d, float(tline[2])
                )
            )
            mysim.append(0)
            gold.append(float(tline[2]))
            mysim_sisg.append(d)
            gold_sisg.append(float(tline[2]))

fin.close()

corr = stats.spearmanr(mysim, gold)
dataset = os.path.basename(args.dataPath)

# this is for sisg-
print(
    "{0:20s}: {1:2.3f}  (OOV: {2:2.0f}%)".format(
        dataset, corr[0] * 100, math.ceil(drop / nwords * 100.0)
    )
)


# this sis for sisg+
if args.sisg:
    corr_sisg = stats.spearmanr(mysim_sisg, gold_sisg)
    print(
        "{0:20s} (SISG): {1:2.0f}  (OOV: {2:2.0f}%)".format(
            dataset, corr_sisg[0] * 100, math.ceil(drop / nwords * 100.0)
        )
    )
