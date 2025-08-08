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
import numpy as np
from scipy import stats
import os
import math
import argparse


def load_c_style_vectors(filename):
    vectors = {}

    with open(filename, "rb") as f:
        # 1. 헤더 읽기
        header = f.readline()
        words, size = map(int, header.strip().split())
        print(f"[INFO] Loaded {words} words with vector size {size}")

        for i in range(words):
            # 2. 단어 읽기
            word_bytes = []
            while True:
                ch = f.read(1)
                if ch == b" ":
                    break
                if ch != b"\n":
                    word_bytes.append(ch)
            word = b"".join(word_bytes).decode("utf-8", errors="replace")

            # 3. 벡터 읽기 (float32)
            vec = np.frombuffer(f.read(4 * size), dtype=np.float32)

            # 4. 정규화
            norm = np.linalg.norm(vec)
            if norm != 0:
                vec = vec / norm

            # 5. 딕셔너리에 저장 (list 형태로)
            vectors[word] = vec.tolist()

            if i < 5:  # 디버그용
                print(f"[DEBUG] {word} → {vectors[word][:5]}...")

    return vectors


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
    "--data", "-d", dest="dataPath", action="store", required=True, help="path to data"
)
args = parser.parse_args()

# vectors = {}
vectors = load_c_style_vectors(args.modelPath)
# fin = open(args.modelPath, "rb")
# fin.readline()  # Skip header line
# for _, line in enumerate(fin):
#     try:
#         tab = compat_splitting(line)
#         print(tab)
#         vec = np.array(tab[1:], dtype=float)
#         word = tab[0]
#         if np.linalg.norm(vec) == 0:
#             continue
#         if not word in vectors:
#             vectors[word] = vec
#     except ValueError:
#         continue
#     except UnicodeDecodeError:
#         continue
# fin.close()

mysim = []
gold = []
drop = 0.0
nwords = 0.0

fin = open(args.dataPath, "rb")
for line in fin:
    tline = compat_splitting_by_comma(line)
    # show tline infor
    print("Processing:", tline)
    word1 = tline[0].lower()
    word2 = tline[1].lower()
    nwords = nwords + 1.0

    if (word1 in vectors) and (word2 in vectors):
        v1 = vectors[word1]
        v2 = vectors[word2]
        d = similarity(v1, v2)
        mysim.append(d)
        gold.append(float(tline[2]))
    else:
        drop = drop + 1.0
fin.close()

corr = stats.spearmanr(mysim, gold)
dataset = os.path.basename(args.dataPath)
print(
    "{0:20s}: {1:2.0f}  (OOV: {2:2.0f}%)".format(
        dataset, corr[0] * 100, math.ceil(drop / nwords * 100.0)
    )
)
