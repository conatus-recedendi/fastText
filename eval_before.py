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


def compat_splitting(line):
    # split by ,
    return line.decode("utf8").split()


def compat_splitting_by_comma(line):
    # 디코딩 + BOM 제거 + 앞뒤 공백 제거
    line = line.decode("utf8").strip()

    # 1차: 콤마 기준 분리
    parts = line.split(",")

    # 2차: 각 조각을 다시 공백 기준으로 분리 (여러 공백은 하나로 취급)
    tokens = []
    for part in parts:
        tokens.extend(part.strip().split())

    return tokens


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

vectors = {}
fin = open(args.modelPath, "rb")
for _, line in enumerate(fin):
    try:
        tab = compat_splitting(line)
        vec = np.array(tab[1:], dtype=float)
        word = tab[0]
        if np.linalg.norm(vec) == 0:
            continue
        if not word in vectors:
            vectors[word] = vec
    except ValueError:
        continue
    except UnicodeDecodeError:
        continue
fin.close()

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
