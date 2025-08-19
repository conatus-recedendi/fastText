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


def get_subword_average(word, vectors, minn, maxn):
    """
    Get the average vector of subwords for a given word.
    """
    subword_vectors = []
    subword_vectors.append(vectors[word])  # Include the word itself
    for i in range(minn, maxn + 1):
        for j in range(len(word) - i + 1):
            subword = word[j : j + i]
            if subword in vectors:
                subword_vectors.append(vectors[subword])
    if not subword_vectors:
        return np.zeros_like(next(iter(vectors.values())))
    return np.mean(subword_vectors, axis=0)


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
args = parser.parse_args()

vectors = {}
fin = open(args.modelPath, "rb")
for _, line in enumerate(fin):
    try:
        tab = compat_splitting(line)
        if tab is None or len(tab) < 2:
            continue
        vec = np.array(tab[1:], dtype=float)

        word = tab[0]
        word = word.lstrip("<").rstrip(">")
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
    # print("Processing:", tline)
    word1 = tline[0].lower()
    word1 = "<" + word1 + ">"  # Add < and > to the word
    word2 = tline[1].lower()
    word2 = "<" + word2 + ">"  # Add < and > to
    nwords = nwords + 1.0

    if (word1 in vectors) and (word2 in vectors):
        # v1 = vectors[word1]
        v1 = get_subword_average(word1, vectors, args.minn, args.maxn)
        v2 = get_subword_average(word2, vectors, args.minn, args.maxn)
        d = similarity(v1, v2)
        mysim.append(d)
        gold.append(float(tline[2]))
    else:
        drop = drop + 1.0
    # print(
    #     "Processed {0:20s} and {1:20s} with similarity {2:.4f}".format(
    #         word1, word2, d if (word1 in vectors and word2 in vectors) else 0.0
    #     )
    # )
fin.close()

corr = stats.spearmanr(mysim, gold)
dataset = os.path.basename(args.dataPath)
print(
    "{0:20s}: {1:2.0f}  (OOV: {2:2.0f}%)".format(
        dataset, corr[0] * 100, math.ceil(drop / nwords * 100.0)
    )
)
