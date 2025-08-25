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


def get_nearest_vector(vector, vectors, topk, golden_word, args):
    # find nearest vector in vectors
    nearest = None
    min_dist = float("inf")
    for word, vec in vectors.items():
        if np.array_equal(vector, vec):
            return word
        dist = np.linalg.norm(vector - vec)
        if dist < min_dist:
            min_dist = dist
            nearest = word

    if args.sisg:
        golden_word = golden_word.lower()
        if golden_word not in vectors:
            golden_subword_vec = get_subword_average(
                golden_word, vectors, args.minn, args.maxn
            )
            if min_dist > np.linalg.norm(vector - golden_subword_vec):
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
        word = word.lower()
        # word = word.lstrip("<").rstrip(">")
        if np.linalg.norm(vec) == 0:
            continue
        if not word in vectors:
            vectors[word] = vec
    except ValueError:
        continue
    except UnicodeDecodeError:
        continue
fin.close()

print("Loaded {0:} words.".format(len(vectors)))

semantic = []
syntactic = []

drop = 0.0
nwords = 0.0


fin = open(args.dataPath, "r")
print("Evaluating on data in {0:}".format(args.dataPath))
# len of fin

flag = "semantic"
for line in fin:
    if line.startswith(": gram"):
        # 이 다음 줄부터는 syntactic
        flag = "syntactic"
        continue
    if (line.startswith(": ")) or (len(line.strip()) == 0):
        continue

    tline = compat_splitting_by_comma(line)

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
        nearest_word = get_nearest_vector(v3 - v2 + v1, vectors, 1, word4, args)
        if nearest_word == word4:
            d = 1.0
        else:
            d = 0.0
        if flag == "semantic":
            semantic.append(d)
        else:
            syntactic.append(d)


print("Semantic accuracy: {0:.2f} %".format(np.mean(semantic) * 100))
print("Syntactic accuracy: {0:.2f} %".format(np.mean(syntactic) * 100))
print("Total accuracy: {0:.2f} %".format(np.mean(semantic + syntactic) * 100))
