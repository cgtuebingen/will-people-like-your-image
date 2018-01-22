#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Veronika Kohler
#          Katharina Schwarz
#          Patrick Wieschollek <mail@patwie.com>

"""
Re-Implementation "Will People Like Your Image?" based on TensorPack to support
reproducible multi-gpu training.
"""

import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from arod_provider import LMDBwrapper
from tensorpack import *
import linecache
import re


IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
NUMBER_OF_IMAGES = 199999
NUMBER_OF_TRIPLETS = 10000000


class Triplets(RNGDataFlow):

    def __init__(self, lmdb_path, triplet_path, image_height, image_width, a=0.3, b=0.7):
        self.lmdb = LMDBwrapper(lmdb_path, image_height, image_width)
        self.triplet_path = triplet_path

        self.height = image_height
        self.width = image_width
        self.a = a
        self.b = b

        self.scores = np.zeros(NUMBER_OF_IMAGES)
        for k, img, score in self.lmdb.iter():
            self.scores[int(k)] = score

        # it might be more clever to reduce the search space
        self.asc_sorted_scores = np.argsort(self.scores)

    def size(self):
        return 10000000000000000000

    def get_data(self):

        while True:
            a, p, n = self.get_triplet_ids()

            _, img_a, score_a = self.lmdb.get(a)
            _, img_p, score_p = self.lmdb.get(p)
            _, img_n, score_n = self.lmdb.get(n)

            img_a = cv2.resize(cv2.imdecode(np.asarray(bytearray(img_a), dtype=np.uint8), cv2.IMREAD_COLOR),
                               (self.width, self.height))
            img_p = cv2.resize(cv2.imdecode(np.asarray(bytearray(img_p), dtype=np.uint8), cv2.IMREAD_COLOR),
                               (self.width, self.height))
            img_n = cv2.resize(cv2.imdecode(np.asarray(bytearray(img_n), dtype=np.uint8), cv2.IMREAD_COLOR),
                               (self.width, self.height))

            yield [img_a, float(score_a), img_p, float(score_p), img_n, float(score_n)]

    def get_triplet_ids(self):

        a, p, n = 0, 0, 0
        ids = []
        while not ids:
            line_index = self.rng.choice(a=np.arange(start=1, stop=NUMBER_OF_TRIPLETS))
            line = linecache.getline(self.triplet_path, line_index)
            ids = re.findall(r'\d+', line)
            a, p, n = ids

        return int(a), int(p), int(n)


def show_triplet(a, sa, p, sp, n, sn):
    plot_image = np.concatenate((n, a, p), axis=1)
    plt.xlabel("Negative(" + str(sn) + ")   Original(" + str(sa) + ")   Positiv(" + str(sp) + ")")
    plt.imshow(plot_image)
    plt.xticks([])
    plt.yticks([])
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdbFile', help='lmdbFile')
    parser.add_argument('--tripletFile', help='tripletFile')
    args = parser.parse_args()

    ds = Triplets(args.lmdbFile, args.tripletFile, IMAGE_HEIGHT, IMAGE_WIDTH)
    ds.reset_state()

    for a, sa, p, sp, n, sn in ds.get_data():
        print(sa, sp, sn)
