#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Veronika Kohler
#          Katharina Schwarz
#          Patrick Wieschollek <mail@patwie.com>

"""
Re-Implementation "Will People Like Your Image?" based on TensorPack to support
reproducible multi-gpu training.
"""


import cv2
import argparse
import numpy as np
import msgpack
import msgpack_numpy  # noqa
import math  # noqa

import matplotlib.pyplot as plt
from tensorpack import *

NUMBER_OF_IMAGES = 199999


class LMDBwrapper(object):
    def __init__(self, fn, image_height, image_width):
        logger.info("Open {}".format(fn))
        self.height = image_height
        self.width = image_width

        super(LMDBwrapper, self).__init__()

        keys = [str(i) for i in range(NUMBER_OF_IMAGES)]

        self.obj = LMDBData(fn, shuffle=True, keys=keys)

        if NUMBER_OF_IMAGES != self.obj._size:
            print("WARNING: NUMBER_OF_IMAGES doesen't match size of lmdb file")

        k = [int(k) for k in self.obj.keys]

    def get(self, idx):
        if idx > NUMBER_OF_IMAGES:
            print("WARNING: Index is to large for number of images.")

        k = self.obj.keys[idx]
        v = self.obj._txn.get(k.encode())
        # v is (img, score) pair packes with msgpack
        v = msgpack.loads(v)
        img, score = v
        return k, img, score

    def iter(self):
        c = self.obj._txn.cursor()
        while c.next():
            k, v = c.item()
            if k != b'__keys__':
                v = msgpack.loads(v)
                img, score = v
                yield k, img, score

    def get_test_data(self, count):

        images = []
        scores = []

        c = self.obj._txn.cursor()
        i = 0
        while i <= int(count):
            c.next()
            k, v = c.item()

            if k != b'__keys__':
                a = np.random.randint(NUMBER_OF_IMAGES)
                _, img, score = self.get(a)

                img = cv2.imdecode(np.asarray(bytearray(img), dtype=np.uint8), cv2.IMREAD_COLOR)
                img = cv2.resize(img, (self.width, self.height))
                images.append(img)
                scores.append(score)
                i += 1

        return images, scores


class ImageDecode(MapDataComponent):
    def __init__(self, ds, mode='.jpg', dtype=np.uint8, index=0):
        def func(im_data):
            img = cv2.imdecode(np.asarray(bytearray(im_data), dtype=dtype), cv2.IMREAD_COLOR)
            return img
        super(ImageDecode, self).__init__(ds, func, index=index)


class Triplets(RNGDataFlow):
    def __init__(self, lmdb_path, image_height, image_width, a=0.3, b=0.7):
        self.lmdb = LMDBwrapper(lmdb_path, image_height, image_width)
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

    def valid_triplet(self, a, p, n):
        ap = np.abs(a - p)
        an = np.abs(a - n)
        pn = np.abs(p - n)

        if an == 0:
            an = 0.00000000001
        if pn == 0:
            pn = 0.00000000001

        if self.a > ap / an:
            return False
        if self.b < ap / an:
            return False

        if self.a > ap / pn:
            return False
        if self.b < ap / pn:
            return False

        return True

    def get_data(self):
        while True:
            a, p = self.rng.choice(NUMBER_OF_IMAGES, size=2, replace=False)
            sa = self.scores[a]
            sp = self.scores[p]

            for guess in range(50):
                n = self.rng.randint(NUMBER_OF_IMAGES)
                sn = self.scores[n]

                if self.valid_triplet(sa, sp, sn):
                    _, img_a, score_a = self.lmdb.get(a)
                    _, img_p, score_p = self.lmdb.get(p)
                    _, img_n, score_n = self.lmdb.get(n)

                    img_a = cv2.imdecode(np.asarray(bytearray(img_a), dtype=np.uint8), cv2.IMREAD_COLOR)
                    img_p = cv2.imdecode(np.asarray(bytearray(img_p), dtype=np.uint8), cv2.IMREAD_COLOR)
                    img_n = cv2.imdecode(np.asarray(bytearray(img_n), dtype=np.uint8), cv2.IMREAD_COLOR)

                    img_a = cv2.resize(img_a, (self.width, self.height))
                    img_p = cv2.resize(img_p, (self.width, self.height))
                    img_n = cv2.resize(img_n, (self.width, self.height))

                    yield [img_a, float(score_a), img_p, float(score_p), img_n, float(score_n)]
                    break

    def get_one_triplet(self):

        validTriplet = False
        while(validTriplet is False):
            a, p = self.rng.choice(NUMBER_OF_IMAGES, size=2, replace=False)
            sa = self.scores[a]
            sp = self.scores[p]

            for guess in range(50):
                n = self.rng.randint(NUMBER_OF_IMAGES)
                sn = self.scores[n]

                if self.valid_triplet(sa, sp, sn):
                    yield [a, p, n]
                    validTriplet = True
                    break


def debug(lmdb):
    ds = LMDBDataPoint(lmdb, shuffle=False)
    ds = ImageDecode(ds, index=0)
    ds.reset_state()
    for img, score in ds.get_data():
        print (img.shape, score)


def show_triplet(a, sa, p, sp, n, sn):
    plot_image = np.concatenate((cv2.cvtColor(n, cv2.COLOR_BGR2RGB),
                                 cv2.cvtColor(a, cv2.COLOR_BGR2RGB),
                                 cv2.cvtColor(p, cv2.COLOR_BGR2RGB)), axis=1)
    plt.xlabel("Negative(" + str(sn) + ")   Original(" + str(sa) + ")   Positiv(" + str(sp) + ")")
    plt.imshow(plot_image)
    plt.xticks([])
    plt.yticks([])
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb', type=str, help='path to lmdb', required=True)
    args = parser.parse_args()

    ds = Triplets(args.lmdb, 224, 224)
    ds.reset_state()

    for a, sa, p, sp, n, sn in ds.get_data():
        show_triplet(a, sa, p, sp, n, sn)
