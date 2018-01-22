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
import msgpack_numpy

import matplotlib.pyplot as plt
from tensorpack import *

msgpack_numpy.patch()
NUMBER_OF_IMAGES = 255508


# ease the use of LMDB
class LMDBwrapper(object):
    def __init__(self, fn, image_height, image_width, distribution=False):
        logger.info("Open {}".format(fn))
        self.height = image_height
        self.width = image_width
        self.distribution = distribution

        super(LMDBwrapper, self).__init__()

        keys = [str(i) for i in range(NUMBER_OF_IMAGES)]

        self.obj = LMDBData(fn, shuffle=True, keys=keys)

    def get(self, idx):
        k = self.obj.keys[idx]
        v = self.obj._txn.get(k.encode())
        # v is (img, score) pair packes with msgpack
        v = msgpack.loads(v)
        if self.distribution:
            img, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10 = v
            return k, img, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10
        else:
            img, score = v
            return k, img, score

    def iter(self):
        c = self.obj._txn.cursor()
        while c.next():
            k, v = c.item()
            if k != b'__keys__':
                v = msgpack.loads(v)

        if self.distribution:
            img, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10 = v
            yield k, img, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10
        else:
            img, score = v
            yield k, img, score


class ImageDecode(MapDataComponent):
    def __init__(self, ds, mode='.jpg', dtype=np.uint8, index=0):
        def func(im_data):
            img = cv2.imdecode(np.asarray(bytearray(im_data), dtype=dtype), cv2.IMREAD_COLOR)
            return img
        super(ImageDecode, self).__init__(ds, func, index=index)


class LabeledImage(RNGDataFlow):
    def __init__(self, lmdb_path, image_height, image_width, distribution=False):
        self.lmdb = LMDBwrapper(lmdb_path, image_height, image_width, distribution)
        self.height = image_height
        self.width = image_width
        self.distribution = distribution

    def size(self):
        return 10000000000000000000

    def get_data(self):
        while True:
            img_id = self.rng.choice(NUMBER_OF_IMAGES)

            if self.distribution:
                _, img, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10 = self.lmdb.get(img_id)
            else:
                _, img, score = self.lmdb.get(img_id)

            img = cv2.imdecode(np.asarray(bytearray(img), dtype=np.uint8), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (self.width, self.height))

            if self.distribution:
                yield img, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10
            else:
                yield img, float(score)


def debug(lmdb):
    ds = LMDBDataPoint(lmdb, shuffle=False)
    ds = ImageDecode(ds, index=0)
    ds.reset_state()
    for img, score in ds.get_data():
        print(img.shape, score)


def show_image(img, score):
    plot_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.xlabel("Score: " + score)
    plt.imshow(plot_image)
    plt.xticks([])
    plt.yticks([])
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb', type=str, help='path to lmdb', required=True)
    parser.add_argument('--distribution', help='does the given lmdb file contain distribution '
                                               'or a mean value of the ava score.', action='store_true')
    args = parser.parse_args()

    ds = LabeledImage(args.lmdb, 224, 224, args.distribution)
    ds.reset_state()

    if args.distribution:
        for img, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10 in ds.get_data():
            show_image(img, str(s1) + " " + str(s2) + " " + str(s3) + " " + str(s4) + " " + str(s5) +
                       " " + str(s6) + " " + str(s7) + " " + str(s8) + " " + str(s9) + " " + str(s10))

    else:
        for img, score in ds.get_data():
            show_image(img, str(score))
