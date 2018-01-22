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
from tensorpack import *


class ImageDecode(MapDataComponent):

    def __init__(self, ds, dtype=np.uint8, index=0):
        def func(im_data):
            return cv2.imdecode(np.asarray(bytearray(im_data), dtype=dtype), cv2.IMREAD_COLOR)
        super(ImageDecode, self).__init__(ds, func, index=index)


def calc_mean_score(split_line):
    split_line = list(map(int, split_line))
    _sum = sum(split_line[2:12])

    mean = (split_line[2] * 1 +
            split_line[3] * 2 +
            split_line[4] * 3 +
            split_line[5] * 4 +
            split_line[6] * 5 +
            split_line[7] * 6 +
            split_line[8] * 7 +
            split_line[9] * 8 +
            split_line[10] * 9 +
            split_line[11] * 10
            ) / _sum

    return mean


def compare_lmdb_content_with_ava_txt(labels, name, s1, s2=None, s3=None, s4=None,
                                      s5=None, s6=None, s7=None, s8=None, s9=None, s10=None):

    if not (name in labels):
        return False

    l = labels[name]

    if (s2 is not None) and (s1 == l[0] and s2 == l[1] and s3 == l[2] and s4 == l[3] and
                             s5 == l[4] and s6 == l[5] and s7 == l[6] and s8 == l[7] and
                             s9 == l[8] and s10 == l[9]):
        return True

    if (s2 is None) and (l == score):
        return True

    else:
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb', type=str, help='path to lmdb', required=True)
    parser.add_argument('--avaTxt', type=str,
                        help='path to ava.txt', required=True)
    parser.add_argument(
        '--distribution', help='store distribution instead of mean as label', action='store_true')
    args = parser.parse_args()

    ds = LMDBDataPoint(args.lmdb, shuffle=False)
    ds = ImageDecode(ds, index=0)
    ds.reset_state()

    labels = {}
    with open(args.avaTxt, 'r') as f:
        for line in f:
            splitLine = line.split()

            if args.distribution:
                labels[splitLine[1] + '.jpg'] = splitLine[2:12]
            else:
                mean = calc_mean_score(splitLine)
                labels[splitLine[1] + '.jpg'] = mean
    count = 0
    if args.distribution:
        for img, name, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10 in ds.get_data():
            valid = compare_lmdb_content_with_ava_txt(
                labels, name, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10)

            if valid:
                del labels[name]

                count += 1
                if count % 100 == 0:
                    print(count)
            else:
                print("Invalid entry in ava.lmdb" + name)
                exit()

    else:
        for img, name, score in ds.get_data():
            valid = compare_lmdb_content_with_ava_txt(labels, name, score)

            if valid:
                del labels[name]
            else:
                print("Invalid entry in ava.lmdb" + name)
                exit()

    if not labels:
        print("All images are included in the lmdb File.")
    else:
        print("Only " + str(count) + " Files are included.")
        print("Missing Files: " + str(list(labels.keys())))
