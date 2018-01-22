#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Veronika Kohler
#          Katharina Schwarz
#          Patrick Wieschollek <mail@patwie.com>

"""
Re-Implementation "Will People Like Your Image?" based on TensorPack to support
reproducible multi-gpu training.
"""

from tensorpack.dataflow import *
import os
import numpy as np
import argparse


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


class AvaDataFlow(DataFlow):

    def __init__(self, images, ava_txt, distribution=False):

        self.progressedImages = 1
        self.lastProgress = 0.0

        self.pathToImages = images
        self.imageNames = os.listdir(self.pathToImages)

        self.labels = {}

        with open(ava_txt, 'r') as f:
            for line in f:
                split_line = line.split()

                if distribution:
                    self.labels[split_line[1] + '.jpg'] = split_line[2:12]
                else:
                    mean = calc_mean_score(split_line)
                    self.labels[split_line[1] + '.jpg'] = mean

    def get_data(self):

        for img in self.imageNames:

            with open(self.pathToImages + "/" + img, 'rb') as f:
                jpeg = f.read()
            jpeg = np.asarray(bytearray(jpeg), dtype='uint8')

            if img in self.labels:
                label = self.labels[img]

            else:
                print("No label for image: " + img)
                continue

            self.print_state()

            if type(label) is list:
                yield[jpeg, label[0], label[1], label[2], label[3], label[4],
                      label[5], label[6], label[7], label[8], label[9]]
            else:
                yield [jpeg, label]

    def print_state(self):

        percentage = round(
            (self.progressedImages / len(self.imageNames)) * 100, 1)

        if percentage >= self.lastProgress + 0.1:
            print(str(percentage) + "%")
            self.lastProgress = percentage

        self.progressedImages += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb', type=str, help='path to lmdb', required=True)
    parser.add_argument('--images', type=str,
                        help='path to images', required=True)
    parser.add_argument('--avaTxt', type=str,
                        help='path to AVA.txt', required=True)
    parser.add_argument(
        '--distribution', help='store distribution instead of mean as label', action='store_true')
    args = parser.parse_args()

    if os.path.isfile(args.lmdb):
        os.remove(args.lmdb)

    if args.distribution:
        ds0 = AvaDataFlow(args.images, args.avaTxt, True)

    else:
        ds0 = AvaDataFlow(args.images, args.avaTxt)

    dftools.dump_dataflow_to_lmdb(ds0, args.lmdb)
