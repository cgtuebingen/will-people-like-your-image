#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Veronika Kohler
#          Katharina Schwarz
#          Patrick Wieschollek <mail@patwie.com>

"""
Re-Implementation "Will People Like Your Image?" based on TensorPack to support
reproducible multi-gpu training.
"""

from arod_provider import Triplets
import argparse

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

NUMBER_OF_TRIPLETS = 10000000


def create_text_file(data_path, output_file):
    ds = Triplets(data_path, IMAGE_HEIGHT, IMAGE_WIDTH)
    ds.reset_state()
    percentage = 0.0

    with open(output_file, 'w') as file:
        t = 0.0
        while t < NUMBER_OF_TRIPLETS:
            ids_generator = ds.get_one_triplet()
            ids = next(ids_generator)
            if ids:
                file.write(' '.join(str(i) for i in ids))
                file.write('\n')
                t += 1.0
                if int((t / NUMBER_OF_TRIPLETS) * 100) > percentage:
                    percentage = int((t / NUMBER_OF_TRIPLETS) * 100)
                    print(str(percentage) + " %")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='lmdb Data File.')
    parser.add_argument('--outFile', help='Output File.')
    args = parser.parse_args()

    create_text_file(args.data, args.outFile)
