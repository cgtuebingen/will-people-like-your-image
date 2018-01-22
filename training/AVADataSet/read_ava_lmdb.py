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
import matplotlib.pyplot as plt

from tensorpack import *


class ImageDecode(MapDataComponent):

    def __init__(self, ds, mode='.jpg', dtype=np.uint8, index=0):
        def func(im_data):
            return cv2.imdecode(np.asarray(bytearray(im_data), dtype=dtype), cv2.IMREAD_COLOR)
        super(ImageDecode, self).__init__(ds, func, index=index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb', type=str, help='path to lmdb', required=True)
    parser.add_argument(
        '--distribution', help='store distribution instead of mean as label', action='store_true')
    args = parser.parse_args()

    ds = LMDBDataPoint(args.lmdb, shuffle=False)
    ds = ImageDecode(ds, index=0)
    ds.reset_state()

    if args.distribution:
        for img, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10 in ds.get_data():
            plot_image = img
            plt.xlabel("Distribution:" + str(s1) + " " + str(s2) + " " + str(s3) + " " +
                       str(s4) + " " + str(s5) + " " + str(s6) + " " + str(s7) +
                       " " + str(s8) + " " + str(s9) + " " + str(s10))
            plt.imshow(cv2.cvtColor(plot_image, cv2.COLOR_BGR2RGB))
            plt.show()

    else:
        for img, score in ds.get_data():
            plot_image = img
            plt.xlabel("Score:" + str(score))
            plt.imshow(cv2.cvtColor(plot_image, cv2.COLOR_BGR2RGB))
            plt.show()
