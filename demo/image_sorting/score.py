#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Katharina Schwarz
#          Patrick Wieschollek <mail@patwie.com>

"""
Will People Like Your Image?

tested with TensorFlow 1.1.0-rc1 (git rev-parse HEAD 45115c0a985815feef3a97a13d6b082997b38e5d) and OpenCV 3.1.0
"""

import tensorflow as tf
import cv2
import numpy as np
import cPickle as pickle
import argparse
import glob


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', help='pattern for images')
    args = parser.parse_args()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        new_saver = tf.train.import_meta_graph('../data/ae_model.meta')
        sess.run(tf.global_variables_initializer())
        new_saver.restore(sess, '../data/ae_model')

        feed_node = tf.get_default_graph().get_tensor_by_name("tower_0/image_input:0")
        embedding = tf.get_default_graph().get_tensor_by_name("tower_0/encodings:0")

        with open("../data/mean_bgr.p", "r") as hnd:
            mean_bgr = pickle.load(hnd).transpose(1, 2, 0)

        image_collection = []
        for fn in glob.glob(args.images):
            img = cv2.imread(fn)
            img = cv2.resize(img, (224, 224))
            encodings = sess.run(embedding, {feed_node: img[None, :, :, :]})
            scores = np.linalg.norm(encodings, axis=1)

            print fn, scores
