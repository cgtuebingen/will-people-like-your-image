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
import h5py
import argparse


def batched_data(generator, length, batch_size):
    """group incoming data into batches of given size

    Args:
        generator: python generator for producing single datapoint
        length: total number of datapoints
        batch_size: size of batch

    Yields:
        batched data of given size, (first batch might be be smaller)
    """
    batches = length // batch_size
    first_batchsize = 0
    if not batches * batch_size == length:
        first_batchsize = length - batches * batch_size

    # get first batch with less entries
    if first_batchsize > 0:
        batch = np.zeros((first_batchsize, 256, 256, 3))
        for i in xrange(first_batchsize):
            batch[i, ...] = next(generator)
        yield batch
    # get all other batches of correct size
    for k in xrange(batches):
        batch = np.zeros((batch_size, 256, 256, 3))
        for i in xrange(batch_size):
            batch[i, ...] = next(generator)
        yield batch


def batched_frames(mp4_file, mean_bgr):
    """Extract batchs of frames

    Args:
        mp4_file (str): path to video file
        mean_bgr (np.array): numpy array of mean image

    Yields:
        batched data of frames
    """
    cap = cv2.VideoCapture(mp4_file)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def generator():
        for i in range(length):
            success, img = cap.read()
            yield img.astype(np.float32)

    for raw in batched_data(generator(), length, 32):
        raw = raw.astype(np.float32)
        m = raw.shape[0]
        batch = np.zeros((m, 224, 224, 3))
        for i in xrange(m):
            batch[i, ...] = cv2.resize(raw[i, ...], (224, 224))
        batch = batch - mean_bgr[None, ...]
        batch = batch[:, :, :, [2, 1, 0]]
        yield batch


def main(mp4_file, score_file):
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        new_saver = tf.train.import_meta_graph('../data/ae_model.meta')
        sess.run(tf.global_variables_initializer())
        new_saver.restore(sess, '../data/ae_model')

        feed_node = tf.get_default_graph().get_tensor_by_name("tower_0/image_input:0")
        embedding = tf.get_default_graph().get_tensor_by_name("tower_0/encodings:0")

        with open("../data/mean_bgr.p", "r") as hnd:
            mean_bgr = pickle.load(hnd).transpose(1, 2, 0)

        scores_collection = []
        for frames in batched_frames(mp4_file, mean_bgr):
            encodings = sess.run(embedding, {feed_node: frames})
            scores = np.linalg.norm(encodings, axis=1)
            scores_collection.extend(scores)
            print encodings.shape

        # write embeddings to file
        with h5py.File(score_file, 'w') as hf:
            g1 = hf.create_group('group1')
            g1.create_dataset('dataset1', data=np.array(scores_collection), dtype='float32')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mp4', help='path to small mp4 file')
    args = parser.parse_args()

    cmd = "avconv -i %s -s 256x256 -strict -2 %s" % (args.mp4, args.mp4.replace('.mp4', '-small.mp4'))
    err_msg = "mp4 file should be already resized to 256x256, run '%s'" % cmd
    assert args.mp4.endswith('-small.mp4'), err_msg
    score_file = args.mp4.replace('-small.mp4', '.mp4.score')
    print score_file
    main(args.mp4, score_file)
