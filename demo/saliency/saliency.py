#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Katharina Schwarz
#          Patrick Wieschollek <mail@patwie.com>

"""
Will People Like Your Image?

tested with TensorFlow 1.1.0-rc1 (git rev-parse HEAD 45115c0a985815feef3a97a13d6b082997b38e5d) and OpenCV 3.1.0

EXAMPLE:

    python saliency.py --images "pattern/to/images/*.jpg"
"""

import tensorflow as tf
import cv2
import numpy as np
import cPickle as pickle
import argparse
import glob
from contextlib import contextmanager


@contextmanager
def guided_relu():
    from tensorflow.python.ops import gen_nn_ops   # noqa

    @tf.RegisterGradient("GuidedReLU")
    def GuidedReluGrad(op, grad):
        return tf.where(0. < grad,
                        gen_nn_ops._relu_grad(grad, op.outputs[0]),
                        tf.zeros(grad.get_shape()))

    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': 'GuidedReLU'}):
        yield


def saliency_map(output, input, name="saliency_map"):
    max_outp = tf.reduce_max(output, 1)
    _, h, w, c = input.get_shape().as_list()
    saliency_op = tf.gradients(max_outp, input)[:][0]
    saliency_op = tf.identity(saliency_op, name=name)
    return saliency_op


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', help='pattern for images')
    args = parser.parse_args()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # TODO: not sure, if this "guided_relu" hack works with "tf.train.import_meta_graph"
        with guided_relu():
            img_pl = tf.placeholder(tf.float32, [1, 224, 224, 3])
            new_saver = tf.train.import_meta_graph('../data/ae_model.meta', input_map={"tower_0/image_input:0": img_pl})
            sess.run(tf.global_variables_initializer())
            new_saver.restore(sess, '../data/ae_model')

            # feed_node = tf.get_default_graph().get_tensor_by_name("tower_0/image_input:0")
            embedding = tf.get_default_graph().get_tensor_by_name("tower_0/encodings:0")
            embedding = tf.reshape(embedding, [1, 1000])

            saliency_op = saliency_map(embedding, img_pl)

            with open("../data/mean_bgr.p", "r") as hnd:
                mean_bgr = pickle.load(hnd).transpose(1, 2, 0)

            image_collection = []
            for fn in glob.glob(args.images):

                img = cv2.imread(fn)
                h, w, c = img.shape
                img = cv2.resize(img, (224, 224))
                saliency_images = sess.run(saliency_op, {img_pl: img[None, :, :, :]})[0]
                print saliency_images.shape

                abs_saliency = (1 - np.abs(saliency_images).max(axis=-1))
                abs_saliency -= abs_saliency.min()
                abs_saliency /= abs_saliency.max()
                pos_saliency = (np.maximum(0, saliency_images) / saliency_images.max())
                neg_saliency = (np.maximum(0, -saliency_images) / -saliency_images.min())

                abs_saliency = abs_saliency[:, :]
                pos_saliency = pos_saliency[:, :, [2, 1, 0]]
                neg_saliency = neg_saliency[:, :, [2, 1, 0]]

                abs_saliency *= 255.
                pos_saliency *= 255.
                neg_saliency *= 255.

                abs_saliency = cv2.resize(abs_saliency, (w, h))
                pos_saliency = cv2.resize(pos_saliency, (w, h))
                neg_saliency = cv2.resize(neg_saliency, (w, h))

                # cv2.imwrite(file + "input.jpg", img_O)
                cv2.imwrite(fn + "abs_saliency_resnet.jpg", abs_saliency)
                cv2.imwrite(fn + "pos_saliency_resnet.jpg", pos_saliency)
                cv2.imwrite(fn + "neg_saliency_resnet.jpg", neg_saliency)
