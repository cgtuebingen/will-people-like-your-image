#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Veronika Kohler
#          Katharina Schwarz
#          Patrick Wieschollek <mail@patwie.com>

"""
Re-Implementation "Will People Like Your Image?" based on TensorPack to support
reproducible multi-gpu training.
"""


import numpy as np
import os
import multiprocessing
import argparse
import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorpack import InputDesc, ModelDesc, logger, QueueInput
from tensorpack.models import *
from tensorpack.callbacks import *
from tensorpack.dataflow import BatchData,PrefetchDataZMQ
from tensorpack.tfutils import SaverRestore

from tensorpack.train import TrainConfig, SyncMultiGPUTrainer
import tensorpack.tfutils.symbolic_functions as symbf
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils import argscope, get_model_loader
from tensorpack.utils.gpu import get_nr_gpu

import sys
sys.path.append('../ArodProcessing')
import arod_dataflow_from_txt
import arod_provider

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
BATCH_SIZE = 32


def bn_with_gamma(x):
        return slim.layers.batch_norm(inputs=x, scale=True)


def resnet_shortcut(shortcut, features, stride):

    in_channel = shortcut.get_shape().as_list()[3]

    if in_channel != features:  # features == input channels???
        shortcut = slim.layers.conv2d(inputs=shortcut,
                                      num_outputs=features,
                                      kernel_size=1,
                                      stride=stride,
                                      padding='same',
                                      normalizer_fn=slim.layers.batch_norm,
                                      activation_fn=None,
                                      scope='convshortcut')
    return shortcut


def resnet_block(x, features, stride):

    shortcut = resnet_shortcut(x, features * 4, stride)
    x = slim.layers.conv2d(inputs=x,
                           num_outputs=features,
                           kernel_size=1,
                           stride=1,
                           padding='same',
                           normalizer_fn=slim.layers.batch_norm,
                           scope='conv1')

    x = slim.layers.conv2d(inputs=x,
                           num_outputs=features,
                           kernel_size=3,
                           stride=stride,
                           padding='same',
                           normalizer_fn=slim.layers.batch_norm,
                           scope='conv2')

    x = slim.layers.conv2d(inputs=x,
                           num_outputs=features * 4,
                           kernel_size=1,
                           stride=1,
                           padding='same',
                           normalizer_fn=bn_with_gamma,
                           activation_fn=None,
                           scope='conv3')

    return x + shortcut


def resnet_group(x, name, features, stride, blocks):
    with tf.variable_scope(name):
        for i in range(0, blocks):
            with tf.variable_scope('block{}'.format(i)):
                x = resnet_block(x, features, stride if i == 0 else 1)
                x = tf.nn.relu(x)
    return x


class ResNet(ModelDesc):

    # creates DataFlow Object and bundels the result in batches
    @staticmethod
    def get_data(lmdb_path, txt_path):

        if txt_path:
            ds = arod_dataflow_from_txt.Triplets(lmdb_path, txt_path, IMAGE_HEIGHT, IMAGE_WIDTH)
        else:
            ds = arod_provider.Triplets(lmdb_path, IMAGE_HEIGHT, IMAGE_WIDTH)

        ds.reset_state()
        cpu = min(10, multiprocessing.cpu_count())
        ds = PrefetchDataZMQ(ds, cpu)
        ds = BatchData(ds, BATCH_SIZE)
        return ds

    def embed(self, x, nfeatures=1024):

        if isinstance(x, list):
            x = tf.concat(x, 0)

        # 1st Layer
        x = slim.layers.conv2d(x, kernel_size=7, stride=2, num_outputs=64,
                               normalizer_fn=slim.layers.batch_norm, scope='conv0')
        x = slim.layers.max_pool2d(x, kernel_size=3, stride=2, scope='pool0')

        # Residual Blocks
        x = resnet_group(x=x, features=64, stride=1, name='group0', blocks=3)
        x = resnet_group(x=x, features=128, stride=2, name='group1', blocks=4)
        x = resnet_group(x=x, features=256, stride=2, name='group2', blocks=6)
        x = resnet_group(x=x, features=512, stride=2, name='group3', blocks=3)

        x = tf.reduce_mean(x, [1, 2])

        # 8th Layer: FC and return unscaled activations
        embedding = slim.layers.fully_connected(x, nfeatures, activation_fn=None, scope='embedding')

        return embedding

    # loss function to optimize
    # the embedding of one triplet (emb)
    # margin=0.5 -> horizon for negative examples
    # extra=true -> also return distance for pos and neg
    def loss(self, emb, sa, sn):

        unit_emb = tf.nn.l2_normalize(emb, 1)

        a, p, n = tf.split(emb, 3)
        ua, up, un = tf.split(unit_emb, 3)

        triplet_cost, dist_pos, dist_neg = symbf.triplet_loss(ua, up, un, margin=0.5, extra=True, scope="loss")

        direction_cost = tf.reduce_mean(tf.maximum(0., tf.sign(sa - sn) * tf.transpose(a - n) + 0.2))

        return triplet_cost+direction_cost, dist_pos, dist_neg

    def _get_inputs(self):
        return [InputDesc(tf.float32, (BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, 3), 'input_a'),
                InputDesc(tf.float32, (BATCH_SIZE,), 'score_a'),
                InputDesc(tf.float32, (BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, 3), 'input_p'),
                InputDesc(tf.float32, (BATCH_SIZE,), 'score_p'),
                InputDesc(tf.float32, (BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, 3), 'input_n'),
                InputDesc(tf.float32, (BATCH_SIZE,), 'score_n')]

    def _build_graph(self, inputs):

        a, sa, p, sp, n, sn = inputs
        emb = self.embed([a, p, n])

        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            tf.identity(self.embed(inputs[0]), name="emb")

        cost, pos_dist, neg_dist = self.loss(emb, sa, sn)
        self.cost = tf.identity(cost, name="cost")

        add_moving_summary(pos_dist, neg_dist, self.cost)

    def _get_optimizer(self):
        return tf.train.AdamOptimizer(learning_rate=0.1)


def get_config(lmdb_path, txt_path, nr_gpu):

    df = ResNet.get_data(lmdb_path, txt_path)

    return TrainConfig(
        model=ResNet(),
        dataflow=df,
        callbacks=[
            ModelSaver()
        ],
        extra_callbacks=[
            MovingAverageSummary(),
            ProgressBar(["EMA/cost", "EMA/loss/pos-dist", "EMA/loss/neg-dist"]),
            MergeAllSummaries(),
            RunUpdateOps()
        ],
        nr_tower=nr_gpu,
        steps_per_epoch=1800,
        max_epoch=30,
        session_config=tf.ConfigProto(allow_soft_placement=True)
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--lmdb', help='lmdb path of the arod dataset.')
    parser.add_argument('--txt', help='txt with predefined triplets.')
    parser.add_argument('--load', help='load checkpoint file with pretrained models.')
    args = parser.parse_args()
    logger.auto_set_dir(name="resNetTriplet50_1024Features")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    NR_GPU = len(args.gpu.split(','))

    config = get_config(args.lmdb, args.txt, NR_GPU)

    if args.load:
        config.session_init = SaverRestore(args.load)
    SyncMultiGPUTrainer(config).train()
