#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Veronika Kohler
#          Katharina Schwarz
#          Patrick Wieschollek <mail@patwie.com>

"""
Re-Implementation "Will People Like Your Image?" based on TensorPack to support
reproducible multi-gpu training.
"""

import tensorflow as tf
import argparse
import os
import numpy as np
from tensorflow.python import pywrap_tensorflow

# conversion rules from the names in the npz file to the ones used in the own training.
transformationRules = {"bn": 'BatchNorm', 
                       "mean/EMA": "moving_mean",
                       "variance/EMA": "moving_variance",
                       "/W": "/weights"}


def print_variables_from_stored_model(graph_path):
    """Prints the names of the tensors stored in a tensorflow model.
        graph_path: path to the stored model.
    """
    reader = pywrap_tensorflow.NewCheckpointReader(graph_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print("tensor_name: ", key)


def recursive_file_gen(search_dir):
    """Finds all files under a directory including the ones in sub directories.
        search_dir: directory which should be searched
    """
    for root, dirs, files in os.walk(search_dir):
        for file in files:
            yield os.path.join(root, file)


def get_tensors_from_one_graph_as_map(graph_path):
    """Returns all tensors contained in stored graph.
        graph_path: Path to the stored graph.
    """
    saver = tf.train.import_meta_graph(graph_path)
    graph = tf.get_default_graph()

    operations = []
    for op in tf.get_default_graph().get_operations():
        operations.append(op)

    tensors = {}

    for op in operations:
        tensors[op.name] = op

    return tensors


def get_variable_name(f, path_length):
    """Returns the pure name of a tensor stored in a file in the npz structure.
        Therefore delete the prefix of the path and file extension and apply the above defined transformation rules.
        graph_path: Path to the stored graph.
        f: Complete path to file.
        path_length: Length of the path before the actual file name.
    """
    name = f[path_length + 1:-6]

    for t in transformationRules:
        name = name.replace(t, transformationRules[t])

    return name


def variables_dictionary_from_npz_file(npz):
    """Returns a dictionary containing all variables/tensors stored in the npz structure.
        Therefore delete the prefix of the path and file extension and apply the above defined transformation rules.
        npz: Path to the npz structure.
    """
    path_length = len(npz)
    files = list(recursive_file_gen(npz))

    variables = {}
    for f in files:
        variable_name = get_variable_name(f, path_length)
        variables[variable_name] = np.load(f)

    return variables


def key_contained_in_map(_key, _map):
    """Checks whether a key is contained in a map.
    """
    if _key in _map:
        return True
    else:
        return False


def create_model_from_npz_file(npz, model, target):
    """Creates a tensorflow model from a given npz structure in which the variables for the desired model are stored.
        npz: Path to the npz structure containing files representing the variables in the model.
        model: Path in which the final model should be stored
        target: A target model which contains the desired names for the structure
    """
    reader = pywrap_tensorflow.NewCheckpointReader(target)
    target_map = reader.get_variable_to_shape_map()

    variables = variables_dictionary_from_npz_file(npz)
    i = 0
    for key in variables:

        if key_contained_in_map(key, target_map):
            name = 'var' + str(i)
            val = tf.Variable(variables[key], name=key)
            exec(name + " = val")
            i += 1

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        save_path = saver.save(sess, model)
        print("Model saved in file: %s" % save_path)


def compare_created_model_with_target(target, model):
    """Compares the names of the variables of a given model and a target model.
        target: Metagraph of target.
        model: Metagraph of model.
    """
    model_tensors = get_tensors_from_one_graph_as_map(model+".meta")
    target_tensors = get_tensors_from_one_graph_as_map(target)

    for key in list(target_tensors):
        if 'Adam' not in key:
            if key in model_tensors:
                if target_tensors[key]._outputs[0].shape == model_tensors[key]._outputs[0].shape:
                    continue
                else:
                    print(key + " shape doesn't match.")
            else:
                print(key + " missing in created model.")

    for key in list(model_tensors):
        if key in target_tensors:
            if target_tensors[key]._outputs[0].shape == model_tensors[key]._outputs[0].shape:
                continue
            else:
                print(key + " shape doesn't match.")
        else:
            print(key + " not needed in target model.")

    return


def check_adam(model):
    """Checks whether a unwanted variable of the adam optimizer is still contained in a model.
    """
    reader = pywrap_tensorflow.NewCheckpointReader(model)
    target_map = reader.get_variable_to_shape_map()

    for key in list(target_map):
        if 'Adam_1' in key:
            print(key + "contains Adam.")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', help='path to .npz folder')
    parser.add_argument('--model', help='location of the created model')
    parser.add_argument('--target', help='location of the target model for comparison')
    args = parser.parse_args()

    create_model_from_npz_file(args.npz, args.model, args.target)
    target_metaGraph = "/home/vroni/SS17/Forschungsprojekt/Data/train-log/resnet50_for_embedding:" \
                       "resNetTriplet34/graph-1023-111405.meta"
    compare_created_model_with_target(target_metaGraph, args.model)
    print_variables_from_stored_model(args.target)
