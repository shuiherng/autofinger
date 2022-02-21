#!/usr/bin/env python

import cv2
import os
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfc

from PIL import Image, ImageFont, ImageDraw

tfc.disable_eager_execution()


SEMANTIC_MODEL_FOLDER = '../models/Semantic-Model'
SEMANTIC_MODEL_PATH = SEMANTIC_MODEL_FOLDER + '/semantic_model.meta'
SEMANTIC_VOCABULARY_PATH = SEMANTIC_MODEL_FOLDER + '/vocabulary_semantic.txt'


# Utility functions
# Copied from ctc_utils on https://github.com/OMR-Research/tf-end-to-end/blob/master/ctc_utils.py
def normalize(image):
    return (255. - image) / 255.

def resize(image, height):
    width = int(float(height * image.shape[1]) / image.shape[0])
    sample_img = cv2.resize(image, (width, height))
    return sample_img

def sparse_tensors_to_strs(sparse_tensor):
    indices = sparse_tensor[0][0]
    values = sparse_tensor[0][1]
    dense_shape = sparse_tensor[0][2]

    strs = [[] for _ in range(dense_shape[0])]

    string = []
    ptr = 0
    b = 0

    for idx in range(len(indices)):
        if indices[idx][0] != b:
            strs[b] = string
            string = []
            b = indices[idx][0]

        string.append(values[ptr])
        ptr += 1

    strs[b] = string
    return strs

class SemanticGenerator:
    def __init__(self):
        tfc.reset_default_graph()
        self.session = tfc.InteractiveSession()
        self.vocab_list = None
        with open(SEMANTIC_VOCABULARY_PATH, 'r') as vocab_file:
            self.vocab_list = vocab_file.read().splitlines()
        saver = tfc.train.import_meta_graph(SEMANTIC_MODEL_PATH)
        saver.restore(self.session, SEMANTIC_MODEL_PATH[:-5])
        
        graph = tfc.get_default_graph()
        
        self.input = graph.get_tensor_by_name("model_input:0")
        self.seq_len = graph.get_tensor_by_name("seq_lengths:0")
        self.rnn_keep_prob = graph.get_tensor_by_name("keep_prob:0")
        self.height_tensor = graph.get_tensor_by_name("input_height:0")
        self.width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
        self.logits = tfc.get_collection("logits")[0]
        
        # Constants that are saved inside the model itself
        self.WIDTH_REDUCTION, self.HEIGHT = self.session.run([self.width_reduction_tensor, self.height_tensor])
        
        self.decoded, _ = tf.nn.ctc_greedy_decoder(self.logits, self.seq_len)
    
    def map_output(self, vec):
        return [s for s in map(lambda x: self.vocab_list[x], vec)]
    
    def predict(self, img_file):
        image = Image.open(img_file).convert('L')
        image = np.array(image)
        image = resize(image, self.HEIGHT)
        image = normalize(image)
        image = np.asarray(image).reshape(1, image.shape[0], image.shape[1], 1)
        
        seq_lengths = [image.shape[2] / self.WIDTH_REDUCTION]
        
        prediction = self.session.run(self.decoded, feed_dict = {
            self.input: image,
            self.seq_len: seq_lengths,
            self.rnn_keep_prob: 1.0,
        })
        
        # predictions is of shape (1, n) where n is number of predictions
        predictions = sparse_tensors_to_strs(prediction)
        return self.map_output(predictions[0])
    