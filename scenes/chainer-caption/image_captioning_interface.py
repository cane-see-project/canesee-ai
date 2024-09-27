#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Sample code to generate caption using beam search
'''
import sys
import json
import os
import cv2
# comment out the below if you want to do type check. Remeber this have to be done BEFORE import chainer
#os.environ["CHAINER_TYPE_CHECK"] = "0"
import chainer 
from PIL import Image
import argparse
import numpy as np
import math
from chainer import cuda
import chainer.functions as F
from chainer import cuda, Function, gradient_check, Variable, optimizers
from chainer import serializers

sys.path.append('./code')
from CaptionGenerator import CaptionGenerator

chainer.global_config.autotune = True
chainer.global_config.enable_backprop = False
chainer.global_config.type_check = False

class image_captioning ():
    def __init__(self,
    	rnn_model_place='./data/caption_en_model40.model',
    	cnn_model_place='./data/ResNet50.model',
    	dictonary_place='./data/MSCOCO/mscoco_caption_train2014_processed_dic.json',
    	beamsize=3,
    	depth_limit=50,
    	gpu_id=-1,
    	first_word="<sos>"):

    	self.caption_generator=CaptionGenerator(
	    	rnn_model_place=rnn_model_place,
	    	cnn_model_place=cnn_model_place,
	    	dictonary_place=dictonary_place,
	    	beamsize=beamsize,
	    	depth_limit=depth_limit,
	    	gpu_id=gpu_id,
	    	first_word= first_word,
    	)

    def generate_from_img_nparray(self,image_array):
        image_feature=self.caption_generator.cnn_model(image_array, "feature").data.reshape(1,1,2048)
        return self.caption_generator.generate_from_img_feature(image_feature)

    def resize (self,image_array):
        image_array = cv2.resize(image_array, dsize=(224, 224))
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        pixels = np.asarray(image_array).astype(np.float32)
        pixels = pixels[:,:,::-1].transpose(2,0,1)
        pixels -= self.caption_generator.image_loader.mean_image
        return pixels.reshape((1,) + pixels.shape)


    def resize_img_and_generate(self,image_array):
        return self.generate_from_img_nparray(self.resize(image_array))

#run this file for test purpose
if __name__ == '__main__':
    test_image_path = "test3.jpg"
    image_captioning=image_captioning()
    #test generate_from_img_nparray
    # np_image = image_captioning.caption_generator.image_loader.load(test_image_path)
    # captions = image_captioning.generate_from_img_nparray(np_image)
    # for caption in captions:
    #     print (" ".join(caption["sentence"]))
    #     print (caption["log_likelihood"])

    #test resize_img_and_generate
    image = cv2.imread(test_image_path)
    captions = image_captioning.resize_img_and_generate(image)
    for caption in captions:
        print (" ".join(caption["sentence"]))
        print (caption["log_likelihood"])