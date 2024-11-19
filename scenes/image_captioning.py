import image_captioning_interface
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
import os
import cv2

def init():
    global model
    model = image_captioning_interface.image_captioning(mobile_net_v2_weights='mobilenet_v2_weights_1.4.h5')

def generate_caption(np_RGB_image):
    return model.generate_from_img_nparray_encode(np_RGB_image)

if __name__ == '__main__':
    init()
    while(1):
        test_image = input("enter path:")
        im = Image.open(test_image)
        im = np.array(im)
        print(generate_caption(im))