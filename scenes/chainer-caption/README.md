# image caption generation by chainer on raspberrypi
original repo https://github.com/apple2373/chainer-caption

## descrtiption

This repository contains an implementation of typical image caption generation based on neural network (i.e. CNN + RNN). The model first extracts the image feature by CNN and then generates captions by RNN. CNN is ResNet50 and RNN is a standard LSTM .
This repo is an edited version of https://github.com/yoshihiroo/programming-workshop/tree/master/image_captioning_and_speech


**What is new?**

- upgrade to chainer V7.1
- provide interface that accepts a `numpy array` and return the generated caption
- edit chainer 7 global configurations to make code run faster

## requirements
- python 3.x
- chainer V2 or higher  http://chainer.org
  

## citation:
If you find this implementation useful, please consider to cite: 
```
@article{multilingual-caption-arxiv,
title={{Using Artificial Tokens to Control Languages for Multilingual Image Caption Generation}},
author={Satoshi Tsutsui, David Crandall},
journal={arXiv:1706.06275},
year={2017}
}

@inproceedings{multilingual-caption,
author={Satoshi Tsutsui, David Crandall},
booktitle = {CVPR Language and Vision Workshop},
title = {{Using Artificial Tokens to Control Languages for Multilingual Image Caption Generation}},
year = {2017}
}
```

Install Programs and Tools
-------
Install required programs and tools by commands below.
```
sudo apt-get install python3-pip
sudo pip3 install chainer
sudo pip3 install scipy
sudo pip3 install h5py
sudo apt-get install python-h5py
sudo apt-get install libopenjp2-7-dev
sudo apt-get install libtiff5
sudo pip3 install Pillow
sudo apt-get install espeak
sudo apt-get install libatlas-base-dev
```

**Download CNN & RNN models run**
```
bash download.sh
```


**How to use** `image_captioning_interface.py`:

just impot `image_captioning_interface` in your code and make an object of `image_captioning` class.

This class `image_captioning` has 2 functions:
1. The initialization function
```python
def __init__(self,
    	rnn_model_place='./data/caption_en_model40.model',
    	cnn_model_place='./data/ResNet50.model',
    	dictonary_place='./data/MSCOCO/mscoco_caption_train2014_processed_dic.json',
    	beamsize=3,
    	depth_limit=50,
    	gpu_id=-1,
    	first_word="<sos>")
```
2. caption generation function
```python
def generate_from_img_nparray(self,image_array)
```
this function accepts `RGB` image in an nparry with size of `224 x 224`
and returns the generated caption

