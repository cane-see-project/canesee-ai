#import packages
import tensorflow as tf

import numpy as np

path_w='weights/'

class BahdanauAttentionTest(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttentionTest, self).__init__()
    C = tf.keras.initializers.Constant
    w1, w2, w3, w4, w5, w6 = [np.load(path_w+"decoder_layer_weights/layer_%s_%s_weights_%s.npy" %(4, "bahdanau_attention", j)) \
                                  for j in range(6)]
    self.W1 = tf.keras.layers.Dense(units, kernel_initializer=C(w1), bias_initializer=C(w2))
    self.W2 = tf.keras.layers.Dense(units, kernel_initializer=C(w3), bias_initializer=C(w4))
    self.V = tf.keras.layers.Dense(1, kernel_initializer=C(w5), bias_initializer=C(w6))

  def call(self, features, hidden):
    # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

    # hidden shape == (batch_size, hidden_size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    # score shape == (batch_size, 64, hidden_size)
    score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

    # attention_weights shape == (batch_size, 64, 1)
    # you get 1 at the last axis because you are applying score to self.V
    attention_weights = tf.nn.softmax(self.V(score), axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
  
    context_vector = attention_weights * features
    
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights
