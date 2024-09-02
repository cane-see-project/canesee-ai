path_w="weights/"
#import packages
import tensorflow as tf

import numpy as np
from BahdanauAttentionTest import BahdanauAttentionTest

class RNN_DecoderTest(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(RNN_DecoderTest, self).__init__()
    self.units = units

    C = tf.keras.initializers.Constant
    w_emb = np.load(path_w+"decoder_layer_weights/layer_%s_%s_weights_%s.npy" %(0, "embedding", 0))
    w_gru_1, w_gru_2, w_gru_3 = [np.load(path_w+"decoder_layer_weights/layer_%s_%s_weights_%s.npy" %(1, "gru", j)) for j in range(3)]
    w1, w2 = [np.load(path_w+"decoder_layer_weights/layer_%s_%s_weights_%s.npy" %(2, "dense_1", j)) for j in range(2)]
    w3, w4 = [np.load(path_w+"decoder_layer_weights/layer_%s_%s_weights_%s.npy" %(3, "dense_2", j)) for j in range(2)]

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, embeddings_initializer=C(w_emb))
    self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   kernel_initializer=C(w_gru_1),
                                   recurrent_initializer=C(w_gru_2),
                                   bias_initializer=C(w_gru_3)
                                   )
    self.fc1 = tf.keras.layers.Dense(self.units, kernel_initializer=C(w1), bias_initializer=C(w2))
    self.fc2 = tf.keras.layers.Dense(vocab_size, kernel_initializer=C(w3), bias_initializer=C(w4))

    self.attention = BahdanauAttentionTest(self.units)

  def call(self, x, features, hidden):
    # defining attention as a separate model
    context_vector, attention_weights = self.attention(features, hidden)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # shape == (batch_size, max_length, hidden_size)
    x = self.fc1(output)

    # x shape == (batch_size * max_length, hidden_size)
    x = tf.reshape(x, (-1, x.shape[2]))

    # output shape == (batch_size * max_length, vocab)
    x = self.fc2(x)

    return x, state, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))
