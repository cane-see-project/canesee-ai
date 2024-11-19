import tensorflow as tf
import numpy as np
import pickle
from PIL import Image


# Shape of the vector extracted from mobilenetv2 is (49, 1792)
# These two variables represent that vector shape


class ImageFeatureExtract():
    def __init__(self,mobile_net_v2_weights = 'mobilenet',alpha = 1.4):
        image_model=tf.keras.applications.MobileNetV2(include_top=False, weights = mobile_net_v2_weights,alpha = alpha)
        new_input=image_model.input
        hidden_layer=image_model.layers[-1].output
        self.model = tf.keras.Model(new_input,hidden_layer)
    def extract(self, temp_input):
        return self.model(temp_input)

class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        C = tf.keras.initializers.Constant
        #here, we'll load the weights from the tflite instead
        w1, w2 = [np.load("encoder_layer_weights/layer_%s_%s_weights_%s.npy" %(0, "dense", j)) \
                                      for j in range(2)]
        self.fc = tf.keras.layers.Dense(embedding_dim, kernel_initializer=C(w1), bias_initializer=C(w2))

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    C = tf.keras.initializers.Constant
    w1, w2, w3, w4, w5, w6 = [np.load("decoder_layer_weights/layer_%s_%s_weights_%s.npy" %(4, "bahdanau_attention", j)) \
                                  for j in range(6)]
    self.W1 = tf.keras.layers.Dense(units, kernel_initializer=C(w1), bias_initializer=C(w2))
    self.W2 = tf.keras.layers.Dense(units, kernel_initializer=C(w3), bias_initializer=C(w4))
    self.V = tf.keras.layers.Dense(1, kernel_initializer=C(w5), bias_initializer=C(w6))

  def call(self, features, hidden):

    hidden_with_time_axis = tf.expand_dims(hidden, 1)
    score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

    # attention_weights shape == (batch_size, 64, 1)
    # you get 1 at the last axis because you are applying score to self.V
    attention_weights = tf.nn.softmax(self.V(score), axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class RNN_Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(RNN_Decoder, self).__init__()
    self.units = units

    C = tf.keras.initializers.Constant
    w_emb = np.load("decoder_layer_weights/layer_%s_%s_weights_%s.npy" %(0, "embedding", 0))
    w_gru_1, w_gru_2, w_gru_3 = [np.load("decoder_layer_weights/layer_%s_%s_weights_%s.npy" %(1, "gru", j)) for j in range(3)]
    w1, w2 = [np.load("decoder_layer_weights/layer_%s_%s_weights_%s.npy" %(2, "dense_1", j)) for j in range(2)]
    w3, w4 = [np.load("decoder_layer_weights/layer_%s_%s_weights_%s.npy" %(3, "dense_2", j)) for j in range(2)]

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

    self.attention = BahdanauAttention(self.units)

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



