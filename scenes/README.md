# Scene description/ image captioning.
Requirements:
Tensorflow 2.0

Pickle

Numpy

PIL 

**Descrtiption**

This code uses subclassed models to create the model. Sequential models and Functional models are data structures that represent a DAG of layers. As such, they can be safely serialized and deserialized.

A subclassed model differs in that it's not a data structure, it's a piece of code. The architecture of the model is defined via the body of the call method. This means that the architecture of the model cannot be safely serialized. **To load a model, you'll need to have access to the code that created it (the code of the model subclass)**. Alternatively, you could be serializing this code as bytecode (e.g. via pickling), but that's unsafe and generally not portable.

For more information about these differences, see the article ["What are Symbolic and Imperative APIs in TensorFlow 2.0?"](https://medium.com/tensorflow/what-are-symbolic-and-imperative-apis-in-tensorflow-2-0-dfccecb01021).

**Saving the model**:
First of all, a subclassed model that has never been used cannot be saved.

That's because a subclassed model needs to be called on some data in order to create its weights.

Until the model has been called, it does not know the shape and dtype of the input data it should be expecting, and thus cannot create its weight variables. You may remember that in the Functional model from the first section, the shape and dtype of the inputs was specified in advance (via keras.Input(...)) -- that's why Functional models have a state as soon as they're instantiated.

There are 3 approaches to save a subclassed model. In this repository, we used  [save_weights](https://www.tensorflow.org/guide/keras/save_and_serialize)  to create a TensorFlow SavedModel checkpoint, which will contain the value of all variables associated with the model:

1-The layers' weights

2-The optimizer's state

3-Any variables associated with stateful model metrics (if any)

To restore your model, you will need access to the code that created the model object.
We hae three models to Save: 1-imoagenet weights 2-NN Enoder 3-RNN deoder

Since these are subclassed Keras models and not Functional or Sequential one, so I could not use model.save and model.load directly.

Instead I had to use **model.save_weights** and **model.load_weights**. 

#Saving the enoder model

model.load_weights can be called only after model.build and model.build requires input_shape parameter which has to be tuple (not list of tuples). For the NN enoder the input_shape is (49,  1280)/ 

#Saving the Attention weights and the RNN deoder weights

For our RNN decoder, **the input_shape annot be defined since we have multiple inputs**. Keras docs specify no way to call model.build with multiple inputs.

So to save the RNN deoder:

1- save each weight matrix in .npy files

2- re-create the subclassed models, but this time you use [Keras initializers](https://keras.io/initializers/) for each weight in each layer. 

3-Instantiate the Encoder and Decoder classes as you normally would


Original Notebook: https://colab.research.google.com/drive/12YtCH2X0pwIBBXPW0TXmeA520MyVv9AF#forceEdit=true&sandboxMode=true&scrollTo=8Q44tNQVRPFt

**What’s new?**

1-saving the imagenet weights in a h5 file

2-saving the tokenizer in a pkl file and calling it at inference time

3-Using mobilenet 2 for transfer learning (The original notebook used ineption3) Sine it's speed performance is better

3-Training on 50000 images (instead of 30000) and 20 EPOcHS

**Usage**:

if you just want to test on your own images :run mobilenet_inference.py 

if you want to train: run mobilenet_inference.ipynb in Google olaboratory
