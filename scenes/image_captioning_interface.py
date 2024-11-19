from models import ImageFeatureExtract,CNN_Encoder,RNN_Decoder
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
import os
import cv2

# abspath = os.path.abspath(__file__)
# dname = os.path.dirname(abspath)
# os.chdir(dname)

#max_length of_train_sequences
max_length=46

#parameters used in training process
embedding_dim = 256
units = 512
len_tokenizer_word_index=26555
vocab_size = len_tokenizer_word_index + 1

class image_captioning ():
    def __init__(self,
    mobile_net_v2_weights = 'mobilenet',
    alpha = 1.4):
        self.image_features_extract_model = ImageFeatureExtract(mobile_net_v2_weights = mobile_net_v2_weights,alpha = alpha)
        self.E = CNN_Encoder(embedding_dim)
        self.D = RNN_Decoder(embedding_dim, units, vocab_size)
        infile = open('tokenizer.pickle','rb')
        self.toketokenizer= pickle.load(infile)
        infile.close()

    def preprocess_image_nparray(self,image):
        img = tf.convert_to_tensor(image, dtype=tf.uint8)
        #try for DCT
        # img = tf.convert_to_tensor(image, dtype=tf.float32)
        # s0, s1, s2 = tf.split(img, num_or_size_splits=3, axis=2)
        # s0 = tf.signal.dct(s0,type=3)
        # s1 = tf.signal.dct(s1,type=3)
        # s2 = tf.signal.dct(s2,type=3)
        # img = tf.concat([s0,s1,s2], 2)
        #try
        # img = tf.reshape(img,[720*1280*3])
        # img = tf.signal.dct(img,type=2)
        # img = tf.reshape(img,[720,1280,3])
        # img = tf.cast(img,tf.uint8)
        # print(img)
        img = tf.image.resize(img, (224, 224))
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img) 
        return img

    def load_image(self,image_path):
        img = tf.io.read_file(image_path)
        # img = tf.image.decode_jpeg(img, dct_method = "INTEGER_ACCURATE",channels=3)
        img = tf.image.decode_jpeg(img,channels=3)
        img = tf.image.resize(img, (224, 224))
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img) 
        return img

    def encode(self,image):
        img = tf.convert_to_tensor(image, dtype=tf.uint8)
        img = tf.image.encode_jpeg(img,
            quality=100,
            progressive=False,
            optimize_size=False,
            chroma_downsampling=True,
            density_unit='in',
            x_density=300,
            y_density=300)
        # img = tf.image.encode_png(
        #     img, compression=-1, name=None)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (224, 224))
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img) 
        return img

    def generate_from_img_nparray_encode(self,image):
        im = tf.expand_dims(self.encode(image), 0)
        return self.generate_caption(im)

    def generate_from_saving_img_opencv(self,image):
        image_path = "temp.jpg"
        cv2.imwrite(image_path,image,[cv2.IMWRITE_JPEG_QUALITY,100,cv2.IMWRITE_JPEG_PROGRESSIVE,1,cv2.IMWRITE_JPEG_OPTIMIZE,1\
        ,cv2.IMWRITE_JPEG_LUMA_QUALITY,100,cv2.IMWRITE_JPEG_CHROMA_QUALITY,100])
        return self.generate_from_img_path(image_path)
    def generate_from_saving_img_pillow(self,image):
        image_path = "temp.jpg"
        image = Image.fromarray(image)
        image.save(image_path)
        return self.generate_from_img_path(image_path)

    def generate_from_img_nparray(self,image):
        im = tf.expand_dims(self.preprocess_image_nparray(image), 0)
        return self.generate_caption(im)

    def generate_from_img_path(self,image):
        im = tf.expand_dims(self.load_image(image), 0)
        return self.generate_caption(im)

    def generate_caption(self,temp_input):
        hidden = self.D.reset_state(batch_size=1)

        img_tensor_val = self.image_features_extract_model.extract(temp_input)
        img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

        features = self.E(img_tensor_val)
        
        dec_input = tf.expand_dims([self.toketokenizer.word_index["<start>"]], 0)
        result = []
        for i in range(max_length):
            predictions, hidden, attention_weights = self.D(dec_input, features, hidden)
            predicted_id = tf.argmax(predictions[0]).numpy()
        
            result.append(self.toketokenizer.index_word[predicted_id])
            if self.toketokenizer.index_word[predicted_id] == '<end>':
                return ' '.join(result[:-1])
            dec_input = tf.expand_dims([predicted_id], 0)
        return ' '.join(result[:-1])

#For testing purpose
if __name__ == '__main__':    
    # while(1):
    #     test_image = input("\n")
    #     im = cv2.imread(test_image)
    #     print (type(im))
    #     print(cv2.imdecode(im,1))
    model = image_captioning (mobile_net_v2_weights='mobilenet_v2_weights_1.4.h5')
    while(1):
        test_image = input("enter path:")

        #opencv
        # im = cv2.imread(test_image)
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        #pillow
        im = Image.open(test_image)
        im = np.array(im)
        
        print("generate_from_img_nparray_encode\n" + model.generate_from_img_nparray_encode(im))
        print("generate_from_img_nparray\n" +model.generate_from_img_nparray(im))
        print("generate_from_saving_img_opencv\n" +model.generate_from_saving_img_opencv(im))
        print("generate_from_saving_img_pillow\n" +model.generate_from_saving_img_pillow(im))
        print("generate_from_img_path\n" +model.generate_from_img_path(test_image)) #take path
        print('\n')
        
