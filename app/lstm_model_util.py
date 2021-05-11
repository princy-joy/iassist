import tensorflow as tf
import matplotlib.image as mpimg
from PIL import Image
import collections, json, re, os, pickle, string, warnings
import random
import numpy as np
import pandas as pd
import datetime,time
warnings.filterwarnings("ignore")
from collections import Counter
from glob import glob
from skimage import io
from sklearn.utils import shuffle
from sys import getsizeof

# Tensorflow Packages
from tensorflow.keras.applications import MobileNet
import tensorflow.keras.applications.mobilenet  
from tensorflow.keras.applications.inception_v3 import InceptionV3
import tensorflow.keras.applications.inception_v3
import tensorflow.keras.preprocessing.image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from model_config import train_step, CNN_Encoder, RNN_Decoder

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
WIDTH, HEIGHT = 299, 299
OUTPUT_DIM = 2048
attention_features_shape = 64
max_length = 37
embedding_dim = 200
units = 512
vocab_size = 8779

# Load necessary data to initialize model
with open(APP_ROOT+'/data/tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)
with open(APP_ROOT+'/data/img_vec_batch.txt', 'rb') as f:
    img_vec_batch = pickle.load(f)
with open(APP_ROOT+'/data/cap_batch.txt', 'rb') as f:
    cap_batch = pickle.load(f)
with open(APP_ROOT+'/data/full_image_array.pkl', 'rb') as f:
    full_image = pickle.load(f)

def encodeImage(img):
    incep_cnn = InceptionV3(weights='imagenet')
    incep_cnn = Model(incep_cnn.input, incep_cnn.layers[-2].output)
    preprocess_input = tensorflow.keras.applications.inception_v3.preprocess_input
    img = img.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
    x = tensorflow.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = incep_cnn.predict(x)
    x = np.reshape(x, OUTPUT_DIM )
    return x

def initialize_model():
    # Load sample dataset
    batch_dataset = tf.data.Dataset.from_tensor_slices((img_vec_batch, cap_batch))
    batch_dataset = batch_dataset.shuffle(100).batch(64)
    batch_dataset = batch_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)
    # Initialize basic model
    for (batch, (img_tensor, target)) in enumerate(batch_dataset):
      batch_loss, t_loss = train_step(img_tensor, target)
    # Load weights to model
    encoder.load_weights(APP_ROOT+"/models/lstm_encoder_weights")
    decoder.load_weights(APP_ROOT+"/models/lstm_decoder_weights")
    return encoder, decoder

def evaluate(image_path):
    encoder, decoder = initialize_model()
    attention_plot = np.zeros((max_length, attention_features_shape))
    hidden = decoder.reset_state(batch_size=1)
    img = tensorflow.keras.preprocessing.image.load_img(image_path, target_size=(HEIGHT, WIDTH))
    img_encode = encodeImage(img)
    img_encode = tf.convert_to_tensor(img_encode, dtype=tf.float32)
    img_encode = tf.reshape(img_encode, (1, img_encode.shape[0]))
    features = encoder(img_encode)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []
    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        if tokenizer.index_word[predicted_id] == '<end>':
            pred_caption = ''.join(result)
            return pred_caption
        result.append(tokenizer.index_word[predicted_id])
        dec_input = tf.expand_dims([predicted_id], 0)
    pred_caption = ''.join(result)
    return pred_caption

def caption_image(image_path):
    return evaluate(image_path)
