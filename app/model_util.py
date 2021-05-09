import os
import pickle
from math import log

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.applications.inception_v3 import (InceptionV3,
                                                        preprocess_input)
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Input, add
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

WIDTH = 299
HEIGHT = 299
CHANNELS = 3
START = "startseq" 
STOP = "endseq"
max_length = 194
APP_ROOT = os.path.dirname(os.path.abspath(__file__))


def encode(img):
    inception = InceptionV3(weights = 'imagenet', input_shape = (WIDTH, HEIGHT, CHANNELS))
    cnn_model = Model(inputs = inception.input, outputs = inception.layers[-2].output)

    img = image.load_img(img,target_size=(299,299))
    img = image.img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis = 0) 
    feature_vector = cnn_model.predict(img)
    return feature_vector

def generate_caption(photo):
    with open(APP_ROOT+'/models/word_to_index.pkl', 'rb') as f:
      wordtoidx = pickle.load(f)
    with open(APP_ROOT+'/models/index_to_word.pkl', 'rb') as f:
      idxtoword = pickle.load(f)
    caption_model = load_model(APP_ROOT+'/models/caption_model.hdf5')

    in_text = START
    for i in range(max_length):
        sequence = [wordtoidx[w] for w in in_text.split() if w in wordtoidx]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idxtoword[yhat]
        in_text += ' ' + word
        if word == STOP:
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

def generate_caption_alt(image, beam_width = 3, alpha = 0.7,max_len = 38):
    with open(APP_ROOT+'/models/word_to_index.pkl', 'rb') as f:
      wordtoidx = pickle.load(f)
    with open(APP_ROOT+'/models/index_to_word.pkl', 'rb') as f:
      idxtoword = pickle.load(f)
    
    caption_model = load_model(APP_ROOT+'/models/model_weights.h5')

    l = [('<start>', 1.0)]
    for i in range(max_len):
      temp = []
      for j in range(len(l)):
        sequence = l[j][0]
        prob = l[j][1]
        if sequence.split()[-1] == '<end>':
          t = (sequence, prob)
          temp.append(t)
          continue
        encoding = [wordtoidx[word] for word in sequence.split() if word in wordtoidx]
        encoding = pad_sequences([encoding], maxlen = max_len, padding = 'post')
        pred = caption_model.predict([image, encoding])[0]
        pred = list(enumerate(pred))
        pred = sorted(pred, key = lambda x: x[1], reverse = True)
        pred = pred[:beam_width]
        for p in pred:
          if p[0] in idxtoword:
              t = (sequence + ' ' + idxtoword[p[0]], (prob + log(p[1])) / ((i + 1)**alpha))
              temp.append(t)
      temp = sorted(temp, key = lambda x: x[1], reverse = True)
      l = temp[:beam_width]
    caption = l[0][0]
    caption = caption.split()[1:-1]
    caption = ' '.join(caption)
    return caption


def caption_image(img):
    enc_img = encode(img)
    return generate_caption_alt(enc_img)