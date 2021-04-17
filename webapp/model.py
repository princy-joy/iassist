import tensorflow as tf
import numpy as np
import json
import os
import requests
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

model = load_model('./models/model_weights_50.h5')

print(model.summary())