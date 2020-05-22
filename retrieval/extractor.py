import tensorflow as tf
import numpy as np
import  matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse

from numpy import linalg as LA

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

class Extractor:
  def __init__(self):      
    # weights: 'imagenet'
    # pooling: 'max' or 'avg'
    # input_shape: (width, height, 3), width and height should >= 48
    self.input_shape = (224, 224, 3)
    self.weight = 'imagenet'
    self.pooling = 'max'
    self.model = VGG16(weights = self.weight, input_shape = (self.input_shape[0], self.input_shape[1], self.input_shape[2]), pooling = self.pooling, include_top = True)
    self.model = Model(inputs=self.model.inputs, outputs=self.model.get_layer('fc2').output)
    self.model.predict(np.zeros((1, 224, 224 , 3)))

    '''
    Use vgg16 model to extract features
    Output normalized feature vector
    '''
  def extract_feat(self, image):
    if(isinstance(image, np.ndarray)):
      if(image.shape == (224,224,3)):
        img = np.asarray(image, dtype=np.float64) # This is an int array!            
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feat = self.model.predict(img)
        norm_feat = feat[0]/LA.norm(feat[0])
        return norm_feat
      else:
        raise ValueError("Input shape is incorrect")        
    else:
      raise ValueError("Input is incorrect")              

    return None