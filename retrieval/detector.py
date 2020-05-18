import os
import sys
import time
import skimage.io
import matplotlib.pyplot as plt
import tensorflow as tf
import mrcnn.model as modellib
import numpy as np

# Path to Datasets
DEFAULT_DATASETS_DIR = os.path.join("", "../../../../datasets")

sys.path.insert(0, "../")
from config.dataset import FashionDataset
from config.fashion_config import FashionConfig

class Detector:
	def __init__(self, model_path):
		config = FashionConfig()  

		self.class_names = ['BG', 'top', 'long', 'bottom']
		# Using GPU
		with tf.device('/gpu:0'):
			self.model = modellib.MaskRCNN(mode="inference", config=config, model_dir="")
		
		self.model.load_weights(model_path, by_name=True)

	def detection(self, image): 
		if not isinstance(image, np.ndarray):
			raise ValueError("Input is incorrect")

		# Run detection
		start = time.time()
		detection_results = self.model.detect([image], verbose=1)
		end = time.time()

		# Results
		print("Cost time: ",end-start," (s)")
		result = detection_results[0]
		
		return result

	def train(self, model, config):    
    	# Dataset
		dataset_train = FashionDataset()
		dataset_train.load_data(DEFAULT_DATASETS_DIR+"/train.json", DEFAULT_DATASETS_DIR+"/train")
		dataset_train.prepare()

		# Validation dataset
		dataset_val = FashionDataset()
		dataset_val.load_data(DEFAULT_DATASETS_DIR+"/validation.json", DEFAULT_DATASETS_DIR+"/val")
		dataset_val.prepare()

	    # Training
		print("Training network heads")
		model.train(dataset_train, dataset_val, 
	    	learning_rate=config.LEARNING_RATE,
	    	epochs=15, layers='heads')

		model.train(dataset_train, dataset_val,
			learning_rate=config.LEARNING_RATE / 10,
			epochs=30, layers="all")