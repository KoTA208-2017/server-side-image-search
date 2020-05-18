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

	def detection(self, image): 
		# Handle wrong input
		if not isinstance(image, np.ndarray):
			raise ValueError("Input is incorrect")
			print("bel")
			return None

		# Run detection
		start = time.time()
		detection_results = self.model.detect([image], verbose=1)
		end = time.time()

		# Results
		print("Cost time: ",end-start," (s)")
		result = detection_results[0]
		
		return result


	def get_width(self, xy):
		width = abs(xy[1] - xy[3])
		return width

	def get_height(self, xy):
		height = abs(xy[0] - xy[2])
		return height

	def get_area(self, xy):
		width = self.get_width(xy)
		height = self.get_height(xy)
		area = width * height
		return area

	def get_biggest_box(self, xy_list):
		biggest_area = 0
		for i, xy in enumerate(xy_list):
			area = self.get_area(xy)
			if area > biggest_area:
				biggest_area = area
				biggest_xy = xy
				ix = i
		return biggest_xy

	def crop_object(self, img, xy):    
		target = img[xy[0]:xy[2], xy[1]:xy[3], :]
		# Resize to 224 x 224
		resized = skimage.transform.resize(target, (224, 224), preserve_range=True)
		return resized


