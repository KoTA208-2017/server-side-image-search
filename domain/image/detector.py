import os
import sys
import time
import skimage.io
import matplotlib.pyplot as plt
import tensorflow as tf
import mrcnn.model as modellib
from mrcnn import visualize
import numpy as np

# Path to Datasets
DEFAULT_DATASETS_DIR = os.path.join("", "../../../../datasets")
DEFAULT_LOGS_DIR = os.path.join("", "logs/train")
# Path to trained weights file

sys.path.insert(0, "../../")
from technical_service.config.Dataset import FashionDataset
from technical_service.config.fashion_config import FashionConfig

class Detector:
	def __init__(self, model_path, mode):
		self.config = FashionConfig()  

		self.class_names = ['BG', 'top', 'long', 'bottom']
		# Using GPU
		if(mode == "detection"):
			with tf.device('/gpu:0'):
				self.model = modellib.MaskRCNN(mode="inference", config=self.config, model_dir="")
				self.model.load_weights(model_path, by_name=True)
		else:
			self.model = modellib.MaskRCNN(mode="training", config=self.config, model_dir=DEFAULT_LOGS_DIR)
			self.model.load_weights(model_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
				"mrcnn_bbox", "mrcnn_mask"]) 				

	def train(self, dataset_train, dataset_val):    
	    # Training
		print("Training network heads")
		self.model.train(dataset_train, dataset_val, 
			learning_rate=self.config.LEARNING_RATE,
			epochs=15, layers='heads')

		self.model.train(dataset_train, dataset_val,
			learning_rate=self.config.LEARNING_RATE / 10,
			epochs=30, layers="all")

	def detection(self, image): 
		# Handle wrong input
		if not isinstance(image, np.ndarray):
			raise ValueError("Input is incorrect")			
			return None

		# Check image channel
		if(image.shape[2] == 4):
			image = image[...,:3]

		# Run detection
		start = time.time()
		detection_results = self.model.detect([image], verbose=1)
		end = time.time()

		# Results
		print("Cost time: ",end-start," (s)")
		result = detection_results[0]
		return result


	def get_width(self, detection_result):
		width = abs(detection_result[1] - detection_result[3])
		return width

	def get_height(self, detection_result):
		height = abs(detection_result[0] - detection_result[2])
		return height

	def get_area(self, detection_result):
		width = self.get_width(detection_result)
		height = self.get_height(detection_result)
		area = width * height
		return area

	def get_biggest_box(self, detection_result_list):
		biggest_area = 0
		for i, xy in enumerate(detection_result_list):
			area = self.get_area(xy)
			if area > biggest_area:
				biggest_area = area
				biggest_xy = xy
				ix = i
		return biggest_xy

	def save_cropped_image(self, image, image_dir, cropped):
		saved_image = np.copy(image).astype('uint8') 
		if cropped:
			image_name = "cropped.png"
		else:
			image_name = "resized.png"

		plt.imsave(os.path.join(image_dir, image_name), saved_image, cmap = plt.cm.gray)	

	def crop_object(self, img, detection_result, image_dir):    
		target = img[detection_result[0]:detection_result[2], detection_result[1]:detection_result[3], :]
		self.save_cropped_image(target, image_dir, True)
		# Resize to 224 x 224
		resized = skimage.transform.resize(target, (224, 224), preserve_range=True)
		self.save_cropped_image(resized, image_dir, False)
		return resized	

	def save_image(self, image, detection_result, image_dir):
		visualize.save_image(image, "detection_result", detection_result['rois'], 
			detection_result['masks'], detection_result['class_ids'], detection_result['scores'],
			self.class_names, image_dir, mode=0)	

