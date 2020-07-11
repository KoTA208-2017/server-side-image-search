import os
import h5py
import numpy as np
import argparse
import time
from PIL import Image
import skimage.io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Local Import
from technical_service.database.database import DAO
from technical_service.database.model import Product

from domain.image.detector import Detector
from domain.image.extractor import Extractor


IMAGE_PATH = "domain/server/image/product_images/"

if __name__ == '__main__':
	database = DAO()
	# Get Data Product
	products = database.getAll()

	# Feature Extraction 
	# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
	print("--------------------------------------------------")
	print("         feature extraction starts")
	print("--------------------------------------------------")
	start = time.time()

	# Array
	feats = []
	ids = []

	log_dir = os.path.join("logs", "extraction") 
	os.mkdir(log_dir)
	# Extractor using VGG16
	modelVGG = Extractor()
	# Detector
	image_detector = Detector("weight/mask_rcnn_fashion.h5", "detection")

	for product in products:
		# Image
		image = skimage.io.imread(IMAGE_PATH+product.image)

		image_detection = image_detector.detection(image)
		if(len(image_detection['rois']) > 0):
			# Objek Dominan
			big_box = image_detector.get_biggest_box(image_detection['rois'])
			# Crop Image
			image = image_detector.crop_object(image, big_box, log_dir)
			norm_feat = modelVGG.extract_feat(image)    
			# Add feature and product id to array
			feats.append(norm_feat)
			ids.append(product.id)

	feats = np.array(feats)
	# directory for storing extracted features
	output = "featureCNN_map.h5"

	print("--------------------------------------------------")
	print("      writing feature extraction results ...")
	print("--------------------------------------------------")

	h5f = h5py.File(output, 'w')
	h5f.create_dataset('feats', data = feats)
	h5f.create_dataset('id', data = ids)
	h5f.close()
	end = time.time()
	print("Cost time: ",end-start," (s)")