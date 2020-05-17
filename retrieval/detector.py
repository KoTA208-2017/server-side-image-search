import os
import sys
# Path to Datasets
DEFAULT_DATASETS_DIR = os.path.join("", "../../../../datasets")

sys.path.insert(0, "../")
from config.dataset import FashionDataset

class Detector:

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