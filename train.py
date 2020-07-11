import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

import mrcnn.model as modellib
from mrcnn.model import log

# from config.fashion_config import FashionConfig
from domain.image.detector import Detector
from technical_service.config.Dataset import FashionDataset

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join("", "weight/mask_rcnn_coco.h5")

# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join("", "logs")

DEFAULT_DATASETS_DIR = os.path.join("", "../../../../datasets")


if __name__ == '__main__':        
    dataset_train = FashionDataset()
    dataset_train.load_data(DEFAULT_DATASETS_DIR+"/train.json", DEFAULT_DATASETS_DIR+"/train")
    dataset_train.prepare()

        # # Validation dataset
    dataset_val = FashionDataset()
    dataset_val.load_data(DEFAULT_DATASETS_DIR+"/validation.json", DEFAULT_DATASETS_DIR+"/val")
    dataset_val.prepare()

    # Train
    detector = Detector(COCO_WEIGHTS_PATH, "training")
    detector.train(dataset_train, dataset_val)    
