import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

import mrcnn.model as modellib
from mrcnn.model import log

from config.fashion_config import FashionConfig
from retrieval.detector import Detector

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join("", "weight/mask_rcnn_coco.h5")

# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join("", "logs")


if __name__ == '__main__':    
    # Configurations    
    config = FashionConfig()        

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)    

    # Select weights file to load    
    weights_path = COCO_WEIGHTS_PATH            
    
    # Exclude the last layers because they require a matching
    # number of classes
    model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])    

    # Train
    detector = Detector("weight/mask_rcnn_fashion.h5")
    detector.train(model, config)    
