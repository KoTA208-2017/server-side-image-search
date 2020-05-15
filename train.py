import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from config.fashion_config import FashionConfig
from config.dataset import FashionDataset

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join("", "weight/mask_rcnn_coco.h5")

# Path to Datasets
DEFAULT_DATASETS_DIR = os.path.join("", "datasets")

# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join("", "logs")

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = FashionDataset()
    dataset_train.load_data(args.dataset+"/train.json", args.dataset+"/train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = FashionDataset()
    dataset_val.load_data(args.dataset+"/validation.json", args.dataset+"/val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=15	,
                layers='heads')

    model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=30, 
            layers="all")

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect fashion.')

    parser.add_argument('--dataset', required=False,
    					default=DEFAULT_DATASETS_DIR,                        
                        help='Directory of the fashion dataset')
    parser.add_argument('--weights', required=False,
    					default=COCO_WEIGHTS_PATH,                        
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations    
    config = FashionConfig()    
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)    

    # Select weights file to load    
    weights_path = COCO_WEIGHTS_PATH        
    print("Loading weights ", weights_path)
    
    # Exclude the last layers because they require a matching
    # number of classes
    model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])    

    # Train
    train(model)
