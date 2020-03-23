# Import MASK R-CNN
from mrcnn.config import Config

class FashionConfig(Config):
	# Give the configuration a recognizable name
    NAME = "fashion"

    # Train on 1 GPU and 1 image per GPU. Batch size is 1 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # background + 1 (top, long, bottom)

    # You can experiment with this number to see if it improves training
    STEPS_PER_EPOCH = 500

    # This is how often validation is run. If you are using too much hard drive space
    # on saved models (in the MODEL_DIR), try making this value larger.
    VALIDATION_STEPS = 100
    
    # Matterport originally used resnet101, but I downsized to fit it on my graphics card
    IMAGE_MIN_DIM = 512