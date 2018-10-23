"""Design a configuration object that describes the way Buildings must be
instanciated

BuildingConfig inheritates from `mrcnn.config.Config`.

"""

import os

from mrcnn.config import Config


class BuildingConfig(Config):

    NAME = "buildings"
    NUM_CLASSES = 1 + 3
    IMAGE_MIN_DIM = 384
    IMAGE_MAX_DIM = 384

    BACKBONE = "resnet101"
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    RPN_ANCHOR_SCALES = [0.5, 1, 2]
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    RPN_ANCHOR_STRIDES = 1
    TRAIN_ROIS_PER_IMAGE = 32

    training_path = os.path.join("data", "open_ai_tanzania",
                                 "preprocessed", str(IMAGE_MIN_DIM),
                                 "training", "images")
    validation_path = os.path.join("data", "open_ai_tanzania",
                                   "preprocessed", str(IMAGE_MIN_DIM),
                                   "validation", "images")

    GPU_COUNT = 1
    IMAGES_PER_GPU = 5
    STEPS_PER_EPOCH = len(os.listdir(training_path)) // IMAGES_PER_GPU
    VALIDATION_STEPS = len(os.listdir(validation_path)) // IMAGES_PER_GPU
    LEARNING_MOMENTUM = 0.9
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0001


class BuildingInferenceConfig(Config):

    NAME = "buildings"
    NUM_CLASSES = 1 + 3
    IMAGE_MIN_DIM = 384
    IMAGE_MAX_DIM = 384

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
