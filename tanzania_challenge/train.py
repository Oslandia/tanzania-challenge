"""Train a neural network model designed for instance-specific segmentation

The network design is provided by Mask-RCNN package.

"""

import argparse
import os

from mrcnn import model as modellib
from mrcnn import utils
from mrcnn.config import Config

from tanzania_challenge import buildings, config



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=("Train an instance "
                                                  "segmentation model with "
                                                  "Mask-RCNN"))
    parser.add_argument('-d', '--datapath',
                        default="./data/open_ai_tanzania",
                        help=("Path towards dataset"))
    parser.add_argument('-pm', '--pretrained-model',
                        default="../Mask_RCNN/mask_rcnn_coco.h5",
                        help=("Path of a pre-trained model"))

    args = parser.parse_args()


    config = config.BuildingConfig()

    train_bd = buildings.BuildingDataset()
    train_bd.load_buildings(os.path.join(args.datapath, "preprocessed"),
                            subset="training",
                            img_size=config.IMAGE_MIN_DIM)
    train_bd.prepare()

    val_bd = buildings.BuildingDataset()
    val_bd.load_buildings(os.path.join(args.datapath, "preprocessed"),
                          subset="validation",
                          img_size=config.IMAGE_MIN_DIM)
    val_bd.prepare()
    
    model_path = os.path.join(args.datapath, "output",
                              "instance_segmentation", "checkpoints")
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=model_path)
    model.load_weights(args.pretrained_model, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])

    model.train(train_dataset=train_bd, val_dataset=val_bd,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                layers='heads')

    model.train(train_dataset=train_bd, val_dataset=val_bd,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                layers='4+')

    model.train(train_dataset=train_bd, val_dataset=val_bd,
                learning_rate=config.LEARNING_RATE,
                epochs=60,
                layers='all')
