"""Train a neural network model designed for instance-specific segmentation

The network design is provided by Mask-RCNN package.

"""

import argparse
import os

from mrcnn import model as modellib
from mrcnn import utils
from mrcnn.config import Config

from tanzania_challenge import buildings, config


def prepare_dataset(datapath, img_size, mode="training"):
    """Prepare an image dataset on the model of mrcnn datasets

    Parameters
    ----------
    datapath : str
        Data folder on the file system
    img_size : int
        Image size, in pixel
    mode : str
        Dataset type, either "training" or "validation", or "testing"

    Returns
    -------
    buildings.BuildingDataset
        Dataset of buildings, ready for training
    """
    dataset = buildings.BuildingDataset()
    dataset.load_buildings(datapath, subset=mode, img_size=img_size)
    dataset.prepare()
    return dataset

def train(datapath):
    """Train a Mask-RCNN model from scratch with images contained into `datapath`
    folder and return it after 60 training epochs

    Parameters
    ----------
    datapath : str
        Data folder on the file system, images are in "preprocessed" subfolder,
    whilst models are stored into a "output" subfolder

    Returns
    -------
    keras.models.Model
        Trained Mask-RCNN model

    """
    building_config = config.BuildingConfig()
    training_bd = prepare_dataset(os.path.join(datapath, "preprocessed"),
                                  building_config.IMAGE_MIN_DIM, "training")
    val_bd = prepare_dataset(os.path.join(datapath, "preprocessed"),
                             building_config.IMAGE_MIN_DIM, "validation")
    model_path = os.path.join(datapath, "output",
                              "instance_segmentation", "checkpoints")
    model = modellib.MaskRCNN(mode="training", config=building_config,
                              model_dir=model_path)
    model.train(train_dataset=train_bd, val_dataset=val_bd,
                learning_rate=building_config.LEARNING_RATE,
                epochs=20,
                layers='all')
    model.train(train_dataset=train_bd, val_dataset=val_bd,
            learning_rate=building_config.LEARNING_RATE/2,
            epochs=40,
            layers='all')
    model.train(train_dataset=train_bd, val_dataset=val_bd,
            learning_rate=building_config.LEARNING_RATE/10,
            epochs=60,
            layers='all')
    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=("Train an instance "
                                                  "segmentation model with "
                                                  "Mask-RCNN"))
    parser.add_argument('-d', '--datapath',
                        default="./data/open_ai_tanzania",
                        help=("Path towards dataset"))

    args = parser.parse_args()

    train(args.datapath)
