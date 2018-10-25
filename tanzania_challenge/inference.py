"""
"""

import argparse
import json
import numpy as np
import os
from PIL import Image

from mrcnn import model as modellib

from tanzania_challenge import config
from tanzania_challenge.train import prepare_dataset


def predict_on_filename(datapath, img_size, filename):
    """Do an instance segmentation prediction on a given image starting from
    its location on file system, and provides resulting masks, class IDs and
    prediction scores associated to each detected instance

    Parameters
    ----------
    datapath : str
        Path of data on file system
    img_size : int
        Size of tiled images to provide to the model
    filename : str
        Name of the image file into the testing dataset

    Returns
    -------
    dict
        Mask, class IDs and prediction scores for each detected instance
    """
    building_config = config.BuildingInferenceConfig()
    model_path = os.path.join(datapath, "output",
                              "instance_segmentation", "checkpoints")
    model = modellib.MaskRCNN(mode="inference", config=building_config,
                              model_dir=model_path)
    weights_path = model.find_last()
    model.load_weights(weights_path, by_name=True)
    image = Image.open(os.path.join(datapath, "preprocessed", str(img_size),
                                    "testing", "images", filename + ".tif"))
    image_data = np.array(image)
    prediction = model.detect(np.expand_dims(image_data, 0))[0]
    results = {"masks": np.moveaxis(prediction["masks"], 2, 0).tolist(),
               "class_ids": prediction["class_ids"].tolist(),
               "scores": prediction["scores"].tolist()}
    return results


def predict_on_folder(datapath, img_size):
    """Do an instance segmentation prediction on a given image starting from
    its location on file system, and provides resulting masks, class IDs and
    prediction scores associated to each detected instance

    Parameters
    ----------
    datapath : str
        Path of data on file system
    img_size : int
        Size of tiled images to provide to the model

    Returns
    -------
    dict
        Mask, class IDs and prediction scores for each detected instance
    """
    building_config = config.BuildingInferenceConfig()
    model_path = os.path.join(datapath, "output",
                              "instance_segmentation", "checkpoints")
    model = modellib.MaskRCNN(mode="inference", config=building_config,
                              model_dir=model_path)
    weights_path = model.find_last()
    model.load_weights(weights_path, by_name=True)

    test_bd = prepare_dataset(os.path.join(datapath, "preprocessed"),
                              img_size, "testing")
    nb_images = test_bd.num_images
    log = {}
    for image_id in test_bd.image_ids[:nb_images]:
        image = test_bd.load_image(image_id)
        prediction = model.detect(np.expand_dims(image, 0))[0]
        results = {"masks": np.moveaxis(prediction["masks"], 2, 0).tolist(),
                   "class_ids": prediction["class_ids"].tolist(),
                   "scores": prediction["scores"].tolist()}
        with open(test_bd.image_info[image_id]["prediction_path"], "w") as fobj:
            json.dump(results, fobj)
        log[test_bd.image_info[image_id]["name"].split(".")[0]] = len(prediction["class_ids"])
    return log


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=("Infer building instances on"
                                                  "images with a trained Mask-"
                                                  "RCNN model."))
    parser.add_argument('-d', '--datapath',
                        default="./data/open_ai_tanzania",
                        help=("Path towards dataset"))
    parser.add_argument('-s', '--image-size',
                        type=int,
                        default=384,
                        help=("Image size (in pixels)"))
    parser.add_argument('-f', '--filename',
                        help=("Image file name"))

    args = parser.parse_args()

    predict(args.datapath, args.image_size, args.filename)
