"""Building dataset class on the model of Mask-RCNN Dataset

To add building instances into the dataset, use the load_buildings() method by
passing as the first parameter the folder that contains building informations
(images, features, items)

"""

import json
import cv2
import os
import numpy as np
from PIL import Image
from shapely import wkb, geometry
from mrcnn.utils import Dataset


class BuildingDataset(Dataset):
    """Reformatted Tanzania building dataset, that have to be automatically
    detected through Mask-RCNN instance-specific segmentation algorithms.

    """
    
    condition_dict = {"Complete": 1, "Incomplete": 2, "Foundation": 3}
    
    def load_buildings(self, datadir, subset="training", img_size=400):
        """Load buildings that are described in `datadir`/`img_size`/`subset`.

        This folder must contains images in a `images` subfolder, labels in
        another `items` subfolder and some geographical features in a
        `features` subfolder.
        
        Parameters
        ----------
        datadir : str
            Path to image subsets
        subset : str
            Subset of interest, either 'training', 'validation' or 'testing'
        img_size : int
            Size of the dataset images (width=height)

        """
        self.add_class("building", 1, "complete")
        self.add_class("building", 2, "incomplete")
        self.add_class("building", 3, "foundation")

        image_dir = os.path.join(datadir, str(img_size), subset, "images")
        feature_dir = os.path.join(datadir, str(img_size), subset, "features")
        item_dir = os.path.join(datadir, str(img_size), subset, "items")
        prediction_dir = os.path.join(datadir, str(img_size), subset, "predicted_labels")
        image_filenames = os.listdir(image_dir)
        raw_path = os.path.join(os.path.dirname(os.path.dirname(datadir)),
                                "input", "images")
        for i, filename in enumerate(image_filenames):
            raw_filename = os.path.join(raw_path, "_".join(filename.split("_")[:2])+".tif")
            self.add_image("building", image_id=i,
                           name=filename,
                           raw_image_path=raw_filename,
                           path=os.path.join(image_dir, filename),
                           feature_path=os.path.join(feature_dir,
                                                      filename.replace(".tif", ".json")),
                           item_path=os.path.join(item_dir,
                                                  filename.replace(".tif", ".json")),
                           prediction_path=os.path.join(prediction_dir,
                                                        filename.replace(".tif", ".json")),
                           width=img_size, height=img_size)

    def load_mask(self, image_id):
        """Load mask of the building of ID `image_id`

        Parameters
        ----------
        image_id : str
            Image IDs, *i.e.* its name on the file system
        mask_dir : str
            Name of the mask folder, *e.g.* "labels" or "masks"

        Returns
        -------
        np.array
            Mask as an array of shape [K, img_size, img_size], with K being the
        amount of objects to detect
        np.array
            Class of each object to detect (shape [K]) 
        """
        info = self.image_info[image_id]
        with open(info["item_path"]) as fobj:
            items = json.load(fobj)
        with open(info["feature_path"]) as fobj:
            features = json.load(fobj)
        conditions = [self.condition_dict.get(v["condition"], 1)
                      for k, v in items.items()]
        mask = np.zeros(shape=(info["width"], info["height"], len(conditions)),
                        dtype='int8')
        if len(conditions) == 0:
            return mask, np.array(conditions, dtype=np.uint8)
        for k, v in items.items():
            polygon = wkb.loads(v["geom"], hex=True)
            assert(type(polygon) != geometry.MultiPolygon,
                   "Multipolygon in image {}!".format(image_id))
                # for p in list(polygon):
                #     points = self.extract_points_from_polygon(p, features)
                #     submask = np.array(mask[:, :, int(k)], dtype='int8')
                #     mask[:, :, int(k)] = cv2.fillPoly(submask, points, 1)
            points = self.extract_points_from_polygon(polygon, features)
            submask = np.array(mask[:, :, int(k)], dtype='int8')
            mask[:, :, int(k)] = cv2.fillPoly(submask, points, 1)
        return mask, np.array(conditions, dtype=np.uint8)

    def extract_points_from_polygon(self, p, features):
        """Extract points from a polygon

        Parameters
        ----------
        p : shapely.geometry.Polygon
            Polygon to detail
        features : dict
            Geographical features associated to the image
        Returns
        -------
        np.array
            Polygon vertices

        """
        raw_xs, raw_ys = p.exterior.xy
        xs = self.transform_x(raw_xs, features["east"], features["west"], features["width"])
        ys = self.transform_y(raw_ys, features["south"], features["north"], features["height"])
        points = np.array([[x, y] for x, y in zip(xs, ys)], dtype=np.int32)
        return np.expand_dims(points, axis=0)

    def transform_x(self, coord, east, west, width):
        """Transform abscissa from pixel to geographical coordinate

        Parameters
        ----------
        coord : list
            Coordinates to transform
        east : float
            East coordinates of the image
        west : float
            West coordinates of the image
        width : int
            Image width
        Returns
        -------
        list
            Transformed X-coordinates
        """
        return [int(width * (west-c) / (west-east)) for c in coord]

    def transform_y(self, coord, south, north, height):
        """Transform abscissa from pixel to geographical coordinate

        Parameters
        ----------
        coord : list
            Coordinates to transform
        south : float
            South coordinates of the image
        north : float
            North coordinates of the image
        height : int
            Image height

        Returns
        -------
        list
            Transformed Y-coordinates
        """
        return [int(height * (north-c) / (north-south)) for c in coord]

    def image_reference(self, image_id):
        """Return the path of the image. (Must be defined as the current class
        inheritates from mrcnn.utils.Dataset)

        Parameters
        ----------
        image_id : int
            Image ID

        Returns
        -------
        int
            Image ID
        """
        info = self.image_info[image_id]
        if info["source"] == "building":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)
