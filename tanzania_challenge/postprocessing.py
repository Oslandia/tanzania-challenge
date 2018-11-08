"""
"""

import cv2
import geojson
import json
import numpy as np
import os
from osgeo import gdal, osr
import shapely.geometry as shgeom


def pixel_to_coordinates(x, y, imfeatures):
    """Transform point coordinates from pixel to geographical coordinates

    Parameters
    ----------
    x : int
        Point abscissa
    y : int
        Point ordinates
    imfeatures : dict
        Image geographical description

    Returns
    -------
    tuple
        Point latitude and longitude
    """
    lat = int(imfeatures["west"] + (imfeatures["east"]-imfeatures["west"]) * x / imfeatures["width"])
    lon = int(imfeatures["north"] + (imfeatures["south"]-imfeatures["north"]) * y / imfeatures["height"])
    return lat, lon


def reproject_point(lat, lon, srid):
    """Transform coordinates into a (x,y)-compatible projection

    Parameters
    ----------
    coordinates : dict of 4 float values
        Min-x, min-y, max-x and max-y coordinates with keys 'west', 'south',
    'east', 'north'
    image_filename : str
        Image path on the file system (will be used to get the original image
    projection)
    srid : int
        Geographical projection

    Returns
    -------
    dict
        Bounding box of the image (west, south, east, north coordinates)
    """
    source = osr.SpatialReference()
    source.ImportFromEPSG(srid)
    target = osr.SpatialReference()
    target.ImportFromEPSG(4326)
    transform = osr.CoordinateTransformation(source, target)
    x, y = transform.TransformPoint(lat, lon)[:2]
    return x, y


def pixel_to_latlon(x, y, imfeatures):
    """Transform point coordinates from pixel to geographical coordinates in
    the accurate geographical projection

    Parameters
    ----------
    x : int
        Point abscissa
    y : int
        Point ordinates
    imfeatures : dict
        Image geographical description

    Returns
    -------
    tuple
        Point latitude and longitude
    """
    coordinates = pixel_to_coordinates(x, y, imfeatures)
    return reproject_point(coordinates[0], coordinates[1], imfeatures["srid"])


def build_geom(building, imfeatures=None, xy=True, pixel=False,
               min_x=2500, min_y=2500):
    """

    Parameters
    ----------
    building : list
        Pixel points that represents a polygon on the image prediction mask
    imfeatures : dict
        Geographical features
    xy : bool
        If true, points are expressed as (x, y), or (y, x) otherwise
    pixel : bool
        If true, points are expressed as pixel coordinates in the raw image,
    otherwise they are expressed in geographical coordinates
    min_x : int
        Horizontal image pixel shift, with respect to the raw image (from the
    left border)
    min_y : int
        Vertical image pixel shift, with respect to the raw image (from the
    upper border)

    Returns
    -------
    list
        List of geographical points expressed in the accurate projection
    """
    feature = []
    for point in building:
        if pixel:
            feature.append((int(min_x) + int(point[0][0]), int(min_y) + int(point[0][1])))
        else:
            if xy:
                feature.append(pixel_to_latlon(point[0][0], point[0][1],
                                               imfeatures))
            else:
                feature.append(pixel_to_latlon(point[0][1], point[0][0],
                                               imfeatures))
    feature.append(feature[0])
    return feature


def extract_geometry(mask, structure):
    """Extract polygons from a boolean mask with the help of OpenCV utilities

    Parameters
    ----------
    mask : numpy.array
        Image mask where to find polygons
    structure : numpy.array
        Artifact used for image morphological transformation

    Returns
    -------
    list
        List of polygons contained in the mask
    """
    denoised = cv2.morphologyEx(mask, cv2.MORPH_OPEN, structure)
    grown = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, structure)
    _, contours, hierarchy = cv2.findContours(grown, cv2.RETR_TREE,
                                              cv2.CHAIN_APPROX_SIMPLE)
    polygons = [cv2.approxPolyDP(c,
                                 epsilon=0.01*cv2.arcLength(c, closed=True),
                                 closed=True)
                for c in contours]
    return polygons


def add_polygon(polygon, class_id, score, results, geofeatures, min_x=0, min_y=0):
    """Prepare the polygon characterization for Open AI Tanzania challenge

    Results must be in the following format:
    | building_id | conf_completed | conf_incompleted | conf_foundation |
    coords_geo | coords_pixel |
    |-----------|-----------|-----------|-----------|-----------|-----------|
    | 1         |  0.3      | 0.5       | 0.2       | POLYGON((39.0, -5.3),
    ...) | POLYGON((324, 4100), ) |

    where `conf_*` refer to the probability of occurrence of each type of
    building in the dataset, and `coords_*` respectively the buildings
    expressed as geographical and pixel coordinates

    Parameters
    ----------
    polygon : list
        Polygons extracted with OpenCV utilities (list of list of points)
    class_id : numpy.array
        Class of each detected polygon (completed, incompleted or foundation)
    score : numpy.array
        Detection score of each detected polygon (associated to the accurate
    class)
    geofeatures : dict
        Image geographical features
    min_x : int
        Horizontal image pixel shift, with respect to the raw image (from the
    left border)
    min_y : int
        Vertical image pixel shift, with respect to the raw image (from the
    upper border)

    Returns
    -------
    list
        Polygon list enriched with current polygon description
    """
    feature = build_geom(polygon, imfeatures=geofeatures, pixel=False, min_x=min_x, min_y=min_y)
    geom = geojson.Polygon([feature])
    shape = shgeom.shape(geom)
    pixel_feature = build_geom(polygon, pixel=True, min_x=min_x, min_y=min_y)
    pixel_geom = geojson.Polygon([pixel_feature])
    pixel_shape = shgeom.shape(pixel_geom)
    predictions = np.zeros([3])
    predictions[class_id-1] = score
    return results.append([*predictions, shape.wkt, pixel_shape.wkt])


def postprocess_tile(features, predictions, min_x, min_y):
    """Post-process a tile, *i.e.* transform prediction results into
    exploitable format

    Parameters
    ----------
    features : dict
        Image geographical features
    predictions : dict
        Instance segmentation algorithm results (contained a ̀masks` key that
    describes predicted buildings on the image, a `class_ids` key for giving
    the detected building corresponding classes, and a `scores` key for giving
    the prediction scores of each detected building)
    min_x : int
        Horizontal image pixel shift, with respect to the raw image (from the
    left border)
    min_y : int
        Vertical image pixel shift, with respect to the raw image (from the
    upper border)

    Results
    -------
    list
        List of detected buildings as polygons
    """
    results = []
    structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    masks = np.array(predictions["masks"], dtype=np.uint8)
    class_ids = np.array(predictions["class_ids"], dtype=np.uint8)
    scores = np.array(predictions["scores"], dtype=np.float32)
    for mask, class_id, score in zip(masks, class_ids, scores):
        if len(mask) > 0:
            polygon = extract_geometry(mask, structure)
            for p in polygon:
                try:

                    add_polygon(p, class_id, score, results,
                                features, min_x, min_y)
                except ValueError as e:
                    print("Can't add a polygon: ", e)
                    continue
    return results
