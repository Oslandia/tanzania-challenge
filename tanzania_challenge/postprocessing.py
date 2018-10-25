"""
"""

import cv2
import geojson
import json
import numpy as np
import os
from osgeo import gdal, osr
import shapely.geometry as shgeom

def flatten(data):
    if len(data.shape) == 3:
        return np.reshape(data, [-1, data.shape[2]])
    else:
        return np.reshape(data, [-1])


def count_pixels(data):
    if len(data.shape) == 3:
        df = pd.DataFrame(flatten(data), columns=["red", "green", "blue"])
        return df.groupby(["red", "green", "blue"])["red"].count()
    else:
        return pd.Series(flatten(data)).value_counts()

def count_pixels(data):
    if len(data.shape) == 3:
        df = pd.DataFrame(flatten(data), columns=["red", "green", "blue"])
        return df.groupby(["red", "green", "blue"])["red"].count()
    else:
        return pd.Series(flatten(data)).value_counts()


def masking(data, pixel):
    mask = np.zeros(data.shape[0] * data.shape[1], dtype=np.uint8)
    if len(data.shape) == 3:
        mask[np.all(flatten(data) == pixel, axis=1)] = 1
    else:
        mask[flatten(data) == pixel] = 1
    return np.reshape(mask, [data.shape[0], data.shape[1]])


def geo_project_label(data, datasource):
    driver = gdal.GetDriverByName("MEM")
    out_source = driver.Create("", data.shape[0], data.shape[1], 1, gdal.GDT_Int16)
    out_source.SetProjection(datasource.GetProjection())
    geotransform = list(datasource.GetGeoTransform())
    geotransform[0] = geotransform[0] + (datasource.RasterXSize * geotransform[1]) * 10 / 20
    geotransform[3] = geotransform[3] + (datasource.RasterYSize * geotransform[5]) * 10 / 20
    out_source.SetGeoTransform(tuple(geotransform))
    return out_source


def get_image_features(ds):
    width = ds.RasterXSize
    height = ds.RasterYSize
    gt = ds.GetGeoTransform()
    minx = gt[0]
    miny = gt[3] + height * gt[5] * 250/240
    maxx = gt[0] + width * gt[1] * 250/240
    maxy = gt[3]
    srid = int(ds.GetProjection().split('"')[-2])
    return {"west": minx, "south": miny, "east": maxx, "north": maxy,
            "srid": srid, "width": width, "height": height}


def pixel_to_coordinates(x, y, imfeatures):
    lat = int(imfeatures["west"] + (imfeatures["east"]-imfeatures["west"]) * x / imfeatures["width"])
    lon = int(imfeatures["south"] + (imfeatures["north"]-imfeatures["south"]) * y / imfeatures["height"])
    return lat, lon


def set_coordinates_as_x_y(lat, lon, srid):
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
    return y, x


def pixel_to_latlon(x, y, imfeatures):
    coordinates = pixel_to_coordinates(x, y, imfeatures)
    return set_coordinates_as_x_y(coordinates[0],
                                  coordinates[1],
                                  imfeatures["srid"])


def build_geom(building, imfeatures=None, pixel=False, min_x=2500, min_y=2500):
    feature = []
    for point in building:
        if pixel:
            feature.append((int(min_x + point[0][0]), min_y + int(point[0][1])))
        else:
            feature.append(pixel_to_latlon(point[0][0], point[0][1], imfeatures))
    feature.append(feature[0])
    return feature


def extract_geometry(mask, structure):
    """
    """
    denoised = cv2.morphologyEx(mask, cv2.MORPH_OPEN, structure)
    grown = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, structure)
    _, contours, hierarchy = cv2.findContours(grown, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    polygons = [cv2.approxPolyDP(c, epsilon=0.01*cv2.arcLength(c, closed=True), closed=True)
                for c in contours]
    return polygons


def add_polygon(polygon, class_id, score, results, geofeatures, min_x=0, min_y=0):
    """
    """
    feature = build_geom(polygon, imfeatures=geofeatures, pixel=False, min_x=min_x, min_y=min_y)
    geom = geojson.Polygon([feature])
    shape = shgeom.shape(geom)
    pixel_feature = build_geom(polygon, pixel=True, min_x=0, min_y=0)
    pixel_geom = geojson.Polygon([pixel_feature])
    pixel_shape = shgeom.shape(pixel_geom)
    predictions = np.zeros([3])
    predictions[class_id-1] = score
    return results.append([*predictions, shape.wkt, pixel_shape.wkt])


def postprocess(tile_name, input_dict):
    """
    """
    _, _, _, min_x, min_y = tile_name.split("_")
    feature_path = input_dict["-".join(("features", tile_name))].path
    with open(feature_path) as fobj:
        features = json.load(fobj)
    pred_path = feature_path.replace("features", "predicted_labels")
    with open(pred_path) as fobj:
        predictions = json.load(fobj)
    results = []
    structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = predictions["masks"]
    class_ids = predictions["class_ids"]
    scores = predictions["scores"]
    print(np.array(mask).shape)
    if len(mask) > 0:
        polygon = extract_geometry(mask, structure)
        add_polygon(polygon[0], class_ids, scores, results,
                    features, min_x, min_y)
    return results


def postprocess_tile(features, predictions, min_x, min_y):
    """
    """
    results = []
    structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    masks = np.array(predictions["masks"], dtype=np.uint8)
    class_ids = np.array(predictions["class_ids"], dtype=np.uint8)
    scores = np.array(predictions["scores"], dtype=np.float32)
    print(np.array(masks).shape)
    for mask, class_id, score in zip(masks, class_ids, scores):
        if len(mask) > 0:
            polygon = extract_geometry(mask, structure)
            for p in polygon:
                add_polygon(p, class_id, score, results,
                            features, min_x, min_y)
    return results

def postprocess_folder(tile_name):
    """
    """
    _, _, _, min_x, min_y = tile_name.split("_")
    feature_path = os.path.join("data", "open_ai_tanzania", "preprocessed",
                                "384", "testing", "features",
                                tile_name)
    with open(feature_path) as fobj:
        features = json.load(fobj)
    pred_path = feature_path.replace("features", "predicted_labels")
    with open(pred_path) as fobj:
        predictions = json.load(fobj)
    results = []
    structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = predictions["masks"]
    class_ids = predictions["class_ids"]
    scores = predictions["scores"]
    print(np.array(mask).shape)
    if len(mask) > 0:
        polygon = extract_geometry(mask, structure)
        add_polygon(polygon[0], class_ids, scores, results,
                    features, min_x, min_y)
    return results
