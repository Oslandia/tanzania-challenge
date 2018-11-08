"""This module provides some utility functions for running Open AI Tanzania
challenge pipeline

"""

from configparser import ConfigParser
import mapnik
from osgeo import gdal
import requests

confparser = ConfigParser()
confparser.read("config.ini")

def get_image_features(filename):
    """Retrieve geotiff image features with GDAL

    Use the `GetGeoTransform` method, that provides the following values:
      + East/West location of Upper Left corner
      + East/West pixel resolution
      + 0.0
      + North/South location of Upper Left corner
      + 0.0
      + North/South pixel resolution

    See GDAL documentation (https://www.gdal.org/gdal_tutorial.html)

    Parameters
    ----------
    filename : str
        Name of the image file from which coordinates are extracted

    Returns
    -------
    dict
        Bounding box of the image (west, south, east, north coordinates), srid,
    and size (in pixels)

    """
    ds = gdal.Open(filename)
    width = ds.RasterXSize
    height = ds.RasterYSize
    gt = ds.GetGeoTransform()
    minx = gt[0]
    miny = gt[3] + height * gt[5]
    maxx = gt[0] + width * gt[1]
    maxy = gt[3]
    srid = int(ds.GetProjection().split('"')[-2])
    ds = None
    return {"west": minx, "south": miny, "east": maxx, "north": maxy,
            "srid": srid, "width": width, "height": height}


def get_tile_features(tile_width, tile_height, tile_min_x, tile_min_y, img_features):
    """Retrieve tile features starting from original raw image features

    Parameters
    ----------
    tile_width : int
        Tile width, in pixel
    tile_height : int
        Tile height, in pixel
    tile_min_x : int
        East coordinates of the tile within the raw image, expressed in pixel
    tile_min_y : int
        North coordinates of the tile within the raw image, expressed in pixel
    img_features : dict
        Raw image geo-features (coordinates, SRID, width and height)

    Returns
    -------
    dict
        Bounding box of the image (west, south, east, north coordinates), srid,
    and size (in pixels)

    """
    tile_features = {"width": tile_width,
                     "height": tile_height,
                     "srid": img_features["srid"]}
    tile_features["west"] = (img_features["west"]
                             + tile_min_x
                             * (img_features["east"] - img_features["west"])
                             / img_features["width"])
    tile_features["east"] = (img_features["west"]
                             + (tile_min_x + tile_width)
                             * (img_features["east"] - img_features["west"])
                             / img_features["width"])
    tile_features["north"] = (img_features["north"]
                             + tile_min_y
                             * (img_features["south"] - img_features["north"])
                             / img_features["height"])
    tile_features["south"] = (img_features["north"]
                             + (tile_min_y + tile_height)
                             * (img_features["south"] - img_features["north"])
                             / img_features["height"])
    return tile_features
