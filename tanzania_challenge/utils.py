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
