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



def get_mapnik_projection(srid):
    """
    """
    url = "http://spatialreference.org/ref/epsg/{srid}/mapnikpython/"
    response = requests.get(url.format(srid=srid))
    response_str = response.content.decode()
    mapnik_projection = response_str.split("\n")[1].split(" = ")[-1]
    return mapnik_projection[1:-1]


def RGBToHTMLColor(rgb_tuple):
    """Convert an (R, G, B) tuple to #RRGGBB

    Parameters
    ----------
    rgb_list : list
        List of red, green, blue pixel values

    Returns
    -------
    str
        HTML-version of color
    """
    return '#%02x%02x%02x' % tuple(rgb_tuple)


def generate_raster(output_path, area, features, classes):
    """Generate a raster through requesting a PostGIS database with Mapnik

    Parameters
    ----------
    output_path : str
        Output raster path on the file system
    area : str
        Base name of the image file
    features : dict
        Geographical coordinates of area of interest (west, south, east,
    north), corresponding geographical projection (SRID) and resulting image
    size (width=height)
    classes : dict
        Class name and colors (pixels in [R,G,B]-format, for each class)
    """
    mapnik_projection = get_mapnik_projection(features["srid"])
    m = mapnik.Map(features["width"], features["height"], mapnik_projection)
    m.background = mapnik.Color(RGBToHTMLColor(classes['background']))
    s = mapnik.Style()
    r = mapnik.Rule()
    symbolizer = mapnik.PolygonSymbolizer()
    symbolizer.fill = mapnik.Color(RGBToHTMLColor(classes['Complete']))
    r.symbols.append(symbolizer)
    s.rules.append(r)
    for key, value in classes.items():
        if not key == 'background':
            symbolizer = mapnik.PolygonSymbolizer()
            symbolizer.fill = mapnik.Color(RGBToHTMLColor(value))
            r.symbols.append(symbolizer)
            r.filter = mapnik.Filter("[condition] = '{}'".format(key))
            s.rules.append(r)
    m.append_style('building_styles', s)
    
    subquery = ("(SELECT condition, wkb_geometry FROM {}) AS building"
                "").format(area)
    postgis_params = {'host': confparser.get("database", "host"),
                      'port': confparser.get("database", "port"),
                      'user': confparser.get("database", "user"),
                      'password': confparser.get("database", "password"),
                      'dbname': confparser.get("database", "dbname"),
                      'table': subquery,
                      'geometry_field': "wkb_geometry",
                      'srid': features["srid"],
                      'extent_from_subquery': True}
    ds = mapnik.PostGIS(**postgis_params)

    layer = mapnik.Layer('buildings')
    layer.datasource = ds
    layer.srs = mapnik_projection
    layer.styles.append('building_styles')
    m.layers.append(layer)
    m.zoom_to_box(mapnik.Box2d(features['west'], features['south'],
                               features['east'], features['north']))
    mapnik.render_to_file(m, output_path, 'png')
