"""This module is the project main module, where all pipeline tasks are run
until completion.

To run a given Luigi Task:

```
luigi --workers <nb-workers> --local-scheduler --module tanzania_challenge.main
<Task> --filename <Filename> --kwargs
```

18/09/21 update: the last pipeline task is `MergeAllLabelRasters`, it gives
labelled version of raw images, with complete buildings in green, incomplete
buildings in yellow and foundations in red. You may run it with the parameters of
your choice. As an example:

```
luigi --workers 3 --local-scheduler --module tanzania_challenge.main
MergeAllLabelRasters --datapath ./data/open_ai_tanzania --tile-size 10000
--background-color 0 0 0 --complete-color 50 200 50 --incomplete-color 200 200
50 --foundation-color 200 50 50
```

"""

import geopandas as gpd
import json
import luigi
from luigi.contrib.postgres import CopyToTable, PostgresQuery
import os
from osgeo import gdal
import psycopg2
import sh
import subprocess

from tanzania_challenge import utils


class GenerateSubTile(luigi.Task):
    """Generate subtiled raw images, in order to handle them more efficiently
    in memory

    Attributes
    ----------
    datapath : str
        Path towards the Tanzania challenge data
    filename : str
        Name of the area of interest, *e.g.* `grid_001`
    min_x : int
        Tile upper-left corner abscissa (north coordinate)
    min_y : int
        Tile upper-left corner ordinate (west coordinate)
    tile_width : int
        Number of pixels that must be considered in the west-east direction
    tile_height : int
        Number of pixels that must be considered in the north-south direction

    """
    datapath = luigi.Parameter(default="./data/open_ai_tanzania")
    dataset = luigi.Parameter(default="training")
    filename = luigi.Parameter()
    min_x = luigi.IntParameter()
    min_y = luigi.IntParameter()
    tile_size = luigi.IntParameter(default=5000)
    tile_width = luigi.IntParameter(default=5000)
    tile_height = luigi.IntParameter(default=5000)

    @property
    def input_dataset(self):
        return "testing" if self.dataset == "testing" else "training"

    def output(self):
        output_path = os.path.join(self.datapath, "preprocessed",
                                   str(self.tile_size),
                                   self.dataset, "images")
        os.makedirs(output_path, exist_ok=True)
        output_filename_suffix = "_{}_{}_{}_{}.tif".format(self.tile_width,
                                                           self.tile_height,
                                                           self.min_x,
                                                           self.min_y)
        output_filename = self.filename + output_filename_suffix
        return luigi.LocalTarget(os.path.join(output_path, output_filename))

    def run(self):
        input_path = os.path.join(self.datapath, "input", self.input_dataset,
                                  "images", self.filename + ".tif")
        gdal_translate_args = ['-srcwin',
                               self.min_x, self.min_y,
                               self.tile_width, self.tile_height,
                               input_path,
                               self.output().path]
        sh.gdal_translate(gdal_translate_args)


class GenerateAllSubTiles(luigi.Task):
    """

    Attributes
    ----------
    datapath : str
        Path towards the Tanzania challenge data
    filename : str
        Name of the area of interest, *e.g.* `grid_001`
    tile_size : int
        Number of pixels that must be considered in both direction (east-west,
    north-south) in tile definition. This constraint is relaxed when
    considering border tiles (on east and south borders, especially).

    """
    datapath = luigi.Parameter(default="./data/open_ai_tanzania")
    dataset = luigi.Parameter(default="testing")
    filename = luigi.Parameter()
    tile_size = luigi.IntParameter(default=5000)

    def requires(self):
        task_in = {}
        ds = gdal.Open(os.path.join(self.datapath, "input", self.dataset,
                                    "images", self.filename + ".tif"))
        xsize = ds.RasterXSize
        ysize = ds.RasterYSize
        for x in range(0, xsize, self.tile_size):
            tile_width = min(xsize - x, self.tile_size)
            for y in range(0, ysize, self.tile_size):
                task_id = str(x) + "-" + str(y)
                tile_height = min(ysize - y, self.tile_size)
                task_in[task_id] = GenerateSubTile(self.datapath,
                                                   self.dataset,
                                                   self.filename,
                                                   x, y, self.tile_size,
                                                   tile_width, tile_height)
        ds = None
        return task_in


class GetImageFeatures(luigi.Task):
    """Retrieve some basic geographical features of images and save them into a
    geojson file.

    Amongst the considered features, we get:
    - geographical coordinates (west, south, east, north)
    - image dimension (width, height) in pixels
    - geographical projection (SRID)

    Attributes
    ----------
    datapath : str
        Path towards the Tanzania challenge data
    filename : str
        Name of the area of interest, *e.g.* `grid_001`

    """
    datapath = luigi.Parameter(default="./data/open_ai_tanzania")
    filename = luigi.Parameter()
    dataset = luigi.Parameter(default="training")

    @property
    def input_dataset(self):
        return "testing" if self.dataset == "testing" else "training"

    def output(self):
        output_path = os.path.join(self.datapath, "input",
                                   self.dataset, "features")
        os.makedirs(output_path, exist_ok=True)
        output_filename = os.path.join(output_path, self.filename + ".json")
        return luigi.LocalTarget(output_filename)

    def run(self):
        input_filename = os.path.join(self.datapath, "input", self.input_dataset,
                                      "images", self.filename + ".tif")
        coordinates = utils.get_image_features(input_filename)
        with self.output().open('w') as fobj:
            json.dump(coordinates, fobj)


class GetTileFeatures(luigi.Task):
    """Retrieve some basic geographical features of tiled images and save them into a
    geojson file. As we consider here tiles instead of raw images, we have to
    pay a particular attention on the output path.

    Amongst the considered features, we get:
    - geographical coordinates (west, south, east, north)
    - image dimension (width, height) in pixels
    - geographical projection (SRID)

    Attributes
    ----------
    datapath : str
        Path towards the Tanzania challenge data
    filename : str
        Name of the area of interest, *e.g.* `grid_001`
    min_x : int
        Tile upper-left corner abscissa (north coordinate)
    min_y : int
        Tile upper-left corner ordinate (west coordinate)
    tile_width : int
        Number of pixels that must be considered in the west-east direction
    tile_height : int
        Number of pixels that must be considered in the north-south direction

    """
    datapath = luigi.Parameter(default="./data/open_ai_tanzania")
    dataset = luigi.Parameter(default="training")
    filename = luigi.Parameter()
    min_x = luigi.IntParameter()
    min_y = luigi.IntParameter()
    tile_size = luigi.IntParameter(default=5000)
    tile_width = luigi.IntParameter(default=5000)
    tile_height = luigi.IntParameter(default=5000)

    def requires(self):
        return GenerateSubTile(self.datapath, self.dataset,
                               self.filename,
                               self.min_x, self.min_y,
                               self.tile_size,
                               self.tile_width, self.tile_height)

    def output(self):
        output_path = os.path.join(self.datapath, "preprocessed",
                                   str(self.tile_size),
                                   self.dataset, "features")
        os.makedirs(output_path, exist_ok=True)
        output_filename_suffix = "_{}_{}_{}_{}.json".format(self.tile_width,
                                                            self.tile_height,
                                                            self.min_x,
                                                            self.min_y)
        output_filename = self.filename + output_filename_suffix
        return luigi.LocalTarget(os.path.join(output_path, output_filename))

    def run(self):
        coordinates = utils.get_image_features(self.input().path)
        with self.output().open('w') as fobj:
            json.dump(coordinates, fobj)


class GetAllTileFeatures(luigi.Task):
    """

    Attributes
    ----------
    datapath : str
        Path towards the Tanzania challenge data
    filename : str
        Name of the area of interest, *e.g.* `grid_001`
    tile_size : int
        Number of pixels that must be considered in both direction (east-west,
    north-south) in tile definition. This constraint is relaxed when
    considering border tiles (on east and south borders, especially).

    """
    datapath = luigi.Parameter(default="./data/open_ai_tanzania")
    dataset = luigi.Parameter(default="testing")
    filename = luigi.Parameter()
    tile_size = luigi.IntParameter(default=5000)

    def requires(self):
        task_in = {}
        ds = gdal.Open(os.path.join(self.datapath, "input", self.dataset,
                                    "images", self.filename + ".tif"))
        xsize = ds.RasterXSize
        ysize = ds.RasterYSize
        for x in range(0, xsize, self.tile_size):
            tile_width = min(xsize - x, self.tile_size)
            for y in range(0, ysize, self.tile_size):
                task_id = str(x) + "-" + str(y)
                tile_height = min(ysize - y, self.tile_size)
                task_in[task_id] = GetTileFeatures(self.datapath,
                                                   self.dataset,
                                                   self.filename,
                                                   x, y, self.tile_size,
                                                   tile_width, tile_height)
        ds = None
        return task_in


class StoreLabelsToDatabase(luigi.Task):
    """Store image labels to a database, considering that the input format is
    `geojson`. We use `ogr2ogr` program, and consider the task as accomplished
    after saving a `txt` file within tanzania data folder.

    Attributes
    ----------
    datapath : str
        Path towards the Tanzania challenge data
    filename : str
        Name of the area of interest, *e.g.* `grid_001`

    """
    datapath = luigi.Parameter(default="./data/open_ai_tanzania")
    filename = luigi.Parameter()

    def requires(self):
        return GetImageFeatures(self.datapath, self.filename)

    def output(self):
        output_path = os.path.join(self.datapath, "input", "training", "ogr")
        os.makedirs(output_path, exist_ok=True)
        filename = self.filename + "-task-ogr2ogr.txt"
        output_filename = os.path.join(output_path, filename)
        return luigi.LocalTarget(output_filename)

    def run(self):
        with self.input().open('r') as fobj:
            coordinates = json.load(fobj)
        label_filename = os.path.join(self.datapath, "input", "training",
                                      "labels", self.filename + ".geojson")
        dbname = utils.confparser.get("database", "dbname")
        user = utils.confparser.get("database", "user")
        password = utils.confparser.get("database", "password")
        port = utils.confparser.get("database", "port")
        host = utils.confparser.get("database", "host")
        conn_string = ('PG:dbname={dbname} user={user} password={pw} '
                       'port={port} host={host}').format(dbname=dbname,
                                                         user=user,
                                                         pw=password,
                                                         port=port,
                                                         host=host)
        ogr2ogr_args = ['-f', 'PostGreSQL',
                        conn_string,
                        os.path.join(self.datapath, "labels",
                                     self.filename + ".geojson"),
                        '-t_srs', "EPSG:{}".format(coordinates["srid"]),
                        '-nln', self.filename,
                        '-overwrite']
        with self.output().open("w") as fobj:
            sh.ogr2ogr(ogr2ogr_args)
            fobj.write(("ogr2ogr used file {} to insert OSM data "
                        "into {} database").format(label_filename, "tanzania"))


class ExtractTileItems(luigi.Task):
    """
    """
    datapath = luigi.Parameter(default="./data/open_ai_tanzania")
    dataset = luigi.Parameter(default="training")
    filename = luigi.Parameter()
    min_x = luigi.IntParameter()
    min_y = luigi.IntParameter()
    tile_size = luigi.IntParameter(default=5000)
    tile_width = luigi.IntParameter(default=5000)
    tile_height = luigi.IntParameter(default=5000)

    def requires(self):
        return {"db": StoreLabelsToDatabase(self.datapath, self.filename),
                "features": GetTileFeatures(self.datapath, self.dataset,
                                            self.filename,
                                            self.min_x, self.min_y,
                                            self.tile_size,
                                            self.tile_width, self.tile_height)}

    def output(self):
        output_path = os.path.join(self.datapath, "preprocessed",
                                   str(self.tile_size),
                                   self.dataset, "items")
        os.makedirs(output_path, exist_ok=True)
        output_filename_suffix = "_{}_{}_{}_{}.json".format(self.tile_width,
                                                            self.tile_height,
                                                            self.min_x,
                                                            self.min_y)
        output_filename = self.filename + output_filename_suffix
        return luigi.LocalTarget(os.path.join(output_path, output_filename))

    def run(self):
        with self.input()["features"].open("r") as fobj:
            features = json.load(fobj)
        query = ("WITH bbox AS ("
                 "SELECT ST_MakeEnvelope("
                 "{west}, {south}, {east}, {north}, {srid}) AS geom"
                 ") "
                 "SELECT condition, (st_dump("
                 "st_intersection(st_makevalid(wkb_geometry), bbox.geom))"
                 ").geom::geometry(Polygon, {srid}) "
                 "FROM {table} JOIN bbox "
                 "ON st_intersects(st_makevalid(wkb_geometry), bbox.geom)"
                 ";").format(table=self.filename, west=features["west"],
                             south=features["south"], east=features["east"],
                             north=features["north"], srid=features["srid"])
        config = utils.confparser["database"]
        connection_string = ("dbname={dbname} host={host} port={port} "
                             "user={user} password={password}"
                             "").format(dbname=config["dbname"],
                                        host=config["host"],
                                        port=config["port"],
                                        user=config["user"],
                                        password=config["password"])
        connection = psycopg2.connect(connection_string)
        cursor = connection.cursor()
        cursor.execute(query)
        rset = cursor.fetchall()
        res = {}
        for i, x in enumerate(rset):
            if not x[1] is None:
                res[i]={"condition": x[0], "geom": x[1]}
        with self.output().open('w') as fobj:
            json.dump(res, fobj)


class ExtractAllTileItems(luigi.Task):
    """

    Attributes
    ----------
    datapath : str
        Path towards the Tanzania challenge data
    filename : str
        Name of the area of interest, *e.g.* `grid_001`
    tile_size : int
        Number of pixels that must be considered in both direction (east-west,
    north-south) in tile definition. This constraint is relaxed when
    considering border tiles (on east and south borders, especially).

    """
    datapath = luigi.Parameter(default="./data/open_ai_tanzania")
    dataset = luigi.Parameter(default="training")
    filename = luigi.Parameter()
    tile_size = luigi.IntParameter(default=5000)

    def requires(self):
        task_in = {}
        ds = gdal.Open(os.path.join(self.datapath, "input", self.dataset,
                                    "images", self.filename + ".tif"))
        xsize = ds.RasterXSize
        ysize = ds.RasterYSize
        for x in range(0, xsize, self.tile_size):
            tile_width = min(xsize - x, self.tile_size)
            for y in range(0, ysize, self.tile_size):
                task_id = str(x) + "-" + str(y)
                tile_height = min(ysize - y, self.tile_size)
                task_in[task_id] = ExtractTileItems(self.datapath,
                                                    self.dataset,
                                                    self.filename,
                                                    x, y, self.tile_size,
                                                    tile_width, tile_height)
        ds = None
        return task_in

    def complete(self):
        return False


class ExtractValidTileItems(luigi.Task):
    """

    Attributes
    ----------
    datapath : str
        Path towards the Tanzania challenge data
    filename : str
        Name of the area of interest, *e.g.* `grid_001`
    tile_size : int
        Number of pixels that must be considered in both direction (east-west,
    north-south) in tile definition. This constraint is relaxed when
    considering border tiles (on east and south borders, especially).

    """
    datapath = luigi.Parameter(default="./data/open_ai_tanzania")
    dataset = luigi.Parameter(default="training")
    filename = luigi.Parameter()
    tile_size = luigi.IntParameter(default=5000)

    def requires(self):
        task_in = {}
        building_inventory = ""
        tile_name = "{filename}_{tile_width}_{tile_height}_{min_x}_{min_y}"
        raster_name = os.path.join(self.datapath, "input", self.dataset,
                                   "images", self.filename + ".tif")
        features = utils.get_image_features(raster_name)
        x_offset = (features["east"] - features["west"]) * self.tile_size / features["width"]
        y_offset = (features["north"] - features["south"]) * self.tile_size / features["height"]

        for x in range(0, features["width"], self.tile_size):
            cx = features["west"] + (features["east"] - features["west"]) * x / features["width"]
            tile_width = min(features["width"] - x, self.tile_size)
            for y in range(0, features["height"], self.tile_size):
                cy = features["north"] + (features["south"] - features["north"]) * y / features["height"]
                task_id = str(x) + "-" + str(y)
                tile_height = min(features["height"] - y, self.tile_size)
                nb_buildings = self.valid_tile(cx, cy, x_offset, y_offset,
                                               features["srid"])
                filled_tile_name = tile_name.format(filename=self.filename,
                                                    tile_width=tile_width,
                                                    tile_height=tile_height,
                                                    min_x=x, min_y=y)
                building_inventory += (filled_tile_name + " "
                                       + str(nb_buildings) + "\n")
                if nb_buildings > 0:
                    task_in[task_id] = ExtractTileItems(self.datapath,
                                                        self.dataset,
                                                        self.filename,
                                                        x, y, self.tile_size,
                                                        tile_width,
                                                        tile_height)
        inventory_filename = os.path.join(self.datapath, "preprocessed",
                                          "buildings_per_tile.txt")
        fobj = open(inventory_filename, "a+")
        fobj.write(building_inventory)
        fobj.close()
        return task_in

    def valid_tile(self, x, y, x_offset, y_offset, srid):
        """
        """
        query = ("WITH bbox AS ("
                 "SELECT ST_MakeEnvelope("
                 "{west}, {south}, {east}, {north}, {srid}) AS geom"
                 ") "
                 "SELECT count("
                 "st_intersection(st_makevalid(wkb_geometry), bbox.geom)) "
                 "FROM {table} JOIN bbox "
                 "ON st_intersects(st_makevalid(wkb_geometry), bbox.geom)"
                 ";").format(table=self.filename, west=x, south=y-y_offset,
                             east=x+x_offset, north=y, srid=srid)
        config = utils.confparser["database"]
        connection_string = ("dbname={dbname} host={host} port={port} "
                             "user={user} password={password}"
                             "").format(dbname=config["dbname"],
                                        host=config["host"],
                                        port=config["port"],
                                        user=config["user"],
                                        password=config["password"])
        connection = psycopg2.connect(connection_string)
        cursor = connection.cursor()
        cursor.execute(query)
        rset = cursor.fetchone()
        return rset[0]

    def complete(self):
        return False
