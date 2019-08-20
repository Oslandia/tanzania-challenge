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
from multiprocessing import Pool
import os
from osgeo import gdal
import pandas as pd
import psycopg2
import sh
import shapely.wkt as swkt

from tanzania_challenge import utils, train, inference, postprocessing


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
        output_path = os.path.join(
            self.datapath, "preprocessed", str(self.tile_size), self.dataset, "images"
        )
        os.makedirs(output_path, exist_ok=True)
        output_filename_suffix = "_{}_{}_{}_{}.tif".format(
            self.tile_width, self.tile_height, self.min_x, self.min_y
        )
        output_filename = self.filename + output_filename_suffix
        return luigi.LocalTarget(os.path.join(output_path, output_filename))

    def run(self):
        input_path = os.path.join(
            self.datapath, "input", self.dataset, "images", self.filename + ".tif"
        )
        gdal_translate_args = [
            "-srcwin",
            self.min_x,
            self.min_y,
            self.tile_width,
            self.tile_height,
            input_path,
            self.output().path,
        ]
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
        ds = gdal.Open(
            os.path.join(
                self.datapath, "input", self.dataset, "images", self.filename + ".tif"
            )
        )
        xsize = ds.RasterXSize
        ysize = ds.RasterYSize
        for x in range(0, xsize, self.tile_size):
            tile_width = min(xsize - x, self.tile_size)
            for y in range(0, ysize, self.tile_size):
                task_id = str(x) + "-" + str(y)
                tile_height = min(ysize - y, self.tile_size)
                task_in[task_id] = GenerateSubTile(
                    self.datapath,
                    self.dataset,
                    self.filename,
                    x,
                    y,
                    self.tile_size,
                    tile_width,
                    tile_height,
                )
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

    def output(self):
        output_path = os.path.join(self.datapath, "input", self.dataset, "features")
        os.makedirs(output_path, exist_ok=True)
        output_filename = os.path.join(output_path, self.filename + ".json")
        return luigi.LocalTarget(output_filename)

    def run(self):
        input_filename = os.path.join(
            self.datapath, "input", self.dataset, "images", self.filename + ".tif"
        )
        coordinates = utils.get_image_features(input_filename)
        with self.output().open("w") as fobj:
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
        return {
            "tile": GenerateSubTile(
                self.datapath,
                self.dataset,
                self.filename,
                self.min_x,
                self.min_y,
                self.tile_size,
                self.tile_width,
                self.tile_height,
            ),
            "img_features": GetImageFeatures(
                self.datapath, self.filename, self.dataset
            ),
        }

    def output(self):
        output_path = os.path.join(
            self.datapath, "preprocessed", str(self.tile_size), self.dataset, "features"
        )
        os.makedirs(output_path, exist_ok=True)
        output_filename_suffix = "_{}_{}_{}_{}.json".format(
            self.tile_width, self.tile_height, self.min_x, self.min_y
        )
        output_filename = self.filename + output_filename_suffix
        return luigi.LocalTarget(os.path.join(output_path, output_filename))

    def run(self):
        with open(self.input()["img_features"].path) as fobj:
            img_features = json.load(fobj)
        coordinates = utils.get_tile_features(
            self.tile_width, self.tile_height, self.min_x, self.min_y, img_features
        )
        with self.output().open("w") as fobj:
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
        ds = gdal.Open(
            os.path.join(
                self.datapath, "input", self.dataset, "images", self.filename + ".tif"
            )
        )
        xsize = ds.RasterXSize
        ysize = ds.RasterYSize
        for x in range(0, xsize, self.tile_size):
            tile_width = min(xsize - x, self.tile_size)
            for y in range(0, ysize, self.tile_size):
                task_id = str(x) + "-" + str(y)
                tile_height = min(ysize - y, self.tile_size)
                task_in[task_id] = GetTileFeatures(
                    self.datapath,
                    self.dataset,
                    self.filename,
                    x,
                    y,
                    self.tile_size,
                    tile_width,
                    tile_height,
                )
        ds = None
        return task_in


class GetTileFeaturesFromFolder(luigi.Task):
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
    tile_size = luigi.IntParameter(default=384)
    folder = luigi.Parameter()

    def requires(self):
        task_in = {}
        filenames = os.listdir(
            os.path.join(
                self.datapath,
                "preprocessed",
                str(self.tile_size),
                self.dataset,
                self.folder,
            )
        )
        for f in filenames:
            if self.dataset in ["training", "validation"]:
                f1, f2, tile_width, tile_height, x, y = f.split(".")[0].split("_")
                filename = "_".join([f1, f2])
            else:
                filename, tile_width, tile_height, x, y = f.split(".")[0].split("_")
            task_in[f] = GetTileFeatures(
                self.datapath,
                self.dataset,
                filename,
                int(x),
                int(y),
                int(self.tile_size),
                int(tile_width),
                int(tile_height),
            )
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
        with self.input().open("r") as fobj:
            coordinates = json.load(fobj)
        label_filename = os.path.join(
            self.datapath, "input", "training", "labels", self.filename + ".geojson"
        )
        dbname = utils.confparser.get("database", "dbname")
        user = utils.confparser.get("database", "user")
        password = utils.confparser.get("database", "password")
        port = utils.confparser.get("database", "port")
        host = utils.confparser.get("database", "host")
        conn_string = (
            "PG:dbname={dbname} user={user} password={pw} " "port={port} host={host}"
        ).format(dbname=dbname, user=user, pw=password, port=port, host=host)
        ogr2ogr_args = [
            "-f",
            "PostGreSQL",
            conn_string,
            os.path.join(
                self.datapath, "input", "training", "labels", self.filename + ".geojson"
            ),
            "-t_srs",
            "EPSG:{}".format(coordinates["srid"]),
            "-nln",
            self.filename,
            "-overwrite",
        ]
        with self.output().open("w") as fobj:
            sh.ogr2ogr(ogr2ogr_args)
            fobj.write(
                ("ogr2ogr used file {} to insert OSM data " "into {} database").format(
                    label_filename, "tanzania"
                )
            )


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
        return {
            "db": StoreLabelsToDatabase(self.datapath, self.filename),
            "features": GetTileFeatures(
                self.datapath,
                self.dataset,
                self.filename,
                self.min_x,
                self.min_y,
                self.tile_size,
                self.tile_width,
                self.tile_height,
            ),
        }

    def output(self):
        output_path = os.path.join(
            self.datapath, "preprocessed", str(self.tile_size), self.dataset, "items"
        )
        os.makedirs(output_path, exist_ok=True)
        output_filename_suffix = "_{}_{}_{}_{}.json".format(
            self.tile_width, self.tile_height, self.min_x, self.min_y
        )
        output_filename = self.filename + output_filename_suffix
        return luigi.LocalTarget(os.path.join(output_path, output_filename))

    def run(self):
        with self.input()["features"].open("r") as fobj:
            features = json.load(fobj)
        query = (
            "WITH bbox AS ("
            "SELECT ST_MakeEnvelope("
            "{west}, {south}, {east}, {north}, {srid}) AS geom"
            ") "
            "SELECT condition, (st_dump("
            "st_intersection(st_makevalid(wkb_geometry), bbox.geom))"
            ").geom::geometry(Polygon, {srid}) "
            "FROM {table} JOIN bbox "
            "ON st_intersects(st_makevalid(wkb_geometry), bbox.geom)"
            ";"
        ).format(
            table=self.filename,
            west=features["west"],
            south=features["south"],
            east=features["east"],
            north=features["north"],
            srid=features["srid"],
        )
        config = utils.confparser["database"]
        connection_string = (
            "dbname={dbname} host={host} port={port} "
            "user={user} password={password}"
            ""
        ).format(
            dbname=config["dbname"],
            host=config["host"],
            port=config["port"],
            user=config["user"],
            password=config["password"],
        )
        connection = psycopg2.connect(connection_string)
        cursor = connection.cursor()
        cursor.execute(query)
        rset = cursor.fetchall()
        res = {}
        for i, x in enumerate(rset):
            if not x[1] is None:
                res[i] = {"condition": x[0], "geom": x[1]}
        with self.output().open("w") as fobj:
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
    tile_size = luigi.IntParameter(default=5000)

    @property
    def filenames(self):
        image_dir = os.path.join(self.datapath, "input", self.dataset, "images")
        return [fname.split(".")[0] for fname in os.listdir(image_dir)]

    def requires(self):
        for filename in self.filenames:
            ds = gdal.Open(
                os.path.join(
                    self.datapath, "input", self.dataset, "images", filename + ".tif"
                )
            )
            xsize = ds.RasterXSize
            ysize = ds.RasterYSize
            for x in range(0, xsize, self.tile_size):
                tile_width = min(xsize - x, self.tile_size)
                for y in range(0, ysize, self.tile_size):
                    tile_height = min(ysize - y, self.tile_size)
                    yield ExtractTileItems(
                        self.datapath,
                        self.dataset,
                        filename,
                        x,
                        y,
                        self.tile_size,
                        tile_width,
                        tile_height,
                    )
            ds = None

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
        raster_name = os.path.join(
            self.datapath, "input", self.dataset, "images", self.filename + ".tif"
        )
        features = utils.get_image_features(raster_name)
        x_offset = (
            (features["east"] - features["west"]) * self.tile_size / features["width"]
        )
        y_offset = (
            (features["north"] - features["south"])
            * self.tile_size
            / features["height"]
        )

        for x in range(0, features["width"], self.tile_size):
            cx = (
                features["west"]
                + (features["east"] - features["west"]) * x / features["width"]
            )
            tile_width = min(features["width"] - x, self.tile_size)
            for y in range(0, features["height"], self.tile_size):
                cy = (
                    features["north"]
                    + (features["south"] - features["north"]) * y / features["height"]
                )
                task_id = str(x) + "-" + str(y)
                tile_height = min(features["height"] - y, self.tile_size)
                nb_buildings = self.valid_tile(
                    cx, cy, x_offset, y_offset, features["srid"]
                )
                filled_tile_name = tile_name.format(
                    filename=self.filename,
                    tile_width=tile_width,
                    tile_height=tile_height,
                    min_x=x,
                    min_y=y,
                )
                building_inventory += filled_tile_name + " " + str(nb_buildings) + "\n"
                if nb_buildings > 0:
                    task_in[task_id] = ExtractTileItems(
                        self.datapath,
                        self.dataset,
                        self.filename,
                        x,
                        y,
                        self.tile_size,
                        tile_width,
                        tile_height,
                    )
        inventory_filename = os.path.join(
            self.datapath, "preprocessed", "buildings_per_tile.txt"
        )
        fobj = open(inventory_filename, "a+")
        fobj.write(building_inventory)
        fobj.close()
        return task_in

    def valid_tile(self, x, y, x_offset, y_offset, srid):
        """
        """
        query = (
            "WITH bbox AS ("
            "SELECT ST_MakeEnvelope("
            "{west}, {south}, {east}, {north}, {srid}) AS geom"
            ") "
            "SELECT count("
            "st_intersection(st_makevalid(wkb_geometry), bbox.geom)) "
            "FROM {table} JOIN bbox "
            "ON st_intersects(st_makevalid(wkb_geometry), bbox.geom)"
            ";"
        ).format(
            table=self.filename,
            west=x,
            south=y - y_offset,
            east=x + x_offset,
            north=y,
            srid=srid,
        )
        config = utils.confparser["database"]
        connection_string = (
            "dbname={dbname} host={host} port={port} "
            "user={user} password={password}"
            ""
        ).format(
            dbname=config["dbname"],
            host=config["host"],
            port=config["port"],
            user=config["user"],
            password=config["password"],
        )
        connection = psycopg2.connect(connection_string)
        cursor = connection.cursor()
        cursor.execute(query)
        rset = cursor.fetchone()
        return rset[0]

    def complete(self):
        return False


class ExtractTileItemFromFolder(luigi.Task):
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
    tile_size = luigi.IntParameter(default=384)
    folder = luigi.Parameter()

    def requires(self):
        filenames = os.listdir(
            os.path.join(
                self.datapath,
                "preprocessed",
                str(self.tile_size),
                self.dataset,
                self.folder,
            )
        )
        for f in filenames:
            if self.dataset in ["training", "validation"]:
                f1, f2, tile_width, tile_height, x, y = f.split(".")[0].split("_")
                filename = "_".join([f1, f2])
            else:
                filename, tile_width, tile_height, x, y = f.split(".")[0].split("_")
            yield ExtractTileItems(
                self.datapath, self.dataset, filename, int(x), int(y), int(self.tile_size), int(tile_width), int(tile_height)
            )

    def complete(self):
        return False


class TrainMaskRCNN(luigi.Task):
    """Train a MaskRCNN model with tiled image

    Parameters
    ----------
    datapath : str
        Path of Tanzania dataset onto the file system
    tile_size : int
        Size of tiled images to consider during training
    """

    datapath = luigi.Parameter(default="./data/open_ai_tanzania")
    tile_size = luigi.IntParameter(default=384)

    def requires(self):
        return [
            ExtractValidTileItems(
                self.datapath, "training", f.split(".")[0], self.tile_size
            )
            for f in os.path.join(self.datapath, "input", "training", "images")
        ]

    def output(self):
        output_path = os.path.join(
            self.datapath, "output", "instance_segmentation", "checkpoints"
        )
        output_filename = (
            "_".join(
                (
                    self.filename,
                    str(self.tile_width),
                    str(self.tile_height),
                    str(self.min_x),
                    str(self.min_y),
                )
            )
            + ".tif"
        )
        return luigi.LocalTarget(os.path.join(output_path, output_filename))

    def run(self):
        train.TrainMaskRCNN(self.datapath)


class PredictBuildingsOnTile(luigi.Task):
    """
    """

    datapath = luigi.Parameter(default="./data/open_ai_tanzania")
    filename = luigi.Parameter()
    min_x = luigi.IntParameter()
    min_y = luigi.IntParameter()
    tile_size = luigi.IntParameter(default=384)
    tile_width = luigi.IntParameter(default=384)
    tile_height = luigi.IntParameter(default=384)

    # FOR NOW, TRAINING PRODUCES A FOLDER THAT DEPENDS ON DATE,
    # IT CAN'T BE USED AS A LUIGI REQUIRED TASK
    # def requires(self):
    #     return TrainMaskRCNN(self.datapath, self.tile_size)

    def output(self):
        output_path = os.path.join(
            self.datapath,
            "preprocessed",
            str(self.tile_size),
            "testing",
            "predicted_labels",
        )
        os.makedirs(output_path, exist_ok=True)
        output_filename = (
            "_".join(
                (
                    self.filename,
                    str(self.tile_width),
                    str(self.tile_height),
                    str(self.min_x),
                    str(self.min_y),
                )
            )
            + ".json"
        )
        return luigi.LocalTarget(os.path.join(output_path, output_filename))

    def run(self):
        filename = "_".join(
            (
                self.filename,
                str(self.tile_width),
                str(self.tile_height),
                str(self.min_x),
                str(self.min_y),
            )
        )
        result = inference.predict_on_filename(self.datapath, self.tile_size, filename)
        with open(self.output().path, "w") as fobj:
            json.dump(result, fobj)


class PredictBuildingsOnAllTiles(luigi.Task):
    """
    """

    datapath = luigi.Parameter(default="./data/open_ai_tanzania")
    tile_size = luigi.IntParameter(default=384)

    # FOR NOW, TRAINING PRODUCES A FOLDER THAT DEPENDS ON DATE,
    # IT CAN'T BE USED AS A LUIGI REQUIRED TASK
    # def requires(self):
    #     return TrainMaskRCNN(self.datapath, self.tile_size)

    def output(self):
        output_path = os.path.join(
            self.datapath, "preprocessed", str(self.tile_size), "testing"
        )
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(os.path.join(output_path, "predicted_labels"), exist_ok=True)
        output_filename = "prediction_log.json"
        return luigi.LocalTarget(os.path.join(output_path, output_filename))

    def run(self):
        log = inference.predict_on_folder(self.datapath, self.tile_size)
        with open(self.output().path, "w") as fobj:
            json.dump(log, fobj)


class PostProcessTile(luigi.Task):
    """
    """

    datapath = luigi.Parameter(default="./data/open_ai_tanzania")
    filename = luigi.Parameter()
    min_x = luigi.IntParameter()
    min_y = luigi.IntParameter()
    tile_size = luigi.IntParameter(default=384)
    tile_width = luigi.IntParameter(default=384)
    tile_height = luigi.IntParameter(default=384)

    def requires(self):
        return {
            "prediction": PredictBuildingsOnTile(
                self.datapath,
                self.filename,
                self.min_x,
                self.min_y,
                self.tile_size,
                self.tile_width,
                self.tile_height,
            ),
            "features": GetTileFeatures(
                self.datapath,
                "testing",
                self.filename,
                self.min_x,
                self.min_y,
                self.tile_size,
                self.tile_width,
                self.tile_height,
            ),
        }

    def output(self):
        output_path = os.path.join(
            self.datapath,
            "preprocessed",
            str(self.tile_size),
            "testing",
            "tiled_predictions",
        )
        os.makedirs(output_path, exist_ok=True)
        output_filename = (
            "_".join(
                (
                    self.filename,
                    str(self.tile_width),
                    str(self.tile_height),
                    str(self.min_x),
                    str(self.min_y),
                )
            )
            + ".csv"
        )
        return luigi.LocalTarget(os.path.join(output_path, output_filename))

    def run(self):
        feature_path = self.input()["features"].path
        with open(feature_path) as fobj:
            features = json.load(fobj)
        pred_path = feature_path.replace("features", "predicted_labels")
        with open(pred_path) as fobj:
            predictions = json.load(fobj)
        results = postprocessing.postprocess_tile(
            features, predictions, self.min_x, self.min_y
        )
        df = pd.DataFrame(results)
        if not df.empty:
            df.columns = [
                "conf_completed",
                "conf_unfinished",
                "conf_foundation",
                "coords_geo",
                "coords_pixel",
            ]
        df.index.name = "building_id"
        with open(self.output().path, "w") as fobj:
            df.to_csv(self.output().path)


class PostProcessAllTiles(luigi.Task):
    """
    """

    datapath = luigi.Parameter(default="./data/open_ai_tanzania")
    tile_size = luigi.IntParameter(default=384)

    # FOR NOW, TRAINING PRODUCES A FOLDER THAT DEPENDS ON DATE,
    # IT CAN'T BE USED AS A LUIGI REQUIRED TASK
    # def requires(self):
    #     return TrainMaskRCNN(self.datapath, self.tile_size)

    def output(self):
        output_path = os.path.join(
            self.datapath, "preprocessed", str(self.tile_size), "testing"
        )
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(os.path.join(output_path, "predicted_labels"), exist_ok=True)
        output_filename = "PostProcessAllTiles_log.json"
        return luigi.LocalTarget(os.path.join(output_path, output_filename))

    def run(self):
        predict_folder = os.path.join(
            self.datapath,
            "preprocessed",
            str(self.tile_size),
            "testing",
            "predicted_labels",
        )
        predicted_files = os.listdir(predict_folder)
        with Pool() as p:
            results = p.starmap(
                postprocessing.postprocess_folder, [(t,) for t in predicted_files]
            )

        df = pd.DataFrame([l for l in results if len(l) > 0])
        if not df.empty:
            df.columns = [
                "conf_completed",
                "conf_unfinished",
                "conf_foundation",
                "coords_geo",
                "coords_pixel",
            ]
        df.index.name = "building_id"
        with open(self.output().path, "w") as fobj:
            df.to_csv(self.output().path)
        with open(self.output().path, "w") as fobj:
            json.dump(log, fobj)


class PostProcessFromFolder(luigi.Task):
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
    tile_size = luigi.IntParameter(default=384)
    folder = luigi.Parameter()

    def requires(self):
        task_in = {}
        filenames = os.listdir(
            os.path.join(
                self.datapath,
                "preprocessed",
                str(self.tile_size),
                "testing",
                self.folder,
            )
        )
        for f in filenames:
            filename, tile_width, tile_height, x, y = f.split(".")[0].split("_")
            task_in[f] = PostProcessTile(
                self.datapath, filename, x, y, self.tile_size, tile_width, tile_height
            )
        return task_in

    def complete(self):
        return False


class PostProcessTiles(luigi.Task):
    """
    """

    datapath = luigi.Parameter(default="./data/open_ai_tanzania")
    filename = luigi.Parameter()
    tile_size = luigi.IntParameter(default=384)

    @property
    def tiles(self):
        image_dir = os.path.join(
            self.datapath, "preprocessed", str(self.tile_size), "testing", "images"
        )
        return [
            fname.split(".")[0]
            for fname in os.listdir(image_dir)
            if self.filename in fname
        ]

    def requires(self):
        task_dict = {}
        task_dict["prediction"] = PredictBuildingsOnAllTiles(
            self.datapath, self.tile_size
        )
        for t in self.tiles:
            filename, width, height, min_x, min_y = t.split("_")
            task_dict.update(
                {
                    "-".join(("features", t)): GetTileFeatures(
                        self.datapath,
                        "testing",
                        filename,
                        min_x,
                        min_y,
                        self.tile_size,
                        width,
                        height,
                    )
                }
            )
        return task_dict

    def output(self):
        output_path = os.path.join(
            self.datapath,
            "preprocessed",
            str(self.tile_size),
            "testing",
            "tiled_buildings",
        )
        os.makedirs(output_path, exist_ok=True)
        output_filename = self.filename + ".csv"
        return luigi.LocalTarget(os.path.join(output_path, output_filename))

    def run(self):
        results = []
        with Pool() as p:
            results = p.starmap(
                postprocessing.postprocess, [(t, self.input()) for t in self.tiles]
            )

        df = pd.DataFrame([l for l in results if len(l) > 0])
        if not df.empty:
            df.columns = [
                "conf_completed",
                "conf_unfinished",
                "conf_foundation",
                "coords_geo",
                "coords_pixel",
            ]
        df.index.name = "building_id"
        with open(self.output().path, "w") as fobj:
            df.to_csv(self.output().path)


class PostProcessAllImages(luigi.Task):
    """
    """

    datapath = luigi.Parameter(default="./data/open_ai_tanzania")
    tile_size = luigi.IntParameter(default=384)

    @property
    def filenames(self):
        image_dir = os.path.join(self.datapath, "input", "testing", "images")
        return [fname.split(".")[0] for fname in os.listdir(image_dir)]

    def requires(self):
        for filename in self.filenames:
            yield PostProcessTiles(self.datapath, filename, self.tile_size)

    def complete(self):
        return False


class MergeResults(luigi.Task):
    """
    """

    datapath = luigi.Parameter(default="./data/open_ai_tanzania")
    tile_size = luigi.IntParameter(default=384)
    filename = luigi.Parameter()
    folder = luigi.Parameter()

    def requires(self):
        task_in = {}
        filenames = os.listdir(
            os.path.join(
                self.datapath,
                "preprocessed",
                str(self.tile_size),
                "testing",
                self.folder,
            )
        )
        filenames = [f for f in filenames if self.filename in f]
        for f in filenames:
            filename, tile_width, tile_height, x, y = f.split(".")[0].split("_")
            task_in[f] = PostProcessTile(
                self.datapath, filename, x, y, self.tile_size, tile_width, tile_height
            )
        return task_in

    def output(self):
        output_path = os.path.join(
            self.datapath,
            "preprocessed",
            str(self.tile_size),
            "testing",
            "merged_predictions",
        )
        os.makedirs(output_path, exist_ok=True)
        output_filename = self.filename + ".csv"
        return luigi.LocalTarget(os.path.join(output_path, output_filename))

    def run(self):
        combined_csv = pd.concat(
            [
                pd.read_csv(input_file.path, index_col=None)
                for key, input_file in self.input().items()
            ]
        )
        combined_csv["building_id"] = range(combined_csv.shape[0])
        combined_csv.set_index(["building_id"], inplace=True)
        combined_csv.to_csv(self.output().path)


class MergeAllResults(luigi.Task):
    """
    """

    datapath = luigi.Parameter(default="./data/open_ai_tanzania")
    tile_size = luigi.IntParameter(default=384)

    @property
    def filenames(self):
        image_dir = os.path.join(self.datapath, "input", "testing", "images")
        return [fname.split(".")[0] for fname in os.listdir(image_dir)]

    def requires(self):
        for filename in self.filenames:
            yield MergeResults(
                self.datapath, self.tile_size, filename, "predicted_labels"
            )

    def complete(self):
        return False


class GeolocalizeMergedResults(luigi.Task):
    """
    """

    datapath = luigi.Parameter(default="./data/open_ai_tanzania")
    tile_size = luigi.IntParameter(default=384)
    filename = luigi.Parameter()

    def requires(self):
        return MergeResults(
            self.datapath, self.tile_size, self.filename, "predicted_labels"
        )

    def output(self):
        output_path = os.path.join(
            self.datapath,
            "preprocessed",
            str(self.tile_size),
            "testing",
            "geo_predictions",
        )
        os.makedirs(output_path, exist_ok=True)
        output_filename = self.filename + ".geojson"
        return luigi.LocalTarget(os.path.join(output_path, output_filename))

    def run(self):
        predictions = pd.read_csv(self.input().path)
        predictions["geom"] = [swkt.loads(s) for s in predictions["coords_geo"]]
        gdf = gpd.GeoDataFrame(predictions, geometry="geom")
        gdf.to_file(self.output().path, driver="GeoJSON")

class GeolocalizePredictedTileItems(luigi.Task):
    """
    """

    datapath = luigi.Parameter(default="./data/open_ai_tanzania")
    filename = luigi.Parameter()
    min_x = luigi.IntParameter()
    min_y = luigi.IntParameter()
    tile_size = luigi.IntParameter(default=384)
    tile_width = luigi.IntParameter(default=384)
    tile_height = luigi.IntParameter(default=384)

    def requires(self):
        return PostProcessTile(
            self.datapath, self.filename, self.min_x, self.min_y,
            self.tile_size, self.tile_width, self.tile_height
        )

    def output(self):
        output_path = os.path.join(
            self.datapath,
            "preprocessed",
            str(self.tile_size),
            "testing",
            "geo_predictions",
        )
        os.makedirs(output_path, exist_ok=True)
        output_filename = (
            "_".join(
                (
                    self.filename,
                    str(self.tile_width),
                    str(self.tile_height),
                    str(self.min_x),
                    str(self.min_y),
                )
            )
            + ".geojson"
        )
        return luigi.LocalTarget(os.path.join(output_path, output_filename))

    def run(self):
        predictions = pd.read_csv(self.input().path)
        predictions["geom"] = [swkt.loads(s) for s in predictions["coords_geo"]]
        gdf = gpd.GeoDataFrame(predictions, geometry="geom")
        gdf.to_file(self.output().path, driver="GeoJSON")


class GeolocalizeAllResults(luigi.Task):
    """
    """

    datapath = luigi.Parameter(default="./data/open_ai_tanzania")
    tile_size = luigi.IntParameter(default=384)

    @property
    def filenames(self):
        image_dir = os.path.join(self.datapath, "input", "testing", "images")
        return [fname.split(".")[0] for fname in os.listdir(image_dir)]

    def requires(self):
        for filename in self.filenames:
            yield GeolocalizeMergedResults(self.datapath, self.tile_size, filename)

    def complete(self):
        return False
