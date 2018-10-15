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
from luigi.contrib.postgres import CopyToTable
import os
from osgeo import gdal
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
    filename = luigi.Parameter()
    min_x = luigi.IntParameter()
    min_y = luigi.IntParameter()
    tile_width = luigi.IntParameter(default=5000)
    tile_height = luigi.IntParameter(default=5000)

    def output(self):
        output_path = os.path.join(self.datapath, "preprocessed",
                                   str(self.tile_width)+"_"+str(self.tile_height),
                                   "training", "images")
        os.makedirs(output_path, exist_ok=True)
        output_filename_suffix = "_{}_{}_{}_{}.tif".format(self.tile_width,
                                                           self.tile_height,
                                                           self.min_x,
                                                           self.min_y)
        output_filename = self.filename + output_filename_suffix
        return luigi.LocalTarget(os.path.join(output_path, output_filename))

    def run(self):
        input_path = os.path.join(self.datapath, "input", "training", "images",
                                  self.filename + ".tif")
        gdal_translate_args = ['-srcwin',
                               self.min_x, self.min_y,
                               self.tile_width, self.tile_height,
                               input_path,
                               self.output().path]
        sh.gdal_translate(gdal_translate_args)


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

    def output(self):
        output_path = os.path.join(self.datapath, "input",
                                   "training", "features")
        os.makedirs(output_path, exist_ok=True)
        output_filename = os.path.join(output_path, self.filename + ".json")
        return luigi.LocalTarget(output_filename)

    def run(self):
        input_filename = os.path.join(self.datapath, "input", "training",
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
    filename = luigi.Parameter()
    min_x = luigi.IntParameter()
    min_y = luigi.IntParameter()
    tile_width = luigi.IntParameter(default=5000)
    tile_height = luigi.IntParameter(default=5000)

    def requires(self):
        return GenerateSubTile(self.datapath, self.filename,
                               self.min_x, self.min_y,
                               self.tile_width, self.tile_height)

    def output(self):
        output_path = os.path.join(self.datapath, "preprocessed",
                                   str(self.tile_width)+"_"+str(self.tile_height),
                                   "training", "features")
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


class GenerateTileRaster(luigi.Task):
    """Generate a raster for a given label tile, after querying it into the
    dedicated PostGreSQL database through Mapnik.

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
    background_color : list of 3 ints
        RGB tuple for the raster background
    complete_color : list of 3 ints
        RGB tuple corresponding to `complete` building representation
    incomplete_color : list of 3 ints
        RGB tuple corresponding to `incomplete` building representation
    foundation_color : list of 3 ints
        RGB tuple corresponding to `foundation` building representation

    """
    datapath = luigi.Parameter(default="./data/open_ai_tanzania")
    filename = luigi.Parameter()
    min_x = luigi.IntParameter()
    min_y = luigi.IntParameter()
    tile_width = luigi.IntParameter(default=5000)
    tile_height = luigi.IntParameter(default=5000)
    background_color = luigi.ListParameter(default=[0, 0, 0])
    complete_color = luigi.ListParameter(default=[50, 200, 50]) # Green
    incomplete_color = luigi.ListParameter(default=[200, 200, 50]) # Yellow
    foundation_color = luigi.ListParameter(default=[200, 50, 50]) # Red

    @property
    def short_filename(self):
        return "_".join(self.filename.split("_")[:2])

    def requires(self):
        return {"features": GetTileFeatures(self.datapath, self.filename,
                                            self.min_x, self.min_y,
                                            self.tile_width, self.tile_height),
                "labels": StoreLabelsToDatabase(self.datapath,
                                                self.short_filename)}

    def output(self):
        output_path = os.path.join(self.datapath, "preprocessed",
                                   str(self.tile_width),
                                   "training", "labels")
        os.makedirs(output_path, exist_ok=True)
        output_filename_suffix = "_{}_{}_{}_{}.png".format(self.tile_width,
                                                           self.tile_height,
                                                           self.min_x,
                                                           self.min_y)
        output_filename = self.filename + output_filename_suffix
        return luigi.LocalTarget(os.path.join(output_path, output_filename))

    def run(self):
        with self.input()["features"].open('r') as fobj:
            features = json.load(fobj)
        classes = {"background": self.background_color,
                   "Complete": self.complete_color,
                   "Incomplete": self.incomplete_color,
                   "Foundation": self.foundation_color,}
        utils.generate_raster(self.output().path, self.short_filename,
                              features, classes)


class GenerateAllTileRasters(luigi.Task):
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
    background_color : list of 3 ints
        RGB tuple for the raster background
    complete_color : list of 3 ints
        RGB tuple corresponding to `complete` building representation
    incomplete_color : list of 3 ints
        RGB tuple corresponding to `incomplete` building representation
    foundation_color : list of 3 ints
        RGB tuple corresponding to `foundation` building representation

    """
    datapath = luigi.Parameter(default="./data/open_ai_tanzania")
    filename = luigi.Parameter()
    tile_size = luigi.IntParameter(default=5000)
    background_color = luigi.ListParameter(default=[0, 0, 0])
    complete_color = luigi.ListParameter(default=[50, 200, 50]) # Green
    incomplete_color = luigi.ListParameter(default=[200, 200, 50]) # Yellow
    foundation_color = luigi.ListParameter(default=[200, 50, 50]) # Red

    def requires(self):
        task_in = {}
        ds = gdal.Open(os.path.join(self.datapath, "input", "training",
                                    "images", self.filename + ".tif"))
        xsize = ds.RasterXSize
        ysize = ds.RasterYSize
        for x in range(0, xsize, self.tile_size):
            tile_width = min(xsize - x, self.tile_size)
            for y in range(0, ysize, self.tile_size):
                task_id = str(x) + "-" + str(y)
                tile_height = min(ysize - y, self.tile_size)
                if tile_width == tile_height == self.tile_size:
                    task_in[task_id] = GenerateTileRaster(self.datapath,
                                                          self.filename,
                                                          x, y,
                                                          tile_width,
                                                          tile_height,
                                                          self.background_color,
                                                          self.complete_color,
                                                          self.incomplete_color,
                                                          self.foundation_color)
        ds = None
        return task_in

    def complete(self):
        return False


class ReprojectTileRaster(luigi.Task):
    """Reproject label tiles after Mapnik rendering process, as it produces
    simple image without geographical metadata. The tile metadata are deduced
    from corresponding image tile metadata.

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
    background_color : list of 3 ints
        RGB tuple for the raster background
    complete_color : list of 3 ints
        RGB tuple corresponding to `complete` building representation
    incomplete_color : list of 3 ints
        RGB tuple corresponding to `incomplete` building representation
    foundation_color : list of 3 ints
        RGB tuple corresponding to `foundation` building representation

    """
    datapath = luigi.Parameter(default="./data/open_ai_tanzania")
    filename = luigi.Parameter()
    min_x = luigi.IntParameter()
    min_y = luigi.IntParameter()
    tile_width = luigi.IntParameter(default=5000)
    tile_height = luigi.IntParameter(default=5000)
    background_color = luigi.ListParameter(default=[0, 0, 0])
    complete_color = luigi.ListParameter(default=[50, 200, 50]) # Green
    incomplete_color = luigi.ListParameter(default=[200, 200, 50]) # Yellow
    foundation_color = luigi.ListParameter(default=[200, 50, 50]) # Red

    def requires(self):
        return {"image": GenerateSubTile(self.datapath, self.filename,
                                         self.min_x, self.min_y,
                                         self.tile_width, self.tile_height),
                "label": GenerateTileRaster(self.datapath, self.filename,
                                            self.min_x, self.min_y,
                                            self.tile_width, self.tile_height,
                                            self.background_color,
                                            self.complete_color,
                                            self.incomplete_color,
                                            self.foundation_color)}


    def output(self):
        output_path = os.path.join(self.datapath, "preprocessed",
                                   str(self.tile_width),
                                   "training", "proj_labels")
        os.makedirs(output_path, exist_ok=True)
        output_filename_suffix = "_{}_{}_{}_{}.tif".format(self.tile_width,
                                                           self.tile_height,
                                                           self.min_x,
                                                           self.min_y)
        output_filename = self.filename + output_filename_suffix
        return luigi.LocalTarget(os.path.join(output_path, output_filename))

    def run(self):
        image_source = gdal.Open(self.input()["image"].path)
        label_source = gdal.Open(self.input()["label"].path)
        geodriver = image_source.GetDriver()
        out_source = geodriver.Create(self.output().path,
                                      label_source.RasterXSize,
                                      label_source.RasterYSize,
                                      label_source.RasterCount,
                                      gdal.GDT_Int16)
        out_source.SetProjection(image_source.GetProjection())
        out_source.SetGeoTransform(image_source.GetGeoTransform())
        for band in range(1, label_source.RasterCount + 1):
            band_array = label_source.GetRasterBand(band).ReadAsArray()
            out_source.GetRasterBand(band).WriteArray(band_array)
        out_source = image_source = label_source = band_array = None


class MergeLabelRaster(luigi.Task):
    """Merge all tiles that correspond to a single raw image so as to produce a
    big labelled version of that image. This task completes the process for an
    image by reproducing big images.

    This task is done through running gdal_merge tool, with a compressing
    option to keep raster size reasonable.

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
    background_color : list of 3 ints
        RGB tuple for the raster background
    complete_color : list of 3 ints
        RGB tuple corresponding to `complete` building representation
    incomplete_color : list of 3 ints
        RGB tuple corresponding to `incomplete` building representation
    foundation_color : list of 3 ints
        RGB tuple corresponding to `foundation` building representation

    """
    datapath = luigi.Parameter(default="./data/open_ai_tanzania")
    filename = luigi.Parameter()
    tile_size = luigi.IntParameter(default=5000)
    background_color = luigi.ListParameter(default=[0, 0, 0])
    complete_color = luigi.ListParameter(default=[50, 200, 50]) # Green
    incomplete_color = luigi.ListParameter(default=[200, 200, 50]) # Yellow
    foundation_color = luigi.ListParameter(default=[200, 50, 50]) # Red

    def requires(self):
        task_in = {}
        ds = gdal.Open(os.path.join(self.datapath, "input", "training",
                                    "images", self.filename + ".tif"))
        xsize = ds.RasterXSize
        ysize = ds.RasterYSize
        for x in range(0, xsize, self.tile_size):
            tile_width = min(xsize - x, self.tile_size)
            for y in range(0, ysize, self.tile_size):
                task_id = str(x) + "-" + str(y)
                tile_height = min(ysize - y, self.tile_size)
                task_in[task_id] = ReprojectTileRaster(self.datapath,
                                                       self.filename,
                                                       x, y,
                                                       tile_width, tile_height)
        ds = None
        return task_in

    def output(self):
        output_path = os.path.join(self.datapath, "input",
                                   "training", "merged_labels")
        os.makedirs(output_path, exist_ok=True)
        output_filename = self.filename + ".tif"
        return luigi.LocalTarget(os.path.join(output_path, output_filename))

    def run(self):
        input_paths = [value.path for key, value in self.input().items()]
        gdal_merge_args = ['-o', self.output().path,
                           *input_paths,
                           '-co', 'COMPRESS=DEFLATE',
                           '-q', '-v']
        subprocess.call(["gdal_merge.py", *gdal_merge_args])


class MergeAllLabelRasters(luigi.Task):
    """Final task that calls `MergeLabelRaster` for each raw image contained
    into the dataset.

    Attributes
    ----------
    datapath : str
        Path towards the Tanzania challenge data
    tile_size : int
        Number of pixels that must be considered in both direction (east-west,
    north-south) in tile definition. This constraint is relaxed when
    considering border tiles (on east and south borders, especially).
    background_color : list of 3 ints
        RGB tuple for the raster background
    complete_color : list of 3 ints
        RGB tuple corresponding to `complete` building representation
    incomplete_color : list of 3 ints
        RGB tuple corresponding to `incomplete` building representation
    foundation_color : list of 3 ints
        RGB tuple corresponding to `foundation` building representation

    """
    datapath = luigi.Parameter(default="./data/open_ai_tanzania")
    tile_size = luigi.IntParameter(default=5000)
    background_color = luigi.ListParameter(default=[0, 0, 0])
    complete_color = luigi.ListParameter(default=[50, 200, 50]) # Green
    incomplete_color = luigi.ListParameter(default=[200, 200, 50]) # Yellow
    foundation_color = luigi.ListParameter(default=[200, 50, 50]) # Red

    def requires(self):
        datadir = os.path.join(self.datapath, "input", "training", "images")
        filenames = [filename.split('.')[0]
                     for filename in os.listdir(datadir)]
        return {f: MergeLabelRaster(self.datapath, f, self.tile_size,
                                    self.background_color,
                                    self.complete_color,
                                    self.incomplete_color,
                                    self.foundation_color)
                for f in filenames}

    def complete(self):
        return False
