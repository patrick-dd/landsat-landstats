"""
 	Obtains estimate for each pixel

Problem:
Want population estimates for each pixel in a state
Only have estimates for a sample of pixels in the test set. 

Answer: 
Interpolate between points
This can be answered through 
> K-NN clustering
> Density estimation
> supervised learning methods

Begin with K-NN for simplicity
""" 

import numpy as np 
import numpy as np
import pandas as pd
from osgeo import gdal, ogr
from shapely.geometry import Point, Polygon, shape
import shapely.ops as ops
from rtree import index 
from geopandas import GeoDataFrame
from functools import partial
import pyproj
import cPickle
import h5py
from sklearn import neighbors
import cPickle
from databaseConstructor import pixelToCoordinates

# return (lon_coord, lat_coord)



# loading predictions
y_pred = np.array(cPickle.load(file('output_predictions.save', 'rb')))

# load X values
df_train = 
lat = df_train['latitude']
lon = df_train['longitude']
X_train = [(latitude, longitude) for (latitude, longitude) in zip(lat, lon)]

# getting size of image file
filename = 'LANDSAT_TOA/Washington/washington_b1.tif'
satellite_gdal = gdal.Open(filename)
ncols, nrows = satellite_gdal.RasterXSize, satellite_gdal.RasterYSize
ncols, nrows = 10, 10
cols_grid, rows_grid = np.meshgrid(range(0,ncols), range(0,nrows))
rows_grid, cols_grid = rows_grid.ravel(), cols_grid.ravel()
location_series = [Point(pixelToCoordinates(satellite_gdal.GetGeoTransform(), col, row)) \
				for (col, row) in zip(cols_grid, rows_grid)]

coordinates = [ (point.y, point.x) for point in location_series]

output = 


n_neighbors = 10
knn = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')
y_ = knn.fit(X_train, y_pred).predict(coordinates)



