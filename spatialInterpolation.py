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
df_train = cPickle.load(file('knn_X_data.save', 'rb'))
lat = df_train['latitude']
lon = df_train['longitude']
X_train = [(latitude, longitude) for (latitude, longitude) in zip(lat, lon)]
X_train = X_train[0:len(y_pred)]

# getting size of image file
filename = 'LANDSAT_TOA/Washington/washington_b1.tif'
satellite_gdal = gdal.Open(filename)
ncols, nrows = satellite_gdal.RasterXSize, satellite_gdal.RasterYSize

# chopping up the grid into overlapping ranges
# columns
batch_size = 1500
overlap = 350
batch_size_less_olap = batch_size - overlap
col_regions = ncols / batch_size_less_olap + 1
col_slice_lower = [batch_size_less_olap * x for x in np.arange(col_regions)]
col_slice_upper = [batch_size_less_olap * x + batch_size for x in np.arange(col_regions)]
col_slice_upper[-1] = ncols
col_length = len(col_slice_upper)
# rows
row_regions = nrows / batch_size_less_olap + 1
row_slice_lower = [batch_size_less_olap * x for x in np.arange(row_regions)]
row_slice_upper = [batch_size_less_olap * x + batch_size for x in np.arange(row_regions)]
row_slice_upper[-1] = nrows
row_length = len(row_slice_upper)
print 'Makin batches'
n_neighbors = 5
knn = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')
model = knn.fit(X_train, y_pred)

count = 0
for i in np.arange(len(row_slice_upper)):
	for j in np.arange(len(col_slice_upper)):
		print count
		tmp_batch_rows, tmp_batch_cols = np.meshgrid(
									np.arange(row_slice_lower[i], row_slice_upper[i]), 
									np.arange(col_slice_lower[j], col_slice_upper[j]))
		tmp_col_grid = tmp_batch_cols.ravel()
		tmp_row_grid = tmp_batch_rows.ravel()		
		tmp_location_series = [Point(pixelToCoordinates(
								satellite_gdal.GetGeoTransform(), col, row)) \
								for (col, row) in zip(tmp_col_grid, tmp_row_grid)]
		tmp_coordinates = [ (point.y, point.x) for point in tmp_location_series]
		y_interpolated = model.predict(tmp_coordinates)
		print y_interpolated
		cPickle.dump(y_interpolated, file('y_interpolated_%d.save' % count, 
					'wb'), protocol= cPickle.HIGHEST_PROTOCOL)
		count += 1


