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
import pickle
import h5py
from sklearn import neighbors
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from databaseConstructor import pixelToCoordinates

# loading predictions
y_pred = np.array(pickle.load(file('predicted_normalised.p', 'rb')))
print min(y_pred), max(y_pred)
# load X values
df_train = cPickle.load(file('knn_X_data.save', 'rb'))
df_train = df_train[0:len(y_pred)]
location = df_train['location']
lat = [l.y for l in location]
lon = [l.x for l in location]
df_train['lat'] = lat
df_train['lon'] = lon
keep_vars = ['lat', 'lon', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6_VCID_2', 'B7']
X_train = np.array(df_train[keep_vars])
print y_pred.shape, X_train.shape
# getting size of image file
X_interpolate = cPickle.load(file('to_interpolate_data.save', 'rb'))
location = X_interpolate['location']
lat = [l.y for l in location]
lon = [l.x for l in location]
X_interpolate['lat'] = lat
X_interpolate['lon'] = lon

X_interpolate = np.array(X_interpolate[keep_vars])
# chopping up the grid into overlapping ranges
kf = KFold(len(X_train), n_folds=10, shuffle=True, random_state=None)
X_train = np.array(X_train)
y_pred = y_pred




n_neighbours = 5
best_neighbours = 5 
score = 10
for train_index, test_index in kf:
	print len(y_pred), y_pred[0]
	X_train_tmp = X_train[train_index, ]
	y_train_tmp = y_pred[train_index]
	X_test_tmp = X_train[test_index, ]
	y_test_tmp = y_pred[test_index]
	knn = neighbors.KNeighborsRegressor(n_neighbours, weights='uniform')
	model = knn.fit(X_train_tmp, y_train_tmp)
	y_pred_tmp = model.predict(X_test_tmp)
	tmp_score = mean_squared_error(y_test_tmp, y_pred_tmp)
	print 'Number neighbours', n_neighbours
	print 'RMSE', tmp_score
	if tmp_score < score:
		score = tmp_score
		best_neighbours = n_neighbours
		y_interpolated = model.predict(X_interpolate)
	n_neighbours += 5

print 'Optimal number of neighbours', best_neighbours

cPickle.dump(y_interpolated, file('mapping/y_interpolated.save', 
			'wb'), protocol= cPickle.HIGHEST_PROTOCOL)
cPickle.dump(X_interpolate, file('mapping/location.save', 
			'wb'), protocol= cPickle.HIGHEST_PROTOCOL)

print y_interpolated

"""
For batching
"""
"""
# columns
batch_size = 200
overlap = 32
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
p
for i in np.arange(len(row_slice_upper)):
	for j in np.arange(len(col_slice_upper)):
		print count, i, j
		tmp_batch_cols, tmp_batch_rows = np.meshgrid(
									np.arange(row_slice_lower[i], row_slice_upper[i]), 
									np.arange(col_slice_lower[j], col_slice_upper[j]))
		tmp_col_grid = tmp_batch_cols.ravel()
		tmp_row_grid = tmp_batch_rows.ravel()		
		tmp_location_series = [Point(pixelToCoordinates(
								satellite_gdal.GetGeoTransform(), col, row)) \
								for (col, row) in zip(tmp_col_grid, tmp_row_grid)]
		tmp_coordinates = [ (point.x, point.y) for point in tmp_location_series]
		y_interpolated = model.predict(tmp_coordinates)
		print max(y_interpolated), min(y_interpolated)
		count += 1
	cPickle.dump(y_interpolated, file('mapping/y_interpolated_%d.save' % count, 
			'wb'), protocol= cPickle.HIGHEST_PROTOCOL)
	cPickle.dump(tmp_location_series, file('mapping/location_%d.save' % count, 
			'wb'), protocol= cPickle.HIGHEST_PROTOCOL)

"""
