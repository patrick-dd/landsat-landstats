## Create the full database!

"""
A script to create a database of images from the LANDSAT 7 database

This is to generate images for output. So the full state!

The basic idea is to geneate a random grid of images that overlays the state. 
	- Randomisation is to allow some overlap.
	- Still computationally tractable
	A full overlapping grid (as in sklearn.feature_extraction.image.extract_patches_2d)
	chews up too much data.
Then get estimates for these images
Then organise into a DataFrame and group pixel pop densities by 
taking the mean.

We then merge with the County level database to get County boundaries
for each pixel
Group the average pop density by County. Times by area to get population

Inputs:
	state_name: The state name as a string
	state_code: Two letter state code as a string
	year: year as a string
	num_observations: Number of observations to feed into ConvNet (int)
	observation_size: the length/width of each observation (int)
	
"""
import numpy as np
import pandas as pd
from osgeo import gdal, ogr
from shapely.geometry import Point, Polygon, shape
import shapely.ops as ops
from rtree import index 
from geopandas import GeoDataFrame
from functools import partial
import pyproj
import pickle, cPickle
import h5py
from sklearn.feature_extraction.image import extract_patches_2d
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import callbacks
from keras.layers import advanced_activations
from keras.optimizers import Adam, SGD


def import_sat_data(file_name):
	"""
	Returns a DataFrame with pixel intensity and location
	"""
	# getting size of image file
	keep_vars = ['lat', 'lon', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6_VCID_2', 'B7']
	df_interpolate = cPickle.load(file(file_name, 'rb'))
	location = df_interpolate['location']
	lat = [l.y for l in location]
	lon = [l.x for l in location]
	df_interpolate['lat'] = lat
	df_interpolate['lon'] = lon
	df_interpolate = np.array(df_interpolate[keep_vars])
	return df_interpolate


def get_sample(dataframe, obs_size, overlap, nrows, ncols):
	"""
	Generates sample of images for estimation
	Inputs:
		dataframe: a pandas DataFrame of satellite images
		obs_size: the size of the images for estimation (int)
		overlap: the average proportion overlap per images (between 0 and 1)
		nrows, ncols: dimension of original satellite image
	Returns:
		Sample for estimation (array)
	"""
	mini_db = [dataframe[:,i].reshape((nrows, ncols)) for i in range(dataframe.shape[1])]
	mini_db = np.array(mini_db)
	row_length = int((nrows - obs_size + 1) / (overlap * obs_size))
	col_length = int((ncols - obs_size + 1) / (overlap * obs_size))
	row_indices = np.random.random_integers(0, nrows - obs_size + 1, row_length)
	col_indices = np.random.random_integers(0, ncols - obs_size + 1, col_length)
	sample = []
	for row in row_indices:
		for col in col_indices:
			tmp = mini_db[:, row : row + obs_size, col : col + obs_size ]
			sample.append(tmp)
	print mini_db.shape
	return np.array(sample)

def normalise_data(X):
	"""
	Normalises X values for estimation
	"""
	# mean normalisation
	mean_value = np.mean(X)
	std_value = np.std(X)
	X -= mean_value
	X = X / std_value
	print std_value
	return X

def create_model(weights_path=None):
	model = Sequential()
	model.add(Convolution2D(256, 5, 5, 
			border_mode='valid',
			input_shape = (7, obs_size, obs_size)))
	model.add(Activation('relu'))
	model.add(Dropout(0.25))

	# second convolutional pair
	model.add(Convolution2D(64, 3, 3, border_mode='valid'))
	model.add(MaxPooling2D(pool_size=(3, 3)))
	model.add(Activation('relu'))
	model.add(Dropout(0.25))	
	# third convolutional pair
	model.add(Convolution2D(128, 5, 5, border_mode='valid'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Activation('relu'))
	model.add(Dropout(0.25))
	# forth convolutional layer 
	model.add(Convolution2D(64, 3, 3))
	model.add(MaxPooling2D(pool_size=(3, 3)))
	model.add(Activation('relu'))
	model.add(Dropout(0.25))
	# convert convolutional filters to flat so they
	# can be fed to fully connected layers
	model.add(Flatten())
	# first fully connected layer
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.25))
	# second fully connected layer
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(0.3))
	# third fully connected layer
	model.add(Dense(25))
	model.add(Activation('linear'))
	# classification fully connected layer
	model.add(Dense(1))
	model.add(Activation('linear'))
	if weights_path:
		model.load_weights(weights_path)
	return model

def un_normalise_y(y_pred, max_train):
	"""
	Takes the normalised y and unnormalises import
	"""
	## times by max_train
	y_pred *= max_train
	## raise to power of ten
	y_pred = (y_pred+1)**10 - 1
	return y_pred

def get_estimates(X):
	"""
	Gets population estimates for the sample
	Takes:
		X: numpy array
	Returns:
		y_pred: normalised population density predictions
	"""
	model_loaded = create_model('model_weights.h5')
	adam = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8)
	model_loaded.compile(loss='mean_squared_error', optimizer='adam')
	y_pred = model_loaded.predict(X)
	return y_pred

def to_dataframe(y, sample, obs_size):
	"""
	Adds predictions to DataFrame
	Inputs:
		sample: to get location data (array)
		y: predictions (array)
		obs_size: size of images (int)
	Returns:
		df: DataFrame 
	"""
	sample = sample[:,0:2, :, :]
	db = [['estimate', 'lat', 'lon']]
	for i in range(len(y)):
		prediction = y[i][0]
		lat = sample[i, 0, obs_size/2, obs_size/2]
		lon = sample[i, 1, obs_size/2, obs_size/2]
		db.append([prediction, lat, lon])
	return pd.DataFrame(db)

obs_size = 64
overlap = 0.95
nrows = 386
ncols = 885
max_train = 2.65
satellite_data_name = 'to_interpolate_data.save'
df = import_sat_data(satellite_data_name)
sample = get_sample(df, obs_size, overlap, nrows, ncols)
X_test = sample[:, 2:, :, :]
X_test = normalise_data(X_test)
y_pred = get_estimates(X_test)
print len(y_pred)
print np.std(y_pred)
y_pred = un_normalise_y(y_pred, max_train)
df = to_dataframe(y_pred, sample, obs_size)
pickle.dump( df, open( "prediction_db.p", "wb" ) )

