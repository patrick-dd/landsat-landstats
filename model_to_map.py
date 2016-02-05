###
### Converting model estimates to county estimates
###
import numpy as np
import pandas as pd
from osgeo import gdal, ogr
from model_output import *
from shapely.geometry import Point, Polygon, shape
import shapely.ops as ops
from rtree import index 
from geopandas import GeoDataFrame
from functools import partial
import pyproj
import pickle
import cPickle
import h5py
from sklearn.feature_extraction.image import extract_patches_2d
###
### we can just import all the files and run them through keras.
###
### surrounding images are saved in a folder

def pixelToCoordinates(geotransform, column, row):
	"Gets lat lon coordinates from pixel position"
	x_origin = geotransform[0]
	y_origin = geotransform[3]
	pixel_width = geotransform[1]
	pixel_height = geotransform[5]
	rotation_x = geotransform[2]
	rotation_y = geotransform[4]
	#
	lon_coord = x_origin + (column * pixel_width) + (row * rotation_x)
	lat_coord = y_origin + (column * rotation_y) + (row * pixel_height)
	#
	return (lon_coord, lat_coord)


def satelliteImageToDatabase(sat_folder_loc, state_name, year, channels):
	data = []
	data_3d = []
	count = 0
	for extension in channels:
		filename = sat_folder_loc + state_name + '_' + year + '_' + extension + '.tif'
		satellite_gdal = gdal.Open(filename)
		# getting data
		if extension == 'B1':
			ncols, nrows = satellite_gdal.RasterXSize, satellite_gdal.RasterYSize
			cols_grid, rows_grid = np.meshgrid(range(0,ncols), range(0,nrows))
			rows_grid, cols_grid = rows_grid.flatten(), cols_grid.flatten()
			# getting a series of lat lon points for each pixel
			location_series = [Point(pixelToCoordinates(
				satellite_gdal.GetGeoTransform(), col, row)) \
				for (col, row) in zip(cols_grid, rows_grid)]
			# pixel data
			band = satellite_gdal.GetRasterBand(1)
			array = band.ReadAsArray()
			band_series = [array[row][col] for (col, row) in zip(cols_grid, rows_grid)]
			data.append(band_series)
			data_3d.append(np.array(band_series).reshape((nrows, ncols)))
		else:
			band = satellite_gdal.GetRasterBand(1)
			array = band.ReadAsArray()
			band_series = np.array([array[row][col] for (col, row) in zip(cols_grid, rows_grid)])
			data.append(band_series)
			data_3d.append(np.array(band_series).reshape((nrows, ncols)))
	db_image = GeoDataFrame({
		'location': location_series,
		'B1': data[0],
		'B2': data[1],
		'B3': data[2],
		'B4': data[3],
		'B5': data[4],
		'B6_VCID_2': data[5],
		'B7': data[6],
		'row': rows_grid,
		'column': cols_grid,
		})
	return db_image, satellite_gdal, data_3d, cols_grid, rows_grid

def sliceSatelliteImage(image, row, col, nrows, ncols, obs_size, buffer_size):
	"""
	Slices the satellite image to return a buffered 'image'
	Images for the keras model are size obs_size**2
	Returned buffered image is (obs_size + buffer)**2 around image
	"""
	offset = obs_size / 2 + buffer_size
	return image[  : ,	max(0, min(nrows - obs_size, row - offset)) : 
						min(nrows, max(obs_size, col+offset+1)), 
						max(0, min(ncols - obs_size, row - offset)) : 
						min(ncols, max(obs_size, col+offset+1))]

def extract_patches(image_array, obs_size):
	patches_data = []
	for i in range(0, 7):
		patches_data.append(extract_patches_2d(image_array[i,:,:], (obs_size, obs_size)))
	patches_data = np.array(patches_data)
	patches_data = np.swapaxes(patches_data, 0, 1)
	return patches_data

def run_model(images):
	"""
	Calls a python script and runs the model	
	"""
	pop_estimates = get_estimates(images)	
	return pop_estimates

### get surrounding images
def get_estimates_loop(image, cols_grid, rows_grid, nrows, ncols, buffer_size):
	estimates_array = []
	for (col, row) in zip(cols_grid, rows_grid):
		tmp_slice = sliceSatelliteImage(image, row, col, nrows, ncols, obs_size, buffer_size)
		patches = extract_patches(tmp_slice, obs_size)
		print patches.shape
		estimates = run_model(patches)
		pixel_estimate = np.mean(estimates)
		estimates_array.append(pixel_estimate)
	return estimates_array


sat_folder_loc = 'LANDSAT_TOA/Washington/'
state_name = 'Washington'
year = '2010'
channels = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6_VCID_2', 'B7']
buffer_size = 3
a, satellite_gdal, image, cols_grid, rows_grid = satelliteImageToDatabase(
					sat_folder_loc, state_name, year, channels)
ests = get_estimates_loop(np.array(image), cols_grid, rows_grid, satellite_gdal.RasterYSize, 
					satellite_gdal.RasterXSize, buffer_size)

pickle.dump(ests, open('pixel_ests.p', 'w') )


	

