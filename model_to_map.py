###
### Converting model estimates to county estimates
###
import numpy as np
import pandas as pd
from osgeo import gdal, ogr
from shapely.geometry import Point, Polygon, shape
import shapely.ops as ops
from rtree import index 
from geopandas import GeoDataFrame
from functools import partial
import pyproj
import pickle
import h5py
from sklearn.feature_extraction.image import extract_patches_2d
###
### we can just import all the files and run them through keras.
###
### surrounding images are saved in a folder

def satelliteImageToDatabase(sat_folder_loc, state_name, year, channels):
	data = []
	count = 0
	for extension in channels:
		filename = sat_folder_loc + state_name + '_' + year + '_' + extension + '.tif'
		satellite_gdal = gdal.Open(filename)
		# getting data
		if extension == 'B1':
			ncols, nrows = satellite_gdal.RasterXSize, satellite_gdal.RasterYSize
			cols_grid, cols_grid = np.meshgrid(range(0,ncols), range(0,nrows))
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
		else:
			band = satellite_gdal.GetRasterBand(1)
			array = band.ReadAsArray()
			band_series = np.array([array[row][col] for (col, row) in zip(cols_grid, rows_grid)])
			data.append(band_series)

	db_image = GeoDataFrame({
		'location': location_series,
		'B1': data[0],
		'B2': data[1],
		'B3': data[2],
		'B4': data[3],
		'B5': data[4],
		'B6_VCID_2': data[5],
		'B7': data[6],
		'row': rows_grid
		'column': cols_grid
		})
	print np.array(data).shape
	return db_image, satellite_gdal, data

def sliceSatelliteImage(image, row, col, nrows, ncols, obs_size, buffer_size):
	"""
	Slices the satellite image to return a buffered 'image'
	Images for the keras model are size obs_size**2
	Returned buffered image is (obs_size + buffer)**2 around image
	"""
	offset = obs_size / 2 + buffer_size
	return image[  : ,	max(0, row - offset) : min(nrows, col+offset+1), 
						max(0, row - offset) : min(ncols, col+offset+1)]

def extract_patches(image_array, obs_size):
	patches_data = []
	for i in range(0, 7):
		patches_data.append(extract_patches_2d(image_array[i,:,:], (obs_size, obs_size)))
	patches_data = np.array(patches_data)
	return patches_data

def run_model(images):
	"""
	
	"""
	return pop_estimates

### get surrounding images
def get_estimates():
	estimates_array = []
	for (col, row) in zip(cols_grid, rows_grid):
		tmp_slice = sliceSatelliteImage(image, row, col, nrows, ncols, obs_size, buffer_size)
		patches = extract_patches(image_array, obs_size)
		estimates = run_model(patches)
		pixel_estimate = np.mean(estimates)
		estimates_array.append(pixel_estimate)





	

