## Create the full database!

"""
A script to create a database of images from the LANDSAT 7 database

This is to generate images for output. So the full state!

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
import pickle
import h5py
from sklearn.feature_extraction.image import extract_patches_2d

def inner_patch_extractor(array, patch_size, overlap):
	"""
	Extract overlapping patches from an array
	Input:
	 		array: a numpy array of size (patch_size, N)
			patch_size: the width & height in pixels of the patch_size
			overlap: the allowable overlap in pixels between patches
	"""
	d = []
	for i in range(0, p_rows):
		d.append(x[0:patch_size, :])
		x = np.delete(x, range(0, (patch_size - overlap)), axis=0)


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

def spatialIndex(blocks):
	idx = index.Index()
	count = 0
	for block in blocks:
		idx.insert(count, block.bounds)
		count += 1
	return idx

def pointWithinPolygonPop(idx, points, polygons, pop):
	print 'Assigning points to polygons'
	pixelPoint_db = []
	count = 0
	for pixel in points:
		temp_polygon = None
		temp_pop = None
		for j in idx.intersection((pixel.x, pixel.y)):
			if pixel.within(polygons[j]):
				temp_polygon = polygons[j]
				temp_pop = pop[j] 
				break
		pixelPoint_db.append([count, temp_polygon, temp_pop, pixel.x, pixel.y])
		count += 1
	return GeoDataFrame(pixelPoint_db)

def pointWithinPolygon(idx, points, polygons):
	print 'Assigning points to polygons'
	pixelPoint_db = []
	count = 0
	for pixel in points:
		temp_polygon = None
		temp_urban = None
		for j in idx.intersection((pixel.x, pixel.y)):
			if pixel.within(polygons[j]):
				temp_polygon = polygons[j]
				temp_urban = 1
				break
		pixelPoint_db.append([count, temp_polygon, temp_urban, pixel.x, pixel.y])
		count += 1
	return GeoDataFrame(pixelPoint_db)

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
			location_series = np.array(location_series).reshape((nrows, ncols))
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
	return location_series, data

def sampleGenerator(obs_size, location_series, data):
	print 'Creates the Keras ready sample'
	output_data_full = []
	tmp_data = []
	for i in range(0, 7):
		output_data_full.append(extract_patches_2d(image_array[i,:,:], (obs_size, obs_size)))
	output_data_full = np.array(output_data_full)
	print 'Shapes of full output data', output_data_full.shape
	location_data_full =  extract_patches_2d(location_series, (obs_size, obs_size))
	return output_data_full, location_data_full

def saveFiles_X(X, file_size, save_folder_loc, state_name):
	print 'Saving files'
	no_files = 1 + X.shape[0] / file_size 
	count = 0
	print 'Number of files', no_files
	for i in range(0, no_files):
		# file size changes for X and y
		temp = X[0:file_size, :, :, :]
		f = h5py.File(save_folder_loc + 'db_' + state_name + '_X_%d.hdf5' % count, 'w')
		f.create_dataset('data', data = temp, compression="gzip")
		f.close()
		if file_size!=(X.shape[0]-1):
			X = X[file_size:, :, :, :]
		count += 1

def saveFiles_y(y, file_size, save_folder_loc, state_name):
	print 'Saving files'
	no_files = 1 + y.shape[0] / file_size 
	count = 0
	print 'Number of files', no_files
	for i in range(0, no_files):
		# file size changes for X and y
		temp = y[0:file_size]
		f = h5py.File(save_folder_loc + 'db_' + state_name + '_location_%d.hdf5'% count, 'w')
		f.create_dataset('data', data = temp, compression="gzip")
		f.close()
		if file_size!=(y.shape[0]-1):
			y = y[file_size:]
		count += 1

def databaseConstruction(sat_folder_loc, save_folder_loc, state_name, state_code, year, channels, file_size, 
		obs_size):
	print 'Constructing output database'
	location_series, data = satelliteImageToDatabase(sat_folder_loc, state_name, year, channels)
	image_data, location_data_full = sampleGenerator(obs_size, location_series, data)
	saveFiles_X(image_data, file_size, save_folder_loc, state_name)
	saveFiles_y(location_data_full, file_size, save_folder_loc, state_name)
