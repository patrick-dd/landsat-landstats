"""
A script to create a database of images from the LANDSAT 7 database

Saves the data as a pickle file

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
import cPickle
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
			rows_grid, cols_grid = np.meshgrid(range(0,ncols), range(0,nrows))
			cols_grid, rows_grid = rows_grid.flatten(), cols_grid.flatten()
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
		})
	print np.array(data).shape
	return db_image, satellite_gdal, data

def urbanDatabase(urban_folder_loc, state_code):
	print 'Creating urban database'
	db_urban = GeoDataFrame.from_file(urban_folder_loc+'cb_2012_us_uac10_500k.shp')
	db_urban = db_urban[(db_urban['NAME10'].str.contains(state_code, case=True))]
	return db_urban

def satUrbanDatabase(db_urban, db_image):
	print 'Creating database of satellite image pixels with urban/rural info'
	urban_blocks = np.array(db_urban['geometry'])
	print 'Creating spatial index'
	idx = spatialIndex(urban_blocks)
	print 'Assigning urban/rural status to pixels'
	pixels_location = db_image['location']
	pixelPoint_urban = pointWithinPolygon(idx, pixels_location, urban_blocks)	
	pixelPoint_urban.columns = ['count','poly', 'urban', 
									 'latitude', 'longitude']
	db_image['urban'] = pixelPoint_urban['urban']
	db_image['latitude_u'] = pixelPoint_urban['latitude']
	db_image['longitude_u'] = pixelPoint_urban['longitude']
	return db_image

def sampling(sampling_rate, obs_size, satellite_gdal, db_image):
	print 'Sampling the database'
	nrows, ncols = satellite_gdal.RasterYSize, satellite_gdal.RasterXSize
	nobs = (nrows - obs_size) * (ncols - obs_size)
	# Getting the sum of urban pixels for each patch
	urban_array = db_image['urban'].fillna(0)
	urban_array = np.array(urban_array).reshape((nrows, ncols))
	urban_patches = extract_patches_2d(data_array, (obs_size, obs_size))
	urban_data = [np.sum(patch) for patch in urban_patches]
	urban_data = pd.DataFrame(urban_data)
	# adding lat, long
	urban_data['latitude'] =  db_image['latitude']
	urban_data['longitude'] =  db_image['longitude']
	# Creating sample weights 
	seed = 1996
	urban_rank = urban_data[0].rank(ascending=False)
	urban_data['rank'] = urban_rank
	sumrank = urban_data['rank'].sum()
	urban_data['weight'] = (urban_data['rank']) / sumrank
	urban_sample = urban_data[0].sample(
			int(len(urban_data[0]) * sampling_rate), 
			weights=sample_weights, replace=True)
	urban_sample_idx = np.array(urban_sample.index.values)
	urban_data['sample'] = urban_sample_idx
	urban_sample_idx.sort()
	print 'sample size ', len(urban_sample_idx)
	# adding sample indicator variable
	urban_data['sample'] = urban_sample
	urban_data.ix[urban_data['sample'] > 0, 'sample'] = 1
	urban_data.fillna(0)
	return urban_sample_idx, urban_data

def censusDatabase(census_folder_loc, census_shapefile):
	print 'Getting density'
	# Importing shapefile
	db_shape = GeoDataFrame.from_file(census_folder_loc+census_shapefile)
	# It turns out the earth isn't flat
	# Getting area in km**2
	area_sq_degrees = db_shape['geometry']
	area_sq_km = []
	for region in area_sq_degrees:
		geom_area = ops.transform(
			partial(
			pyproj.transform,
			pyproj.Proj(init='EPSG:4326'),
			pyproj.Proj(
				proj='aea',
				lat1=region.bounds[1],
				lat2=region.bounds[3])),
			region)
		area = geom_area.area / 1000000.0  #convert to km2
		area_sq_km.append( area )
	db_shape['area'] = area_sq_km
	db_shape['density'] = db_shape['POP10'] / db_shape['area']
	return db_shape
	
def mergeCensusSatellite(db_shape, db_image_old):
	blocks = db_shape['geometry']
	points = db_image_old['location']
	pop = db_shape['density']
	print 'Merging Census data with Satellite images'
	idx = spatialIndex(blocks)
	pixel_information = pointWithinPolygonPop(idx, points, blocks, pop)
	pixel_information.columns = ['count','poly', 'pop_density', 
								 'latitude', 'longitude']
	print pixel_information.head()
	db_image_old['pop_density'] = pixel_information['pop_density']
	db_image_old['latitude'] = pixel_information['latitude']
	db_image_old['longitude'] = pixel_information['longitude']
	db_image_new = db_image_old
	return db_image_new
	

def sampleExtractor(data_array, sample_idx, obs_size, axis=None):
	"""
	Takes the image array and sample indices, returns a sample for
	keras use
	"""
	images = extract_patches_2d(data_array, (obs_size, obs_size))
	image_sample = np.take(images, sample_idx, axis=axis)
	return image_sample

def sampleGenerator(obs_size, db_image, channels, 
					satellite_gdal, urban_sample_idx):
	print 'Creates the Keras ready sample'
	ncols = satellite_gdal.RasterXSize
	nrows = satellite_gdal.RasterYSize
	output_data_full = []
	pop_output_data = []
	pop_array = db_image['pop_density'].fillna(0)
	pop_array = pop_array.reshape((nrows, ncols))
	image_array = []
	for channel in channels:
		image_array.append( np.array(db_image[channel]).reshape((nrows, ncols)) )
	image_array = np.array(image_array)
	print 'Loop to construct sample'
	np.random.shuffle(urban_sample_idx)
	tmp_data = []
	for i in range(0, 7):
		output_data_full.append(sampleExtractor(image_array[i,:,:], urban_sample_idx, obs_size, axis=0))
	output_data_full = np.swapaxes(output_data_full, 0, 1)
	print 'Shapes of pop_array and tmp_pop'
	tmp_pop = sampleExtractor(pop_array, urban_sample_idx, obs_size, axis=0)
	for i in range(0, len(urban_sample_idx)):
		obs_pop = np.mean(tmp_pop[i])
		pop_output_data.append(obs_pop)
	pop_output_data = np.nan_to_num(np.array(pop_output_data))
	output_data_full = np.array(output_data_full)
	return output_data_full, pop_output_data

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
		f = h5py.File(save_folder_loc + 'db_' + state_name + '_y_%d.hdf5'% count, 'w')
		f.create_dataset('data', data = temp, compression="gzip")
		f.close()
		if file_size!=(y.shape[0]-1):
			y = y[file_size:]
		count += 1

def databaseConstruction(census_folder_loc, census_shapefile, urban_folder_loc,
		sat_folder_loc, save_folder_loc, state_name, state_code, year, channels,
		file_size, sample_rate, obs_size):
	print 'Constructing database'
	db_image, satellite_gdal = satelliteImageToDatabase(sat_folder_loc, state_name, year, channels)
	print db_image.head()
	db_urban = urbanDatabase(urban_folder_loc, state_code)
	db_image = satUrbanDatabase(db_urban, db_image)
	urban_sample_idx, knn_data = sampling(sample_rate, obs_size, satellite_gdal, db_image)
	cPickle.dump(knn_data, file('knn_X_data.save', 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
	db_shape = censusDatabase(census_folder_loc, census_shapefile)
	db_image = mergeCensusSatellite(db_shape, db_image)
	X, y = sampleGenerator(obs_size, db_image, channels, 
						satellite_gdal, urban_sample_idx)
	saveFiles_y(y, file_size, save_folder_loc, state_name)
	saveFiles_X(X, file_size, save_folder_loc, state_name)



