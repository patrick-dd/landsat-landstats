"""

A script to create a database of images from the LANDSAT 7 database

Saves the data as a hdf5 file

"""
import numpy as np
import pandas as pd
import cPickle
import h5py
from osgeo import gdal, ogr
import pyproj
from shapely.geometry import Point, Polygon, shape
import shapely.ops as ops
from rtree import index 
from geopandas import GeoDataFrame
from functools import partial
from sklearn.feature_extraction.image import extract_patches_2d


def pixelToCoordinates(geotransform, column, row):
	"""
	Returns lat lon coordinates from pixel position 
	using an affine transformation
	Input:
		geotransfrom: coefficient vector
		column, row: integers
	Returns:
		lat, lon projection coordinates
	"""
	x_origin = geotransform[0]
	y_origin = geotransform[3]
	pixel_width = geotransform[1]
	pixel_height = geotransform[5]
	rotation_x = geotransform[2]
	rotation_y = geotransform[4]
	# The affine transformation
	lon_coord = x_origin + (column * pixel_width) + (row * rotation_x)
	lat_coord = y_origin + (column * rotation_y) + (row * pixel_height)
	#
	return (lon_coord, lat_coord)

def spatialIndex(blocks):
	"""
	Input:
		blocks: an array of shapely polygons
	Returns:
		idx: an Rtree index
	"""
	idx = index.Index()
	count = 0
	for block in blocks:
		idx.insert(count, block.bounds)
		count += 1
	return idx

def pointWithinPolygonPop(idx, points, polygons, pop):
	"""
	Finds the census tract containing pixels
	Inputs:
		idx: an Rtree spatial index instance
		points: an array of Shapely points 
		polygons: an array of Shapely polygons
		pop: an array of population, of same ordering as polygons!
	Returns:
		A GeoDataFrame with points, polycounts and population 
	"""
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
	"""
	Finds whether a pixel is urban or not
	This is useful for generating sample weights 
	Inputs:
		idx: an Rtree spatial index instance
		points: an array of Shapely points 
		polygons: an array of Shapely polygons
	Returns:
		A GeoDataFrame with points, polycounts and an urban dummy 
	"""
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
	"""
	Converts satellite images to a GeoDataFrame
	The satellite image used here is the 2010 LANDSAT 7 TOA composite
	Inputs:
		sat_folder_loc: location of satellite image
		state_name: name of state's image you're converting
		year: year of image
		channels: channels of satellite image to keep
	Returns:
		df_image: GeoDataFrame with pixel pixel_information
		nrows, ncols: size of satellite image
	"""
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
	df_image = GeoDataFrame({
		'location': location_series,
		'B1': data[0],
		'B2': data[1],
		'B3': data[2],
		'B4': data[3],
		'B5': data[4],
		'B6_VCID_2': data[5],
		'B7': data[6],
		})
	return df_image, nrows, ncols

def urbanDatabase(urban_folder_loc, state_code):
	"""
	Creates the GeoDataFrame for urbanness
	Inputs:
		urban_folder_loc: location of urban data
		state_code: 2 letter code of state we're investigating
	Returns:
		df_urban: GeoDataFrame for state state_code
	"""
	df_urban = GeoDataFrame.from_file(urban_folder_loc+'cb_2012_us_uac10_500k.shp')
	df_urban = df_urban[(df_urban['NAME10'].str.contains(state_code, case=True))]
	return df_urban

def satUrbanDatabase(df_urban, df_image):
	"""
	Combines satellite and urban databaseConstruction
	Inputs:
		df_urban: GeoDataFrame with urban measure
		df_image: GeoDataFrame with image information 
	Returns:
		df_image: combined GeoDataFrame
	"""
	urban_blocks = np.array(df_urban['geometry'])
	idx = spatialIndex(urban_blocks)
	pixels_location = df_image['location']
	pixelPoint_urban = pointWithinPolygon(idx, pixels_location, urban_blocks)	
	pixelPoint_urban.columns = ['count','poly', 'urban', 
									 'latitude', 'longitude']
	df_image['urban'] = pixelPoint_urban['urban']
	df_image['latitude_u'] = pixelPoint_urban['latitude']
	df_image['longitude_u'] = pixelPoint_urban['longitude']
	return df_image

def sampling(sampling_rate, obs_size, nrows, ncols, df_image):
	"""
	Constructs a weighted sample of images from the GeoDataFrame
	Inputs:
		sampling_rate: proportion of images sampled  (float)
		obs_size: model uses images of size obs_size**2 ()
		nrows, ncols: dimensions of satellite data 
		df_image: GeoDataFrame with information 
	Returns:
		urban_sample_idx: index of sampled images 
		df_sample: sampled images 
	"""
	# Getting the sum of urban pixels for each patch
	urban_array = df_image['urban'].fillna(0)
	urban_array = np.array(urban_array).reshape((nrows, ncols))
	urban_patches = extract_patches_2d(urban_array, (obs_size, obs_size))
	urban_count = [np.sum(patch) for patch in urban_patches]
	df_sample = pd.DataFrame(urban_count)
	# collecting indices of north west corner
	indices = np.arange(nrows*ncols).reshape((nrows, ncols))
	indices = indices[ : -obs_size + 1, : -obs_size + 1 ]
	df_sample['index'] = df_sample.ravel()
	# Creating sample weights 
	seed = 1996
	urban_rank = df_sample[0].rank(ascending=False)
	df_sample['rank'] = urban_rank
	sumrank = df_sample['rank'].sum()
	df_sample['weight'] = (df_sample['rank']) / sumrank
	urban_sample = df_sample[0].sample(
			int(len(df_sample[0]) * sampling_rate), 
			weights=df_sample['weight'], replace=True)
	urban_sample_idx = np.array(urban_sample.index.values)
	df_sample = df_sample.ix[urban_sample_idx]
	urban_sample_idx.sort()
	return urban_sample_idx, df_sample

def censusDatabase(census_folder_loc, census_shapefile):
	"""
	Gets population density from census data
	Inputs:
		census_folder_loc: location of data (string)		
		census_shapefile: name of shapefile (string)
	Returns:
		df_census: GeoDataFrame with census information
	"""
	# Importing shapefile
	df_census = GeoDataFrame.from_file(census_folder_loc+census_shapefile)
	# It turns out the earth isn't flat
	# Getting area in km**2
	area_sq_degrees = df_census['geometry']
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
	df_census['area'] = area_sq_km
	df_census['density'] = df_census['POP10'] / df_census['area']
	return df_census
	
def mergeCensusSatellite(df_census, df_image):
	"""
	Merges census data with satellite data
	Inputs:
		df_census: census GeoDataFrame
		df_image: image GeoDataFrame
	Returns:
		df_image: image GeoDataFrame with census data 
	"""
	blocks = df_census['geometry']
	points = df_image_old['location']
	pop = df_census['density']
	idx = spatialIndex(blocks)
	pixel_information = pointWithinPolygonPop(idx, points, blocks, pop)
	pixel_information.columns = ['count','poly', 'pop_density', 
								 'latitude', 'longitude']
	df_image['pop_density'] = pixel_information['pop_density']
	df_image['latitude'] = pixel_information['latitude']
	df_image['longitude'] = pixel_information['longitude']
	return df_image
	

def sampleExtractor(data_array, sample_idx, obs_size, axis=None):
	"""
	Extracts a sample of images
	Inputs:
		data_array: satellite images 
		sample_idx: index of images to be sampled 
		obs_size: size of sampled images
		axis: axis along which samples are taken
	Returns:
		image_sample: numpy array of images. Keras ready!
	"""
	images = extract_patches_2d(data_array, (obs_size, obs_size))
	image_sample = np.take(images, sample_idx, axis=axis)
	return image_sample

def sampleGenerator(obs_size, df_image, channels, nrows, 
					ncols, urban_sample_idx):
	"""
	Constructs a sample of observations that Keras can play with 
	We take the mean population density of each image 
	Inputs:
		obs_size: size of images that Keras reads 
		df_image: image GeoDataFrame
		channels: satellite bandwidth channels
		nrows, ncols: dimensions of satellite image 
		urban_sample_idx: index of images to be sampled
	Returns:
		image_output_data: image data (X variable)
		pop_output_data: population data (y variable)
	Note: Keras uses the last x percent of data in cross validation 
	Have to shuffle here to ensure that the last ten percent isn't just
	the southern most rows of information
	"""
	image_output_data = []
	pop_output_data = []
	pop_array = df_image['pop_density'].fillna(0)
	pop_array = pop_array.reshape((nrows, ncols))
	image_array = []
	for channel in channels:
		image_array.append( np.array(df_image[channel]).reshape((nrows, ncols)) )
	image_array = np.array(image_array)
	np.random.shuffle(urban_sample_idx)
	tmp_data = []
	for i in range(0, 7):
		image_output_data.append(sampleExtractor(image_array[i,:,:], urban_sample_idx, obs_size, axis=0))
	image_output_data = np.swapaxes(image_output_data, 0, 1)
	tmp_pop = sampleExtractor(pop_array, urban_sample_idx, obs_size, axis=0)
	for i in range(0, len(urban_sample_idx)):
		# We take the mean pop density
		obs_pop = np.mean(tmp_pop[i])
		pop_output_data.append(obs_pop)
	pop_output_data = np.nan_to_num(np.array(pop_output_data))
	image_output_data = np.array(image_output_data)
	return image_output_data, pop_output_data

def saveFiles_X(X, file_size, save_folder_loc, state_name):
	"""
	Saves the image information
	Inputs:
		X: numpy array of image samples
		file_size: number of image samples per file 
		save_folder_loc: location to save data
		state_name: state you're sampling 
	Returns:
		Nothing, it just saves the data! 
	"""
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
	"""
	Saves the population information
	Inputs:
		y: numpy array of population samples
		file_size: number of image samples per file (int)
		save_folder_loc: location to save data
		state_name: state you're sampling 
	Returns:
		Nothing, it just saves the data! 
	"""
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
	"""
	Constructs the data
	Inputs:
		census_folder_loc: location of data (string)
		census_shapefile: name of shapefile (string)
		urban_folder_loc: location of urban data (string)
		sat_folder_loc: location of satellite images (string)
		save_folder_loc: location of folder to save data (string)
		state_name: The state name (string)
		state_code: Two letter state code (string) 
		year: year of investigation string  
		channels: bandwidths used in estimation list of strings
		file_size: number of image samples per file (int)
		sample_rate: proportion of images sampled (float)
		obs_size: the length/width of each observation (int)
	Returns:
		Data for you to play with  :)
	"""
	df_image, nrows, ncols = satelliteImageToDatabase(sat_folder_loc, state_name, year, channels)
	df_urban = urbanDatabase(urban_folder_loc, state_code)
	df_image = satUrbanDatabase(df_urban, df_image)
	urban_sample_idx, knn_data = sampling(sample_rate, obs_size, nrows, ncols, df_image)
	cPickle.dump(knn_data, file('knn_X_data.save', 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
	df_census = censusDatabase(census_folder_loc, census_shapefile)
	df_image = mergeCensusSatellite(df_census, df_image)
	X, y = sampleGenerator(obs_size, df_image, channels, nrows, ncols, urban_sample_idx)
	saveFiles_y(y, file_size, save_folder_loc, state_name)
	saveFiles_X(X, file_size, save_folder_loc, state_name)



