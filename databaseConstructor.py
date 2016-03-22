"""

A script to create a database of images from the LANDSAT 7 database

Saves the data as a hdf5 file

"""
import numpy as np
import pandas as pd
import cPickle
#import h5py
from osgeo import gdal, ogr
import pyproj
from shapely.geometry import Point, Polygon, shape
import shapely.ops as ops
from rtree import index 
from geopandas import GeoDataFrame
from functools import partial
from sklearn.feature_extraction.image import extract_patches_2d
from multiprocessing import Pool
import parmap
import os, sys

class databaseConstructor:
    """

    A class to take a 
        - satellite image (tiff)
        - population estimates (shp)
        - urban/rural location indicator (shp)
    And return smaller images (arrays) for use in Keras

    """
    def __init__(
            self, census_folder_loc, census_shapefile, urban_folder_loc,
            urban_shapefile, sat_folder_loc, save_folder_loc, state_name, 
            state_code, year, channels, file_size, sample_rate, obs_size, 
            slice_depth, processes):
        """
        Initialises the class
        Inputs:
            (string)
            census_folder_loc: location of data
            census_shapefile: name of shapefile 
            urban_folder_loc: location of urban data 
            urban_shapefile: name of shapefile
            sat_folder_loc: location of satellite images 
            save_folder_loc: location of folder to save data 
            state_name: The state name 
            state_code: Two letter state code  
            (int)
            year: year of investigation string  
            (list of str)
            channels: bandwidths used in estimation list of strings
            (int)
            file_size: number of image samples per file 
            (float)
            sample_rate: proportion of images sampled 
            (int)
            obs_size: the length/width of each observation 
        """
        self.census_folder_loc = census_folder_loc
        self.census_shapefile = census_shapefile
        self.urban_folder_loc = urban_folder_loc
        self.urban_shapefile = urban_shapefile
        self.sat_folder_loc = sat_folder_loc
        self.save_folder_loc = save_folder_loc
        self.state_name = state_name
        self.state_code = state_code
        self.year = year
        self.channels = channels
        self.file_size = file_size
        self.sample_rate = sample_rate
        self.obs_size = obs_size
        self.slice_depth = slice_depth
        self.processes = processes
        if not os.path.exists( self.census_folder_loc + '/' 
                                + self.census_shapefile ):
            sys.exit('Error: File ' + self.census_folder_loc + '/' 
                    + self.census_shapefile + ' does not exist')

    def pixel_to_coordinates(self, column, row):
        """

        Returns lat lon coordinates from pixel position 
        using an affine transformation

        See http://www.gdal.org/gdal_datamodel.html

        Input:
            (array) geotransfrom
            (int) column, row
        Returns:
            (float) lat, lon projection coordinates

        """
        x_origin = self.geotransform[0]
        y_origin = self.geotransform[3]
        pixel_width = self.geotransform[1]
        pixel_height = self.geotransform[5]
        rotation_x = self.geotransform[2]
        rotation_y = self.geotransform[4]
        # The affine transformation
        lon_coord = x_origin + (column * pixel_width) + (row * rotation_x)
        lat_coord = y_origin + (column * rotation_y) + (row * pixel_height)
        return (lon_coord, lat_coord)
    
    def spatialIndex(self, blocks):
        """
        Input:
            (array of shapely polygons) blocks
        Returns:
            (rtree index) idx
        """
        idx = index.Index()
        for count, block in enumerate(blocks):
            idx.insert(count, block.bounds)
        return idx

    def point_within_polygon_pop(self, idx, points, polygons, pop):
        """
        Finds the census tract containing pixels
        Inputs:
            (rtree spatial index instance) idx
            (array shapely points) points
            (array shapely polygons) polygons
            (array of population estimates) pop
        Returns:
            (geodataframe) points, polycounts and population 
        """
        pixelPoint_db = []
        for pixel in points:
            temp_polygon = None
            temp_pop = None
            for j in idx.intersection((pixel.x, pixel.y)):
                if pixel.within(polygons[j]):
                    temp_polygon = polygons[j]
                    temp_pop = pop[j]
                    break
            pixelPoint_db.append([temp_polygon, temp_pop, pixel.x, pixel.y])
        return GeoDataFrame(pixelPoint_db)

    def point_within_polygon_urban(self, idx, points, polygons):
        """
        Finds whether a pixel is urban or not
        This is useful for generating sample weights 
        Inputs:
            (rtree) idx: an Rtree spatial index instance
            (array of shapely points) points 
            (array of shapely polygons) polygons
        Returns:
            A GeoDataFrame with points, polycounts and an urban dummy 
        """
        pixelPoint_db = []
        for pixel in points:
            temp_polygon = None
            temp_urban = 0
            for j in idx.intersection((pixel.y, pixel.x)):
                if pixel.within(polygons[j]):
                    temp_polygon = polygons[j]
                    temp_urban = 1
                    break
            pixelPoint_db.append([temp_polygon, temp_urban, pixel.x, pixel.y])
        return GeoDataFrame(pixelPoint_db)

    def point_wrapper(self, x, y):
        """
        A wrapper to use the point function in parallel 
        """
        return Point((x, y))

    def array_wrapper(self, col, row, array):
        """
        A wrapper to use the point function in parallel 
        """
        return array[row][col]

    def get_location(self):
        """

        Extracts the location of each pixel in the satellite image

        """
        self.ncols = self.satellite_gdal.RasterXSize / 50
        self.nrows = self.satellite_gdal.RasterYSize / 50
        print 'Columns, rows', self.ncols, self.nrows
        rows_grid, cols_grid = np.meshgrid(
                    range(4 * ncols, 5 * ncols), 
                    range(2 * nrows, 3 * nrows))
        cols_grid, rows_grid = rows_grid.flatten(), cols_grid.flatten()
        # getting a series of lat lon points for each pixel
        print 'Getting geo data'
        self.geotransform = satellite_gdal.GetGeoTransform()
        print 'Getting locations'
        self.location_series = parmap.starmap(
                        self.pixel_to_coordinates, 
                        zip(cols_grid, rows_grid), 
                        self.geotransform, self.processes)
        print 'Converting to Points'
        self.location_series = parmap.starmap(
                        self.point_wrapper, 
                        zip(cols_grid, rows_grid), 
                        self.processes)

    def satelliteImageToDatabase(self):
        """
        Converts satellite images to a GeoDataFrame
        The satellite image used here is the 2010 LANDSAT 7 TOA composite
        Returns:
            (geodataframe) df_image
            (int) nrows, ncols. size of satellite image
        """
        data = []
        count = 0
        self.filename = self.sat_folder_loc + self.state_name + \
                       '_' + self.year + '_B1.tif'
        # testing whether files exist
        if not os.path.exists( self.filename ):
                sys.exit('Error: File ' + self.filename + 
                        ' does not exist')
        self.satellite_gdal = gdal.Open(self.filename)
        self.get_location()
        for extension in channels:
            self.filename = self.sat_folder_loc + self.state_name + \
                        '_' + self.year + '_' + self.extension + '.tif'
            # testing whether files exist
            if not os.path.exists( self.filename ):
                sys.exit('Error: File ' + self.filename + 
                        ' does not exist')
            # getting data
            print 'Loading bandwidth', extension
            band = satellite_gdal.GetRasterBand(1)
            array = band.ReadAsArray()
            band_series = parmap.starmap(
                    array_wrapper, zip(cols_grid, rows_grid), 
                    array, self.processes)
            data.append(band_series)
        self.df_image = GeoDataFrame({
        'location': location_series,
        'B1': data[0],
        'B2': data[1],
        'B3': data[2],
        'B4': data[3],
        #'B5': data[4],
        #'B6_VCID_2': data[5],
        #'B7': data[6],
        })
    
    def urban_database(self):
        """
        
        Creates the GeoDataFrame for urbanness
        
        """
        df_urban = GeoDataFrame.from_file(
                    self.urban_folder_loc+self.urban_shape_file)
        self.df_urban = df_urban[(
                    df_urban['NAME10'].str.contains(self.state_code, case=True))]
    
    def sat_urban_database(self):
        """
        
        Combines satellite and urban database construction
        
        """
        self.urban_blocks = np.array(self.df_urban['geometry'])
        print 'Amount of urban blocks: ', len(self.urban_blocks)
        self.idx = self.spatialIndex(self.urban_blocks)
        pixel_point_urban = self.point_within_polygon_urban(
                self.idx, self.location_series, self.urban_blocks) 
        pixel_point_urban.columns = ['poly', 'urban', 'latitude', 'longitude']
        self.df_image['urban'] = pixel_point_urban['urban']
        self.df_image['latitude_u'] = pixel_point_urban['latitude']
        self.df_image['longitude_u'] = pixel_point_urban['longitude']
    
    def image_slicer(image):
        """
        Cuts the larger satellite image into smaller images 
        A less intense version of extract_patches_2d
        Inputs:
        - image: the geotiff image 
        - obs_size: the observation size 
        - overlap: proportion of image you want to overlap
        Returns:
        - patches: the tiles
        - indices: index of the nw corner of each patch 
        """
        self.patches = []
        step = int(self.obs_size * self.overlap)
        self.indices = []
        if self.slice_depth == 1:
            for y in range(0, self.nrows, step):
                for x in range(0, self.ncols, step):
                    mx = min(x+self.obs_size, self.ncols)
                    my = min(y+self.obs_size, self.nrows)
                    tile = image[ y: my, x: mx ]
                    if tile.shape == (self.obs_size, self.obs_size):
                        self.patches.append(tile)
                        self.indices.append((x, y))
        else:
            for y in range(0, self.nrows, step):
                for x in range(0, self.ncols, step):
                    mx = min(x+obs_size, self.ncols)
                    my = min(y+obs_size, self.nrows)
                    tile = image[ :, y : my, x: mx ]
                    if tile.shape == (self.slice_depth, 
                            self.obs_size, self.obs_size):
                        self.patches.append(tile)
                        self.indices.append((x, y))
        self.patches = np.array(self.patches)
        self.indices = np.array(self.indices)

    def adder(self, x):
        """
        Adds two variables. Useful in parallelising operation
        Takes the north west corner of an image and returns the centroid
        Inputs: 
        x: north west corner
        midpoint: half the length of an image    
        Returns:
        image midpoint
        """
        return x + (self.obs_size / 2)

    def sampling(self):
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
        
        Note: Keras uses the last x percent of data in cross validation 
            Have to shuffle here to ensure that the last ten percent isn't just
            the southern most rows of information
        """
        # Getting the sum of urban pixels for each patch
        urban_array = self.df_image['urban'].fillna(0)
        urban_array = np.array(urban_array).reshape((self.nrows, self.ncols))
        print 'Pre slicing max min'
        print urban_array.max()
        print urban_array.min()
        print 'extract patches'
        urban_patches, u_indices = self.image_slicer(urban_array)
        print 'counting urban'
        urban_count = np.array([np.sum(patch) for patch in urban_patches])
        print 'After slicing'
        print urban_count.max()
        print urban_count.min()
        df_sample = pd.DataFrame(urban_count)
        # Getting the locations
        print 'get locations for individual frames'
        mid_point = self.obs_size / 2
        pool = Pool(self.processes)
        cols_grid = pool.map(self.adder, u_indices[:,0])
        rows_grid = pool.map(self.adder, u_indices[:,1])
        print 'location series'
        self.frame_location_series = parmap.starmap(
                pixelToCoordinates,
                zip(cols_grid, rows_grid), self.geotransform, processes=8)
        print 'Converting to Points'
        self.frame_location_series = \
                pool.map(Point, self.frame_location_series)
        df_sample['location'] = location_series
        # Creating sample weights 
        seed = 1996
        print df_sample[0].hist
        urban_rank = df_sample[0].rank(ascending=False)
        # weighting the urban areas way more heavily
        urban_rank = [ u**5 for u in urban_rank ]
        df_sample['rank'] = urban_rank
        sumrank = df_sample['rank'].sum()
        df_sample['weight'] = (df_sample['rank']) / sumrank
        urban_sample = df_sample[0].sample(
            int(len(df_sample[0]) * sampling_rate), 
            weights=df_sample['weight'], replace=True)
        self.urban_sample_idx = np.array(urban_sample.index.values)
        self.df_sample = df_image.ix[urban_sample_idx]
        self.urban_sample_idx.sort()
        # shuffling indices for Keras cross validation
        np.random.shuffle(self.urban_sample_idx)
           
    def census_database(self):
        """
        Gets population density from census data
        Inputs:
        census_folder_loc: location of data (string)    
        census_shapefile: name of shapefile (string)
        Returns: 
            df_census: GeoDataFrame with census information
        """ 
        ## Importing shapefile 
        self.df_census = GeoDataFrame.from_file(self.census_folder_loc + 
                                        self.census_shapefile) 
        # It turns out the earth isn't flat 
        # Getting area in km**2 
        area_sq_degrees = df_census['geometry'] 
        area_sq_km = [] 
        for region in area_sq_degrees: 
            geom_area = ops.transform( partial( 
                pyproj.transform, pyproj.Proj(init='EPSG:4326'), 
                pyproj.Proj( proj='aea', 
                    lat1=region.bounds[1], lat2=region.bounds[3])), region) 
        area = geom_area.area / 1000000.0  #convert to km2
        area_sq_km.append( area )
        self.df_census['area'] = area_sq_km
        self.df_census['density'] = self.df_census['POP10'] /\
                self.df_census['area'] 

    def merge_census_satellite(self): 
        """ 
        Merges census data with satellite data 
        Inputs: 
            df_census: census GeoDataFrame
            df_image: image GeoDataFrame
        Returns: 
            df_image: image GeoDataFrame with census data
        """
        blocks = self.df_census['geometry'] 
        points = self.df_image['location'] 
        pop = self.df_census['density']
        idx = self.spatialIndex(blocks)
        pixel_information = \
                self.point_within_polygon_pop(idx, points, blocks, pop)
        pixel_information.columns = \
                ['poly', 'pop_density', 'latitude', 'longitude']
        self.df_image['pop_density'] = pixel_information['pop_density']
        self.df_image['latitude'] = pixel_information['latitude']
        self.df_image['longitude'] = pixel_information['longitude']

    def sample_extractor(data_array, axis=None):
        """

        Extracts a sample of images
            (array) data_array, satellite images
            (axis) axis of array
            (array) image_sample, Keras ready numpy array of images
        
        """
        patches, indices = image_slicer(data_array)
        print 'patches.shape: ', patches.shape
        self.image_sample = np.take(patches, self.urban_sample_idx, axis=0)

    def sample_generator_pop(self):
        """
        
        Constructs a sample of observations that Keras can play with 
        We take the mean population density of each image 
        
        """
        # getting population data
        pop_output_data = []
        pop_array = self.df_image['pop_density'].fillna(0)
        pop_array = pop_array.reshape((self.nrows, self.ncols))
        tmp_pop = sample_extractor(pop_array, self.urban_sample_idx, axis=0)
        for i in range(0, len(urban_sample_idx)):
        # We take the mean pop density
            obs_pop = np.mean(tmp_pop[i])
            pop_output_data.append(obs_pop)
        self.pop_output_data = np.nan_to_num(np.array(pop_output_data))
        
    def sample_generator_sat(self):
        """
        
        Constructs a sample of observations that Keras can play with 
        
       """
        # satellite image data
        image_array = []
        for channel in self.extensions:
            tmp_img = np.array(
                    self.df_image[channel]).reshape((self.nrows, self.ncols))
            tmp_img =  self.sample_extractor(
                    tmp_img[i,:,:], urban_sample_idx, axis=1)
            image_array = np.array(tmp_img)
        image_array = np.swapaxes(image_array, 0, 1)
        self.image_output_data = np.array(image_array)
    
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
        no_files = 1 + self.image_output_data.shape[0] / self.file_size 
        # the count is for a non 'full' file of the size self.file_size
        count = 0
        print 'Number of files', no_files
        for i in range(0, no_files):
            # file size changes for X and y
            temp = self.image_output_data[0:self.file_size, :, :, :]
            f = h5py.File(
                    self.save_folder_loc + 'db_' + self.state_name + \
                            '_X_%d.hdf5' % count, 'w')
            f.create_dataset('data', data = temp, compression="gzip")
            f.close()
            if file_size!=(self.image_output_data.shape[0]-1):
                self.image_output_data = \
                        self.image_output_data[self.file_size:, :, :, :]
            count += 1

    def save_files_y(self):
        """
        
        Saves the population information
        
        """
        no_files = 1 + self.pop_output_data.shape[0] / self.file_size 
        count = 0
        print 'Number of files', no_files
        for i in range(0, no_files):
            # file size changes for X and y
            temp = y[0:file_size]
            f = h5py.File(
                    self.save_folder_loc + 'db_' + self.state_name + \
                            '_y_%d.hdf5'% count, 'w')
            f.create_dataset('data', data = temp, compression="gzip")
            f.close()
            if file_size!=(y.shape[0]-1):
                y = y[file_size:]
            count += 1

#def databaseConstruction(census_folder_loc, census_shapefile, urban_folder_loc,
#    sat_folder_loc, save_folder_loc, state_name, state_code, year, channels,
#    file_size, sample_rate, obs_size):
#    """
#    Constructs the data
#    Inputs:
#    census_folder_loc: location of data (string)
#    census_shapefile: name of shapefile (string)
#    urban_folder_loc: location of urban data (string)
#    sat_folder_loc: location of satellite images (string)
#    save_folder_loc: location of folder to save data (string)
#    state_name: The state name (string)
#    state_code: Two letter state code (string) 
#    year: year of investigation string  
#    channels: bandwidths used in estimation list of strings
#    file_size: number of image samples per file (int)
#    sample_rate: proportion of images sampled (float)
#    obs_size: the length/width of each observation (int)
#    Returns:
#    Data for you to play with  :)
#    """
#    print 'Collecting satellite data'
#    df_image, nrows, ncols, satellite_gdal = \
#        satelliteImageToDatabase(sat_folder_loc, state_name, year, channels)
#    cPickle.dump(df_image, file('to_interpolate_data.save', 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
#    print 'Organising urban data'
#    df_urban = urbanDatabase(urban_folder_loc, state_code)
#    print 'Combining dataframes'
#    df_image = satUrbanDatabase(df_urban, df_image)
#    print 'Sampling'
#    urban_sample_idx, knn_data = sampling(sample_rate, obs_size, nrows, ncols, 
#                        df_image, satellite_gdal)
#    cPickle.dump(knn_data, file('knn_X_data.save', 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
#    print 'Collecting census data'
#    df_census = censusDatabase(census_folder_loc, census_shapefile)
#    print 'Merging dataframes'
#    df_image = mergeCensusSatellite(df_census, df_image)
#    print 'Creating samples for Keras'
#    X, y = sampleGenerator(obs_size, df_image, channels, nrows, ncols, urban_sample_idx)
#    print 'Saving files'
#    saveFiles_y(y, file_size, save_folder_loc, state_name)
#    saveFiles_X(X, file_size, save_folder_loc, state_name)
#    print 'Job done!'
