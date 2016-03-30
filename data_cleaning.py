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
from multiprocessing import Pool
import parmap
import os, sys


def point_wrapper(x):
    """
    A wrapper to use the point function in parallel 
    """
    return Point(x[0], x[1])

 
def pixel_to_coordinates(column, row, geotransform):
    """

    Returns lat lon coordinates from pixel position 
    using an affine transformation

    See http://www.gdal.org/gdal_datamodel.html

    Outside of the class because it's used in parallel

    Input:
        (array) geotransfrom
        (int) column, row
    Returns:
        (float) lat, lon projection coordinates

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
    return (lon_coord, lat_coord)

def array_wrapper(col, row, array):
    """
    A wrapper to use the point function in parallel 
    """
    return array[row][col]

def adder(x):
    """
   
    Wrapper for parallelisation

    Takes the north west corner of an image and returns the centroid
    
    """
    return x + 16 


class database_constructor:
    """

    A class to take a 
        - satellite image (tiff)
        - population estimates (shp)
        - urban/rural location indicator (shp)
    And return smaller images (arrays) for use in Keras

    """
    def __init__(
            self, census_folder_loc, census_shapefile,
            sat_folder_loc, save_folder_loc, state_name, 
            state_code, year, channels, file_size, sample_rate, obs_size, 
            processes, step):
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
        self.sat_folder_loc = sat_folder_loc
        self.save_folder_loc = save_folder_loc
        self.state_name = state_name
        self.state_code = state_code
        self.year = year
        self.channels = channels
        self.file_size = file_size
        self.sample_rate = sample_rate
        self.obs_size = obs_size
        self.processes = processes
        self.step = step
        if not os.path.exists( self.census_folder_loc + '/' 
                                + self.census_shapefile ):
            sys.exit('Error: File ' + self.census_folder_loc + '/' 
                    + self.census_shapefile + ' does not exist')
        self.filename = self.sat_folder_loc + self.state_name + \
                       '_' + self.year + '_B1.tif'
        # testing whether files exist
        if not os.path.exists( self.filename ):
                sys.exit('Error: File ' + self.filename + 
                        ' does not exist')
        self.satellite_gdal = gdal.Open(self.filename)

        
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

    def point_within_polygon(self, idx, points, polygons, pop):
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

    def get_location(self):
        """

        Extracts the location of each pixel in the satellite image

        """
        self.ncols = self.satellite_gdal.RasterXSize
        self.nrows = self.satellite_gdal.RasterYSize 
        print 'Columns, rows', self.ncols, self.nrows
        rows_grid, cols_grid = np.meshgrid(
                    range(0, self.ncols), 
                    range(0, self.nrows))
        self.cols_grid = cols_grid.flatten()
        self.rows_grid = rows_grid.flatten()
        # getting a series of lat lon points for each pixel
        self.geotransform = self.satellite_gdal.GetGeoTransform()
        print 'Getting locations'
        self.location_series = np.array(parmap.starmap(
                        pixel_to_coordinates, 
                        zip(self.cols_grid, self.rows_grid), 
                        self.geotransform,
                        processes = self.processes))
        print 'Converting to Points'
        pool = Pool(self.processes)
        self.location_series = pool.map(
                        point_wrapper, 
                        self.location_series)


    def image_slicer(self, image):
        """
        
        Cuts the larger satellite image into smaller images 
        A less intense version of extract_patches_2d
        
        Input:
            (2d array) satellite image or population map

        """
        self.patches = []
        self.indices = []
        if len(image.shape) == 2:
            for y in range(0, self.nrows - self.obs_size, self.step):
                for x in range(0, self.ncols - self.obs_size, self.step):
                    mx = min(x+self.obs_size, self.ncols)
                    my = min(y+self.obs_size, self.nrows)
                    tile = image[ y: my, x: mx ]
                    if tile.shape == (self.obs_size, self.obs_size):
                        self.patches.append(tile)
                        self.indices.append((x, y))
        else:
            for y in range(0, self.nrows, self.step):
                for x in range(0, self.ncols, self.step):
                    mx = min(x+self.obs_size, self.ncols)
                    my = min(y+self.obs_size, self.nrows)
                    tile = image[ :, y : my, x: mx ]
                    if tile.shape == (self.obs_size, self.obs_size):
                        self.patches.append(tile)
                        self.indices.append((x, y))
        self.patches = np.array(self.patches)
        self.indices = np.array(self.indices)

    def sample_extractor(self, data_array, axis):
        """

        Extracts a sample of images
            (array) data_array, satellite images
            (axis) axis of array
            (array) image_sample, Keras ready numpy array of images
        
        """
        self.image_slicer(data_array)
        self.image_sample = np.take(
                self.patches, self.sample_idx, axis=axis,
                mode = 'raise')
 
    def import_sat_image(self):
        """
        Converts satellite images to a GeoDataFrame
        The satellite image used here is the 2010 LANDSAT 7 TOA composite
        Returns:
            (geodataframe) df_image
            (int) nrows, ncols. size of satellite image
        """
        data = []
        count = 0
        self.get_location()
        for extension in self.channels:
            self.filename = self.sat_folder_loc + self.state_name + \
                        '_' + self.year + '_' + extension + '.tif'
            # testing whether files exist
            if not os.path.exists( self.filename ):
                sys.exit('Error: File ' + self.filename + 
                        ' does not exist')
            # getting data
            print 'Loading bandwidth', extension
            band = self.satellite_gdal.GetRasterBand(1)
            band_array = band.ReadAsArray()
            #band_series = parmap.starmap(
            #        array_wrapper, 
            #        zip(self.cols_grid, self.rows_grid), 
            #        band_array, processes = self.processes)
            data.append(band_array.flatten())
        self.df_image = GeoDataFrame({'location': self.location_series})
        for count, extension in enumerate(self.channels):
            self.df_image[extension] = data[count]
 
    def import_census_data(self):
        """
        Gets population density from census data
        Inputs:
        census_folder_loc: location of data (string)    
        census_shapefile: name of shapefile (string)
        Returns: 
            df_census: GeoDataFrame with census information
        """ 
        ## Importing shapefile 
        print 'Importing census data'
        self.df_census = GeoDataFrame.from_file(
                self.census_folder_loc + self.census_shapefile) 
        print 'Census data loaded'
        # It turns out the earth isn't flat 
        # Getting area in km**2 
        print 'Calculating area'
        area_sq_degrees = self.df_census['geometry'] 
        area_sq_km = [] 
        for region in area_sq_degrees: 
            geom_area = ops.transform(
                    partial(
                        pyproj.transform, 
                        pyproj.Proj(init='EPSG:4326'), 
                        pyproj.Proj(
                            proj='aea', 
                            lat1=region.bounds[1], 
                            lat2=region.bounds[3]
                            )
                        ), 
                    region) 
            area = geom_area.area / 1000000.0  #convert to km2
            area_sq_km.append( area )
        self.df_census['area'] = area_sq_km
        self.df_census['density'] = \
                self.df_census['POP10'] / self.df_census['area']
                #self.df_census['POP10'] / self.df_census['area']
        print 'Area calculated'

    def join_sat_census(self):
        """
        
        Combines satellite and urban database construction
        
        """
        print 'Joining satellite and census data'
        self.census_blocks = np.array(self.df_census['geometry'])
        self.census_pop = np.array(self.df_census['density'])
        self.idx = self.spatialIndex(self.census_blocks)
        pixel_point = self.point_within_polygon(
                self.idx, self.location_series, self.census_blocks,
                self.census_pop)
        pixel_point.columns = ['poly', 'pop', 'latitude', 'longitude']
        self.df_image['pop_density'] = pixel_point['pop']
        self.df_image['latitude_u'] = pixel_point['latitude']
        self.df_image['longitude_u'] = pixel_point['longitude']
    
    def sampling(self):
        """
        Constructs a weighted sample of images from the GeoDataFrame
        
        Returns:
            (array) sample_idx: index of sampled images 
        
        Note: Keras uses the last x percent of data in cross validation 
            Have to shuffle here to ensure that the last ten percent isn't just
            the southern most rows of information
        """
        # Getting the sum of urban pixels for each patch
        self.pop_array = self.df_image['pop_density'].fillna(0)
        self.pop_array = np.array(
                self.pop_array).reshape((self.nrows, self.ncols))
        print 'extract patches'
        self.image_slicer(self.pop_array)
        print 'get locations for individual frames'
        pool = Pool(self.processes)
        cols_grid = pool.map(adder, self.indices[:,0])
        rows_grid = pool.map(adder, self.indices[:,1])
        self.frame_location_series = parmap.starmap(
                pixel_to_coordinates,
                zip(cols_grid, rows_grid), self.geotransform, 
                processes=self.processes)
        print 'converting locations to Points'
        self.frame_location_series = \
                pool.map(Point, self.frame_location_series)
        pop_count = np.array([np.mean(patch) for patch in self.patches])
        self.df_sample = pd.DataFrame(pop_count, columns=['pop_ave'])
        # Getting the locations
        self.df_sample['location'] = self.frame_location_series
        # Creating sample weights 
        seed = 1996
        self.pop_mean_sample = self.df_sample.sample(
                frac=self.sample_rate,
                replace=True,
                weights='pop_ave',
                random_state = seed)
        self.sample_idx = np.array(self.pop_mean_sample.index.values)
    
    def sample_generator_sat(self):
        """
        
        Constructs a sample of observations that Keras can play with 
        
        """
        # satellite image data
        image_array = []
        for channel in self.channels:
            tmp_img = np.array(
                    self.df_image[channel]).reshape((self.nrows, self.ncols))
            self.sample_extractor(tmp_img[:,:], axis=0)
            image_array.append(np.array(self.image_sample))
        image_array = np.array(image_array)    
        print image_array.shape
        image_array = np.swapaxes(image_array, 0, 1)
        self.image_output_data = np.array(image_array)
    
    def sample_generator_pop(self):
        """
        
        Constructs a sample of observations that Keras can play with 
        We take the mean population density of each image 
        
        """
        # getting population data
        pop_output_data = []
        self.sample_extractor(self.pop_array, axis=0)
        for i in range(0, len(self.sample_idx)):
            # We take the mean pop density
            obs_pop = np.mean(self.image_sample[i])
            pop_output_data.append(obs_pop)
        self.pop_output_data = np.nan_to_num(np.array(pop_output_data))

    def save_files_X(self):
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
            if self.file_size!=(self.image_output_data.shape[0]-1):
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
            temp = self.pop_output_data[0:self.file_size]
            f = h5py.File(
                    self.save_folder_loc + 'db_' + self.state_name + \
                            '_y_%d.hdf5'% count, 'w')
            f.create_dataset('data', data = temp, compression="gzip")
            f.close()
            if self.file_size!=(self.pop_output_data.shape[0]-1):
                self.pop_output_data = self.pop_output_data[self.file_size:]
            count += 1

