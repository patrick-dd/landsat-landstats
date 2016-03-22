"""

A testing script for the database construction

"""
import unittest
from data_cleaning import *

## initialising with existing files
census_shapefile = 'tabblock2010_53_pophu.shp'
state_name = 'Washington'
state_code = 'WA'
year = '2010'

channels = ['B1', 'B2'] 
sat_folder_loc = 'LANDSAT_TOA_LGE/' + state_name + '/'
census_folder_loc = 'Washington_block_data/'
urban_folder_loc = 'urban_areas/'
save_folder_loc = 'keras_data/test/'

file_size = 1024				# number of observations in each file
sample_rate = 0.02				# number of total images sampled
obs_size = 32					# size of image
processes = 2                   # number of cpus

class test_for_db_construction(unittest.TestCase):
    " Test for databaseConstructor.py "
    def setUp(self):
        print ''
        print 'Setting up tests'
        print ''

    def tearDown(self):
        print ''
        print 'Finishing up'
        print ''

    def test_geotransform(self):
        print 'Testing whether geotransform works'
        test_instance = databaseConstructor(
                census_folder_loc, census_shapefile, urban_folder_loc,
		        sat_folder_loc, save_folder_loc, state_name, state_code,
                year, channels, file_size, sample_rate, obs_size, processes)
        test_geo = [ 1, 2, 3, 4, 5, 6 ]
        test_geo = test_instance.pixel_to_coordinates( 1, 2, test_geo)
        self.assertEqual( test_geo, (9, 21))

    def test_rtree_index(self):
        print 'Testing whether the spatial index works'
        test_instance = databaseConstructor(
                census_folder_loc, census_shapefile, urban_folder_loc,
		        sat_folder_loc, save_folder_loc, state_name, state_code,
                year, channels, file_size, sample_rate, obs_size, processes)
        test_rtree = [[(0.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
                    [(10.0, 10.0), (11.0, 11.0), (10.0, 11.0)]]
        test_rtree = [Polygon(t) for t in test_rtree]
        idx = test_instance.spatialIndex(test_rtree)
        self.assertEqual(
                list(idx.intersection((11.0, 11.0, 12.0, 12.0))),
                [1] )

    def test_pop_df(self):
        print 'Testing population dataframe creator'
        test_instance = databaseConstructor(
                census_folder_loc, census_shapefile, urban_folder_loc,
		        sat_folder_loc, save_folder_loc, state_name, state_code,
                year, channels, file_size, sample_rate, obs_size, processes)
        test_shapes = [[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
                    [[10.0, 10.0], [10.0, 11.0], [11.0, 11.0], [11.0, 10.0]]]
        test_shapes = [Polygon(t) for t in test_shapes]
        idx = test_instance.spatialIndex(test_shapes)
        test_points = [Point(0.5, 0.5), Point(10.5, 10.5)]
        test_pop = [100, 200]
        test_data = [[test_shapes[i], test_pop[i], test_points[i].x, 
                        test_points[i].y] for i in [0,1]]
        test_data = GeoDataFrame(test_data)
        test_instance_df = test_instance.point_within_polygon_pop(
                idx, test_points, test_shapes, test_pop)
        self.assertEqual( test_data.equals(test_instance_df), True )

    def test_urban_df(self):
        print 'Testing urban dataframe creator'
        test_instance = databaseConstructor(
                census_folder_loc, census_shapefile, urban_folder_loc,
		        urban_shapefile, sat_folder_loc, save_folder_loc, state_name, 
                state_code, year, channels, file_size, sample_rate, obs_size, 
                slice_depth, processes)
        test_shapes_n = [
                Polygon([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]),\
                None]
        test_shapes = [[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
                    [[20.0, 20.0], [20.0, 21.0], [21.0, 21.0], [21.0, 20.0]]]
        test_shapes = [Polygon(t) for t in test_shapes]
        idx = test_instance.spatialIndex(test_shapes)
        test_points = [Point(0.5, 0.5), Point(10.5, 10.5)]
        test_pop = [1, 0]
        test_data = [[test_shapes_n[i], test_pop[i], test_points[i].x, 
                        test_points[i].y] for i in [0,1]]
        test_data = GeoDataFrame(test_data)
        test_instance_df = test_instance.point_within_polygon_urban(
                idx, test_points, test_shapes)
        self.assertEqual( test_data.equals(test_instance_df), True )

    def test_adder(self):
        print 'Testing adder'
        test_instance = databaseConstructor(
                census_folder_loc, census_shapefile, urban_folder_loc,
		        urban_shapefile, sat_folder_loc, save_folder_loc, state_name, 
                state_code, year, channels, file_size, sample_rate, obs_size, 
                slice_depth, processes)
        output = test_instance.adder( 10 )
        self.assertEqual( output, 26 )

    def test_slicer(self):
        print 'Testing the image slicer '
        test_instance = databaseConstructor(
                census_folder_loc, census_shapefile, urban_folder_loc,
		        urban_shapefile, sat_folder_loc, save_folder_loc, state_name, 
                state_code, year, channels, file_size, sample_rate, obs_size, 
                slice_depth, processes)
 
    def test_extractor(self):
        print 'Testing the patch extractor '
        test_instance = databaseConstructor(
                census_folder_loc, census_shapefile, urban_folder_loc,
		        urban_shapefile, sat_folder_loc, save_folder_loc, state_name, 
                state_code, year, channels, file_size, sample_rate, obs_size, 
                slice_depth, processes)
 
    def test_sample_generator_sat(self):
        print 'Testing the sample generator '
        test_instance = databaseConstructor(
                census_folder_loc, census_shapefile, urban_folder_loc,
		        urban_shapefile, sat_folder_loc, save_folder_loc, state_name, 
                state_code, year, channels, file_size, sample_rate, obs_size, 
                slice_depth, processes)
        
    def test_sample_generator_pop(self):
        print 'Testing the sample generator for population'
        test_instance = databaseConstructor(
                census_folder_loc, census_shapefile, urban_folder_loc,
		        urban_shapefile, sat_folder_loc, save_folder_loc, state_name, 
                state_code, year, channels, file_size, sample_rate, obs_size, 
                slice_depth, processes)
 
if __name__ == "__main__":
    unittest.main()
