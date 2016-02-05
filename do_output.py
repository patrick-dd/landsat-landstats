## Creating a database!

from databaseConstructor import *

## Number 53 for Washington
## Number 41 for Oregon
state_name = 'Oregon'
state_code = 'OR'
year = '2010'

channels = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6_VCID_2', 'B7']
sat_folder_loc = 'LANDSAT_TOA/' + state_name + '/'
census_folder_loc = state_name + '_block_data/'
save_folder_loc = 'keras_data/output/'

# a file size of 10 ~ 1mb
file_size = 2048
obs_size = 64

databaseConstruction(sat_folder_loc, save_folder_loc, state_name, state_code, year, channels, file_size, 
		obs_size)

