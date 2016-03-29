"""
Creates the database using the file databaseConstructor.py
"""

## Creating a database!

from data_cleaning import *
## Note the file naming
## Number _53_ for Washington
## Number _41_ for Oregon
census_shapefile = 'tabblock2010_53_pophu.shp'
state_name = 'Washington'
state_code = 'WA'
year = '2010'

channels = ['B1', 'B2'] #, 'B3', 'B4'] #, 'B5', 'B6_VCID_2', 'B7']
sat_folder_loc = 'data/landsat/' + state_name + '/'
census_folder_loc = 'data/census/'
save_folder_loc = 'data/keras/'

# a file size of 10 ~ 1mb
file_size = 1024				# number of observations in each file
sample_rate = 0.2				# number of total images sampled
obs_size = 32					# size of image
processes = 40                  # number of CPU cores
step = 4                        # size of step in image creation

if __name__ == "__main__":
    print 'Starting database construction'
    db = database_constructor(census_folder_loc, census_shapefile,
            sat_folder_loc, save_folder_loc, state_name, 
            state_code, year, channels, file_size, sample_rate, obs_size,
            processes, step)
    db.import_sat_image()
    db.import_census_data()
    db.join_sat_census()
    db.sampling()
    db.sample_generator_sat()
    db.sample_generator_pop()
    db.save_files_X()
    db.save_files_y()
    print 'Database constructed'
    print 'Good job!'
