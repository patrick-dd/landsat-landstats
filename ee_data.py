"""
 getting satellite data from the google earth engine

 for victoria, we have 2006 census data
 for oregon and washington, we have 2010 census data

 you'll have to run this (and modify) for each state

"""


# importing packages
%matplotlib inline
import matplotlib.pyplot as plt
import ee
import urllib
import zipfile
import os
import numpy as np

# initializing google earth engine
ee.Initialize()

## coarse polygons of the states

# oregon, usa
or_geo = ee.Geometry.Polygon(
        [[[-124.7607421875, 46.4378568950242],
          [-124.892578125, 41.672911819602085],
          [-116.8505859375, 41.934976500546604],
          [-116.15707055111261, 43.93564450044653],
          [-116.0595703125, 46.195042108660154]]])
# washington, usa
wa_geo = ee.Geometry.Polygon(
        [[[-125.22381921671274, 49.091740384374475],
          [-124.61124844157064, 45.37216842328447],
          [-116.323170256953, 45.6794180409067],
          [-116.52003026237128, 47.323556727851155],
          [-116.36673611145437, 49.086778738102986],
          [-120.22607749606635, 49.4346509607392]]])
# victoria, australia
vic_geo = ee.Geometry.Polygon(
        [[[140.985901, -34.016243],
          [140.963928, -38.048092],
          [141.546204, -38.393337],
          [141.601135, -38.246807],
          [142.205383, -38.341656],
          [143.556702, -38.83115],
          [144.459229, -38.246807],
          [144.887695, -38.496593],
          [145.535889, -38.625454],
          [146.413147, -39.070377],
          [146.876221, -38.616871],
          [147.797424, -37.900864],
          [149.500305, -37.762032],
          [149.950745, -37.501011],
          [148.236877, -36.791691],
          [148.105042, -36.791691],
          [148.181946, -36.597889],
          [147.963043, -36.044659],
          [147.810059, -35.995785],
          [147.699371, -35.924644],
          [147.336823, -36.040215],
          [147.018219, -36.102375],
          [146.545807, -35.982452],
          [146.320587, -36.040215],
          [145.947052, -35.995785],
          [145.705353, -35.964668],
          [145.518585, -35.808903],
          [145.337311, -35.862343],
          [145.151367, -35.817814],
          [144.974762, -35.862343],
          [144.930817, -35.986897],
          [144.969269, -36.075741],
          [144.755035, -36.120129],
          [144.271637, -35.742054],
          [143.694855, -35.375614],
          [143.392731, -35.160336],
          [143.337799, -34.786739],
          [142.772003, -34.587997],
          [142.623688, -34.759666],
          [142.502838, -34.728069],
          [142.387482, -34.347973],
          [142.079865, -34.093609],
          [141.893097, -34.116352],
          [141.68985, -34.107258],
          [141.530548, -34.198174],
          [141.195465, -34.075413]]]);

# setting up file naming
# change state_name for file naming
# change shape for clipping
# change year for date of satellite image

channels = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6_VCID_2', 'B7']
state_name = 'Victoria'
shape = vic_geo
year = '2006'
scale = 180             # m / pixel

if not os.path.exists('LANDSAT_TOA/'+state_name+'/'+str(scale)):
    os.makedirs('LANDSAT_TOA/'+state_name+'/'+str(scale))

## Getting satellite data for Victoria
# Load LANDSAT TOA image for year
image = ee.Image('LANDSAT/LE7_TOA_1YEAR/'+year) 

# clipped image
clipped_image = image.clip(shape)

# export
path = clipped_image.getDownloadURL({'scale': scale, 'crs': 'EPSG:4326'})
spatial_data_file = state_name + year + '.zip'
a = urllib.urlretrieve(path, spatial_data_file)
zfile = zipfile.ZipFile(spatial_data_file)
zfile.extractall('LANDSAT_TOA/'+state_name+'/'+str(scale))


# Cleaning up the names of the files
channels = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6_VCID_2', 'B7']

for filename in os.listdir('LANDSAT_TOA/'+state_name+'/'+str(scale)):
    for channel in channels:
        try:
            prefix, band, extension = filename.partition(channel)
            if band==channel:
                os.rename(
                    'LANDSAT_TOA/'+state_name+'/'+str(scale)+'/'+filename,
                    'LANDSAT_TOA/'+state_name+'/'+str(scale)+'/'+state_name\
                    +'_'+year+'_'+str(scale)+'_'+band+extension)
        except:
            pass

# double checking
for filename in os.listdir('LANDSAT_TOA/'+state_name+'/'+str(scale)):
    print filename






