"""

Last step :)

Converting the interpolated county level estimates of population into 
a my-sql database.


"""


import numpy as np
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries, read_file
from shapely.geometry import Point, Polygon, MultiPolygon, shape
from shapely.prepared import prep
import shapely.ops as ops
from pandas import Series, DataFrame
import rasterio
from rtree import index
import shapefile
from itertools import *
from functools import partial
import pyproj
import pickle, cPickle
import pymysql
from sqlalchemy import create_engine

## Loading shapefile
db_county = GeoDataFrame.from_file('county_2010_census/County_2010Census_DP1.shp')
# Loading in the CNN model estimates
df_prediction = pickle.load(file('prediction_db.p', 'rb'))
df_prediction.columns = df_prediction.iloc[0]
df_prediction = df_prediction.reindex(df_prediction.index.drop(0))
df_prediction.shape

# Creating a spatial index
county_blocks = np.array(db_county['geometry'])
idx = index.Index()
count = 0
for c_block in county_blocks:
    idx.insert(count, c_block.bounds)
    count += 1

# assign each point to a county
pixels_location = np.array(df_prediction[['lon', 'lat']])
pixel_county = []
count = 0
for location in pixels_location:
    pixel = Point(location)
    temp_polygon = None
    for j in idx.intersection((pixel.x, pixel.y)):
        if pixel.within(county_blocks[j]):
            temp_polygon = county_blocks[j]
            break
    pixel_county.append([temp_polygon, location[0], location[1]])
    count += 1

# combine datasets. check that locations match in each row
pixel_county = np.array(pixel_county)
df_prediction['latitude'] = pixel_county[:,2]
df_prediction['longitude'] = pixel_county[:,1]
df_prediction['geometry'] = pixel_county[:,0] 
print 'Check that latitude and longitude merge correctly'
print (df_prediction['lat']==df_prediction['latitude']).sum() == df_prediction.shape[0]
print (df_prediction['lon']==df_prediction['longitude']).sum() == df_prediction.shape[0]
# delete unnecessary columns
df_prediction = df_prediction.drop(['lat', 'lon'], axis=1)

# collecting centroids for merge
def getXY(pt):
    return (pt.x, pt.y)

# centroids for county data
centroidseries = db_county['geometry'].centroid
x,y = [list(t) for t in zip(*map(getXY, centroidseries))]
db_county['centroid_latitude'] = x
db_county['centroid_longitude'] = y

# dropping empty geometries (e.g. oceans) and getting centroids
df_prediction = df_prediction.dropna(subset=['geometry'], how='all')
centroidseries = [df_prediction.iloc[i]['geometry'].centroid for i in range(
                    len(df_prediction))]
x,y = [list(t) for t in zip(*map(getXY, centroidseries))]
df_prediction['centroid_latitude'] = x
df_prediction['centroid_longitude'] = y

# merge
result = pd.merge(df_prediction, db_county, on=['centroid_latitude', 'centroid_longitude'])

variables = ['estimate', 'geometry_y','geometry_x', 'NAMELSAD10', 'DP0010001']
result = result[variables]
# check that merge was ok
print 'True if merge worked'
(result['geometry_y'] == result['geometry_x']).sum() == result.shape[0]
# cleaning df
result.drop(['geometry_y'], axis=1)
result.rename(columns={'geometry_x' : 'geometry'}, inplace=True)

# getting area in sqkm
area_sq_degrees = result['geometry']
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
    area = geom_area.area / 1e6 # convert to km2
    area_sq_km.append( area )

result['area'] = area_sq_km
result['census_density'] = result['DP0010001'] / result['area'] 

result['census_density'] = result['census_density'].astype(float)
result['estimate'] = result['estimate'].astype(float)
# getting county averages
estimate_density = DataFrame(result.groupby(['NAMELSAD10'])['estimate'].mean())
actual_density = DataFrame(result.groupby(['NAMELSAD10'])['census_density'].mean())
# making final database
output = pd.merge( estimate_density, actual_density, left_index=True, right_index=True)
output.reset_index(level=0, inplace=True)
# save results
engine = create_engine('mysql+pymysql://root:@localhost/website_db', echo=False)
output.to_sql('website_database', engine, chunksize=20000)
output.to_csv('website_database.csv')