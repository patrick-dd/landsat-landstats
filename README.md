# LANDSAT-landstats

You can use the files in this repository to predict population from
LANDSAT7 images.

At it's core, this is a supervised learning model to predict
socio-economic characteristics from satellite data. You can easily
modify this code to predict other socio-economic characteristics or use
other satellite images.

I ran this on [AWS](aws.amazon.com)'s G8 machine. I tried this on a
mid-2009 MacBookPro and can recommend you don't.

The files of interest are:
 - `cnn_core.py`: runs the convolutional neural network
 - `database_constructor.py`: merges satellite images with population
   databases (shapefiles)
 - `spatial.ipynb`: a worked through example of database construction
 - `ee.ipynb`: downloads LANDSAT 7 TOA images

This analysis relies on some packages 
- [Keras](keras.io)
- [Google Earth Engine](https://developers.google.com/earth-engine/)
- [Shapely](toblerity.org/shapely/manual.html)
- [GeoPandas](geopandas.org/user.html)


