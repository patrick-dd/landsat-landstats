# LANDSAT-landstats

This is currently under re-development

---

You can use the files in this repository to predict population from
LANDSAT7 images.

LANDSAT lanstats is a supervised learning model to predict
socio-economic characteristics from satellite data. You can easily
modify this code to predict other socio-economic characteristics or use
other satellite images.

I ran this on [AWS](http://aws.amazon.com)'s G2 machine. [Get in
touch](http://twitter.com/patrickdoupe) if you'd like a link to the AMI. 
I also tried this on a mid-2009 MacBookPro and can recommend you don't.

The files of interest are:
 - `cnn.py`: runs the convolutional neural network
 - `database_constructor.py`: merges satellite images with population
   databases (shapefiles)
 - `db_test.py`: a testing file for `database_constructor.py`
    
In addition to the usual, the analysis relies on these packages 

- [Keras](http://www.keras.io)
- [Google Earth Engine](https://developers.google.com/earth-engine/)
- [Shapely](http://www.toblerity.org/shapely/manual.html)
- [GeoPandas](http://www.geopandas.org/user.html)
- [Rtree](http://toblerity.org/rtree)

You can download satellite image data through the [Google Earth Engine](https://developers.google.com/earth-engine). You can download Census, Urban and County data via the Census' [website](http://www.census.gov/geo/maps-data/data/tiger-data).


