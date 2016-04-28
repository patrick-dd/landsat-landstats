# LANDSAT-landstats 1.1

You can use the files in this repository to predict population from
satellite images.

LANDSAT lanstats is a supervised learning model to predict
socio-economic characteristics from satellite data. You can easily
modify this code to predict other socio-economic characteristics or use
other satellite images. In addition to the convnet, I've provide files to
construct the data.

The model works pretty well. The current version results in a test R<sup>2</sup> of 0.45. Version 1.0 resulted in a test set R<sup>2</sup> of 0.26. The main difference is the improvement in satellite image resolution from 1000 meters per pixel to 180 meters per pixel. 

I ran the convnet on [AWS](http://aws.amazon.com)'s G2 machine and constructed
the database on AWS's M10 machine. [Get in
touch](http://twitter.com/patrickdoupe) if you'd like a link to the AMI. 
I also tried this on a mid-2009 MacBookPro and can recommend you don't.

The files of interest are:
 - `cnn.py`: runs the convolutional neural network
 - `data_cleaning.py`: merges satellite images with population
   databases (shapefiles)
 - `do.py`: a testing file for `data_cleaning.py`
 - `ee_data.py`: downloads the satellite images   

I downloaded population data from the [US census bureau](https://www.census.gov/geo/maps-data/data/tiger-line.html) and the [Australian
Bureau of Statistics](http://www.abs.gov.au/AUSSTATS/abs@.nsf/DetailsPage/1209.0.55.0022006?OpenDocument). Provided you have .tif satellite images and shapefiles for your socio-economic data, it should be easy to tweak the files. I am happy to share any constructed data and the model weights.

In addition to the usual, the analysis relies on these packages 

- [Keras](http://www.keras.io)
- [Google Earth Engine](https://developers.google.com/earth-engine/)
- [Shapely](http://www.toblerity.org/shapely/manual.html)
- [GeoPandas](http://www.geopandas.org/user.html)
- [Rtree](http://toblerity.org/rtree)
- [GDAL](https://pypi.python.org/pypi/GDAL/)
- [pyproj](https://github.com/jswhit/pyproj)
- [parmap](https://parmap.readthedocs.org)
- [h5py](http://www.h5py.org)




