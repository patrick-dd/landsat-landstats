import numpy as np
import h5py
import cPickle
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import callbacks
from keras.utils.visualize_util import plot
from keras.layers import advanced_activations

obs_size = 64

print('Reading data')

## Initialising with value 0
f = h5py.File('keras_data/db_Oregon_X_0.hdf5', 'r')
X_train = np.array(f['data'])
f.close()
f = h5py.File('keras_data/db_Oregon_y_0.hdf5', 'r')
y_train = np.array(f['data'])
f.close()

for i in range(1,35):
	f = h5py.File('keras_data/db_Oregon_X_%d.hdf5' % i, 'r')
	X_train = np.vstack((X_train, np.array(f['data'])))
	f.close()
	f = h5py.File('keras_data/db_Oregon_y_%d.hdf5' % i, 'r')
	y_train = np.hstack((y_train, f['data']))
	f.close()

## Initialising with value 0
f = h5py.File('keras_data/db_Washington_X_0.hdf5', 'r')
X_test = np.array(f['data'])
f.close()
f = h5py.File('keras_data/db_Washington_y_0.hdf5', 'r')
y_test = np.array(f['data'])
f.close()

for i in range(1,26):
	f = h5py.File('keras_data/db_Washington_X_%d.hdf5' % i, 'r')
	X_test = np.vstack((X_test, np.array(f['data'])))
	f.close()
	f = h5py.File('keras_data/db_Washington_y_%d.hdf5' % i, 'r')
	y_test = np.hstack((y_test, f['data']))
	f.close()

# mean normalisation
mean_value = np.mean(X_train)
X_train = X_train - mean_value
X_test = X_test - mean_value

std_value = np.std(X_train)
X_train = X_train / std_value
X_test = X_test / std_value

# normalize target values

# <weighted_log>
y_train = np.log10(y_train + 1)
y_test = np.log10(y_test + 1)
# </weighted_log>

# <weighted>
max_train = y_train.max()
y_train /= max_train
y_test /= max_train
# </weighted>

# training weights
inv_weight, bin_val = np.histogram(y_train)
clamp_idx = len(inv_weight) - 1
weight_idx = [min(np.searchsorted(bin_val, v, side="left"), clamp_idx) for v in y_train]
sample_weights = 1.0 / inv_weight[weight_idx]

def create_model():
	model = Sequential()
	# first convolutional pair
	model.add(Convolution2D(32, 5, 5, 
            border_mode='valid',
            input_shape = (7, obs_size, obs_size)))
	model.add(Activation('relu'))
	model.add(Convolution2D(32, 5, 5))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	# second convolutional pair
	model.add(Convolution2D(64, 3, 3, border_mode='valid'))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 3))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	# third convolutional pair
	model.add(Convolution2D(128, 3, 3, border_mode='valid'))
	model.add(Activation('relu'))
	model.add(Convolution2D(128, 3, 3))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	# convert convolutional filters to flat so they
	# can be fed to fully connected layers
	model.add(Flatten())
	# first fully connected layer
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	# classification fully connected layer
	model.add(Dense(1))
	model.add(Activation('linear'))
	#
	return model


# setting sgd optimizer parameters
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer='rmsprop')

model_loaded = create_model()
model_loaded.load_weights('/model_weights.hdf5')

y_pred = model.predict(X_nn)

def un_normalise_y(y_pred, max_train):
	"""
	Takes the normalised y and unnormalises import
	"""
	## times by max_train
	y_pred *= max_train
	## raise to power of ten
	y_pred = (y_pred+1)**10 - 1
	return y_pred


y_pred = un_normalise_y(y_pred, max_train)

f = file('cnn_predictions.save', 'wb')
cPickle.dump(y_pred, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()





