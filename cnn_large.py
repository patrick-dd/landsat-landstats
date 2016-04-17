import matplotlib
matplotlib.use('Agg')
import os
os.environ['THEANO_FLAGS'] = 'device=gpu0,floatX=float32,lib.cnmem=0.85'
import numpy as np
from scipy import stats
import h5py
import pickle
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.utils.io_utils import HDF5Matrix
from keras import callbacks
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, SGD
import theano

obs_size = 32 

print 'Reading data'

f = h5py.File('data/keras/oregon/db_Oregon_X_0.hdf5', 'r')
X_train = np.array(f['data'])
f.close()
f = h5py.File('data/keras/oregon/db_Oregon_y_0.hdf5', 'r')
y_train = np.array(f['data'])
f.close()

file_numbers = np.random.randint(100,700,200)

for i in file_numbers:
	f = h5py.File('data/keras/oregon/db_Oregon_X_%d.hdf5' % i, 'r')
	X_train = np.vstack((X_train, np.array(f['data'])))
	f.close()
	f = h5py.File('data/keras/oregon/db_Oregon_y_%d.hdf5' % i, 'r')
	y_train = np.hstack((y_train, f['data']))
	f.close()

# There some observations that are ocean cells have
# have values of zero everywhere
# I remove them here
non_zeros = [ a.any() for a in X_train[:,1,:,:]]
X_train = X_train[non_zeros, :, :, :]
y_train = y_train[non_zeros]
X_train = X_train.astype('float32')
y_train = y_train.astype('float32')
# normalising the input data
X_train_mean = np.mean(X_train, axis=0)
X_train_std = np.mean(X_train, axis=0)
X_train -= X_train_mean
X_train /= X_train_std

# log normalising the output data
y_train = np.log(y_train + 1)
y_mean = np.mean(y_train, axis=0)
y_std = np.std(y_train, axis=0)
y_train -= y_mean
y_train /= y_std

# creating_weights for outcome var
inv_weight, bin_val = np.histogram(y_train)
clamp_idx = len(inv_weight) - 1
weight_idx = [min(np.searchsorted(bin_val, v, side='left'), clamp_idx) \
        for v in y_train]
sample_weights = 1.0 / inv_weight[weight_idx]

def create_model():
    print 'Creating the model'
    model = Sequential()
    # layer one
    model.add(Convolution2D(64, 3, 3, 
			border_mode='same',
			input_shape = (7, obs_size, obs_size),
                        init='he_uniform'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, 
			border_mode='same',
                        init='he_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(Dropout(0.25))
    # layer two
    model.add(Convolution2D(128, 3, 3,
                        border_mode='same',
                        init='he_uniform'))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, 3, 
			border_mode='same',
                        init='he_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(Dropout(0.25))
    # layer three
    model.add(Convolution2D(256, 3, 3,
                        border_mode='same',
                        init='he_uniform'))
    model.add(Activation('relu'))
    model.add(Convolution2D(256, 3, 3, 
			border_mode='same',
                        init='he_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(Dropout(0.25))
    # fuller connected layers
    model.add(Flatten())
    model.add(Dense(4096, init='he_uniform'))
    model.add(Activation('relu'))
    model.add(Dense(4096, init='he_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # regression layer
    model.add(Dense(1, init='glorot_uniform'))
    model.add(Activation('linear'))
    return model

def train_model(model, 
        no_epochs = 10, weights_path=None, learning_rate = 1e-2):
    print 'Constructing model'
    if model == None:
        model = create_model()
    if weights_path:
        model.load_weights(weights_path)
    adam = Adam(lr = learning_rate, 
            beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8)
    model.compile(loss='mean_squared_error', optimizer='adam')
    #
    checkpoint = callbacks.ModelCheckpoint(
            '/tmp/weights.hdf5', 
            monitor='val_loss', 
            verbose=1, 
            save_best_only=True, 
            mode='auto')
    earlystop = callbacks.EarlyStopping(
            monitor='val_loss', 
            patience = 5, 
            verbose=1, 
            mode='min')
    history = callbacks.History()
    print("Starting training")
    model.fit(X_train, y_train, 
            batch_size=32, 
            validation_split=0.15, 
            sample_weight = sample_weights,
            nb_epoch=no_epochs, 
            show_accuracy=False, 
            callbacks = [earlystop, checkpoint, history])
    # save model weights
    if weights_path:
        model.save_weights(weights_path, overwrite=True)
    else:
        model.save_weights('model_weights.hdf5', overwrite=True)
    return model, history

def evaluate_model(model):
    # making space for some new data
    #del X_train
    #del y_train
    print 'Collecting new data'
    #f = h5py.File('data/keras/oregon/db_Washington_X_0.hdf5', 'r')
    #X_test = np.array(f['data'])
    #f.close()
    #f = h5py.File('data/keras/oregon/db_Washington_y_0.hdf5', 'r')
    #y_test = np.array(f['data'])
    #f.close()

    #for i in range(1,60):
    #	print i
    #	f = h5py.File('data/keras/oregon/db_Washington_X_%d.hdf5' % i, 'r')
    #	X_test = np.vstack((X_test, np.array(f['data'])))
    #	f.close()
    #	f = h5py.File('data/keras/oregon/db_Washington_y_%d.hdf5' % i, 'r')
    #	y_test = np.hstack((y_test, f['data']))
    #	f.close()

    print 'Normalising data'
    #zeros = [not a.any() for a in X_test[:,1,:,:]]
    #X_test = X_test[np.where(zeros==False)]
    #y_test = y_test[np.where(zeros==False)]
    #X_test -= X_train_mean
    #X_test /= X_train_std
    #y_test -= y_mean
    #y_test /= y_std
    print 'Evaluating the model'
    predicted = np.array(model.predict(X_train)).flatten()
    print 'Predictions on training output'
    #
    slope, intercept, r_val, p_val, std_err = \
            stats.linregress(predicted, y_train)
    print 'Regression of predictions on actual normalised population'
    print '---------------------------------------------------------'
    print 'R squared:   ', r_val**2
    print 'Intercept:   ', intercept
    print 'Slope:       ', slope
    print 'P value:     ', p_val
    #
    output_data = {
        'Test_predictions' : predicted,
        'y_test_normalised' : y_test,
        'y_mean' : y_mean,
        'y_std' : y_std,
        'X_mean' : X_mean,
        'X_std' : X_std}
    pickle.dump( output_data, open('output_data.p', 'wb') )

def train_loop(model, no_epochs, learning_rates, model_weights):
    """
    
    A loop to gradually lower the learning rate and improve the model

    Inputs:
        no_epochs: int, the maximum number of epochs per loop
        model_weights: str, location of model weights file
        learning_rate: list of floats, learning rates to iterate through

    Returns:
        Saved pickle files with
            Validation loss history time series
            Predictions of output and actual population for plotting
            Model weights

    """
    val_loss_history = []
    for rate in learning_rates:
        if model == None:
            model, hist_rate = train_model(
                    model = None,
                    no_epochs = no_epochs,
                    weights_path = model_weights, 
                    learning_rate = rate)
        else:
            model, hist_rate = train_model(
                    model = model,
                    no_epochs = no_epochs,
                    weights_path = model_weights,
                    learning_rate = rate)
        val_loss_history.append(rate)
    pickle.dump( val_loss_history, open('loss_history.p', 'w') )
    evaluate_model(model)
    print 'Training complete, good job!'

 
learning_rates = [1, 1e-1, 1e-2, 1e-3, 1e-4]
no_epochs = 50
model, _ = train_model(None, no_epochs = no_epochs, learning_rate = 1)
train_loop(model, no_epochs, learning_rates, 'model_weights.hdf5')
