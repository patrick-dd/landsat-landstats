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

f = h5py.File('data/keras/db_Oregon_0.hdf5', 'r')
X_train = np.array(f['features'])
y_train = np.array(f['targets'])
f.close()

f = h5py.File('data/keras/db_Oregon_1.hdf5', 'r')
X_train = np.vstack((X_train, np.array(f['features'])))
y_train = np.vstack((y_train, np.array(f['targets'])))
f.close()

# There some observations that are ocean cells have
# have values of zero everywhere
# I remove them here
non_zeros = [ a.any() for a in X_train[:,1,:,:]]
X_train = X_train[np.where(non_zeros)]
y_train = y_train[np.where(non_zeros)]

# test only do positive
#pop_pos = [True if i > 0 else False for i in y_train]
#X_train = X_train[np.where(pop_pos)]
#y_train = y_train[np.where(pop_pos)]

# normalising the input data
X_train = X_train.astype('float32')
X_mean = np.mean(X_train, axis=0)
X_std = np.mean(X_train, axis=0)
X_train -= X_mean
X_train /= X_std

# log normalising the output data
y_train = y_train.astype('float32')
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
    # layer three
    model.add(Convolution2D(256, 3, 3,
                        border_mode='same',
                        init='he_uniform'))
    model.add(Activation('relu'))
    model.add(Convolution2D(256, 3, 3, 
    			border_mode='same',
                        init='he_uniform'))
    model.add(Activation('relu'))
    model.add(Convolution2D(256, 3, 3, 
    			border_mode='same',
                        init='he_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    # layer four
    model.add(Convolution2D(512, 3, 3,
                        border_mode='same',
                        init='he_uniform'))
    model.add(Activation('relu'))
    model.add(Convolution2D(512, 3, 3, 
    			border_mode='same',
                        init='he_uniform'))
    model.add(Activation('relu'))
    model.add(Convolution2D(512, 3, 3, 
    			border_mode='same',
                        init='he_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    # fuller connected layers
    model.add(Flatten())
    model.add(Dense(4096, init='he_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, init='he_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # regression layer
    model.add(Dense(1, init='glorot_uniform'))
    model.add(Activation('linear'))
    return model

def evaluate_model(model):
    # making space for some new data
    print 'Reading data'

    f = h5py.File('data/keras/db_Washington_0.hdf5', 'r')
    X_test = np.array(f['features'])
    y_test = np.array(f['targets'])
    f.close()

    f = h5py.File('data/keras/db_Washington_1.hdf5', 'r')
    X_test = np.vstack((X_test, np.array(f['features'])))
    y_test = np.vstack((y_test, np.array(f['targets'])))
    f.close()

    # There some observations that are ocean cells have
    # have values of zero everywhere
    # I remove them here
    non_zeros = [ a.any() for a in X_test[:,1,:,:]]
    X_test = X_test[np.where(non_zeros)]
    y_test = y_test[np.where(non_zeros)]
    # test only do positive
    #pop_pos = [True if i > 0 else False for i in y_test]
    #X_test = X_test[np.where(pop_pos)]
    #y_test = y_test[np.where(pop_pos)]

    # small sample for toying
    #X_test = X_test
    #y_test = y_test

    # normalising the input data
    X_test = X_test.astype('float32')
    X_test -= X_mean
    X_test /= X_std

    # log normalising the output data
    y_test = y_test.astype('float32')
    y_test = np.log(y_test + 1)
    y_test -= y_mean
    y_test /= y_std

    print ''
    print 'Evaluating the model in sample'
    predicted_in = np.array(model.predict(X_train)).flatten()
    print 'Predictions on training output'
    #
    slope, intercept, r_val, p_val, std_err = \
            stats.linregress(predicted_in, y_train.reshape(-1))
    print 'Regression of predictions on actual normalised population'
    print '---------------------------------------------------------'
    print 'R squared:   ', r_val**2
    print 'Intercept:   ', intercept
    print 'Slope:       ', slope
    print 'P value:     ', p_val
    #
    input_data = {
        'Train_predictions' : predicted_in,
        'y_train_normalised' : y_train}
    pickle.dump( input_data, open('insample_output_data.p', 'wb') )

    print ''
    print 'Evaluating the model out of sample'
    predicted = np.array(model.predict(X_test)).flatten()
    print 'Predictions on test output'
    #
    slope, intercept, r_val, p_val, std_err = \
            stats.linregress(predicted, y_test.reshape(-1))
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
        'in_predictions' : predicted_in,
        'y_train_normalised' : y_train,
        'y_mean' : y_mean,
        'y_std' : y_std,
        'X_mean' : X_mean,
        'X_std' : X_std}
    pickle.dump( output_data, open('output_data.p', 'wb') )

def train_loop(no_epochs, learning_rates, model_weights=None):
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
    model = create_model()
    if model_weights:
        model.load_weights(model_weights)
    ## compile model
    adam = Adam(lr = 1, 
            beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8)
    model.compile(loss='mean_squared_error', optimizer='adam')
    # setting callbacks
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
    #
    model_cb = [earlystop, checkpoint, history]
    
    model.fit(X_train, y_train, 
            batch_size=32, 
            validation_split=0.15, 
            sample_weight = sample_weights,
            nb_epoch=no_epochs, 
            show_accuracy=False, 
            callbacks =model_cb)
    
    for rate in learning_rates:
        print ''
        print 'Learning rate: ', rate
        print ''
        #
        adam.lr.set_value(rate)
        #
        print("Starting training")
        model.fit(X_train, y_train, 
            batch_size=32, 
            validation_split=0.15, 
            sample_weight = sample_weights,
            nb_epoch=no_epochs, 
            show_accuracy=False, 
            callbacks =model_cb)
    print 'Training complete, good job!'
    # save model weights
    model.save_weights(weights_path, overwrite=True)
    return model
 
learning_rates = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
no_epochs = 50
model = train_loop(no_epochs, learning_rates,model_weights='weights.hdf5')
evaluate_model(model)
print 'Good one. Next?'
