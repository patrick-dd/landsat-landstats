import matplotlib
matplotlib.use('Agg')
import numpy as np
import h5py
import pickle
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import callbacks
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

print 'Creating the model'

# sequential wrapper model
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


# load the weights 
# note: when there is a complete match between your model definition
# and your weight savefile, you can simply call model.load_weights(filename)
model.set_weights('model_weights.hdf5')
print('Model loaded.')



# setting sgd optimizer parameters
model.compile(loss='mean_squared_error', optimizer='adam', lr = 1e-3)

earlystop = callbacks.EarlyStopping(monitor='val_loss', patience = 3, 
    verbose=1, mode='min')
checkpoint = callbacks.ModelCheckpoint('/tmp/weights.hdf5', 
    monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
history = callbacks.History()

print("Starting training")
model.fit(X_train, y_train, batch_size=128, validation_split=0.15, sample_weight=sample_weights,
        show_accuracy=False, callbacks = [earlystop, checkpoint, history])
print("Evaluating")
score = model.evaluate(X_test, y_test, batch_size=128)
predicted = model.predict(X_test)  

fig, ax = plt.subplots()
ax.scatter(y_test, predicted) 
ax.set_xlabel('Log normalised population density', fontsize=20)
ax.set_ylim(0, max(predicted) )
ax.set_xlim(0, max(y_test))
ax.set_ylabel('Model prediction', fontsize=20)
x = np.linspace(0, max(y_test))
ax.plot(x, x, color='y', linewidth=3)
plt.savefig('scatter.png')
plt.cla()
plt.clf()

pickle.dump( predicted, open( "predicted.p", "wb" ) )
pickle.dump( y_test, open( "y_test.p", "wb" ) )

fix, ax = plt.subplots()
ax.plot(history.history['loss'], label = 'Training loss')
ax.plot(history.history['val_loss'], label = 'Validation loss')
ax.set_xlabel('Epoch', fontsize=20)
ax.set_ylabel('RMSE (people per km$^2$)', fontsize=20)
plt.legend()
plt.savefig('loss.png')

# plot(model, to_file='model_architecture.png')

print 'Printing History'
print history.history

# save as JSON
json_string = model.to_json()
# save model weights
model.save_weights('model_weights.hdf5', overwrite=True)
