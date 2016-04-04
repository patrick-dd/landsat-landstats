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
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, SGD

obs_size = 32 

print('Reading data')

## Initialising with value 0
f = h5py.File('keras_data/db_Oregon_X_0.hdf5', 'r')
X_train = np.array(f['data'])
f.close()
f = h5py.File('keras_data/db_Oregon_y_0.hdf5', 'r')
y_train = np.array(f['data'])
f.close()

#for i in range(1,78):
#	f = h5py.File('keras_data/db_Oregon_X_%d.hdf5' % i, 'r')
#	X_train = np.vstack((X_train, np.array(f['data'])))
#	f.close()
#	f = h5py.File('keras_data/db_Oregon_y_%d.hdf5' % i, 'r')
#	y_train = np.hstack((y_train, f['data']))
#	f.close()

## Initialising with value 0
#f = h5py.File('keras_data/db_Washington_X_0.hdf5', 'r')
#X_test = np.array(f['data'])
#f.close()
#f = h5py.File('keras_data/db_Washington_y_0.hdf5', 'r')
#y_test = np.array(f['data'])
#f.close()

#for i in range(1,60):
#	print i
#	f = h5py.File('keras_data/db_Washington_X_%d.hdf5' % i, 'r')
#	X_test = np.vstack((X_test, np.array(f['data'])))
#	f.close()
#	f = h5py.File('keras_data/db_Washington_y_%d.hdf5' % i, 'r')
#	y_test = np.hstack((y_test, f['data']))
#	f.close()

# mean normalisation
mean_value = np.mean(X_train)
std_value = np.std(X_train)

X_train = X_train - mean_value
#X_test = X_test - mean_value

X_train = X_train / std_value
#X_test = X_test / std_value

print mean_value, std_value

# normalize target values

# <log normalised>
y_train = np.log(y_train + 1)
y_mean = np.mean(y_train)
y_std = np.std(y_train)
y_train -= y_mean
y_train /= y_std
# </log normalised>


print 'Creating the model'
model = Sequential()
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
#model.add(Dropout(0.25))

model.add(Convolution2D(128, 3, 3,
                        border_mode='same',
                        init='he_uniform'))
model.add(Activation('relu'))
model.add(Convolution2D(128, 3, 3, 
			border_mode='same',
                        init='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
#model.add(Dropout(0.25))

model.add(Convolution2D(256, 3, 3,
                        border_mode='same',
                        init='he_uniform'))
model.add(Activation('relu'))
model.add(Convolution2D(256, 3, 3, 
			border_mode='same',
                        init='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
#model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(4096, init='he_uniform'))
model.add(Activation('relu'))
model.add(Dense(4096, init='he_uniform'))
model.add(Activation('relu'))

#model.add(Dropout(0.5))
model.add(Dense(1, init='glorot_uniform'))
model.add(Activation('linear'))

# setting sgd optimizer parameters
adam = Adam(lr = 0.01, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8)
#sgd = SGD(lr = 0.1, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(loss='mean_squared_error', optimizer='adam')

earlystop = callbacks.EarlyStopping(monitor='val_loss', patience = 5, 
	verbose=1, mode='min')
checkpoint = callbacks.ModelCheckpoint('/tmp/weights.hdf5', 
	monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
history = callbacks.History()

print("Starting training")
model.fit(X_train, y_train, batch_size=32, validation_split=0.15, nb_epoch=25,
        show_accuracy=False, callbacks = [earlystop, checkpoint, history])

# save as JSON
json_string = model.to_json()
# save model weights
model.save_weights('model_weights.hdf5', overwrite=True)

print("Evaluating")
predicted = np.array(model.predict(X_test)).flatten()
print 'Predictions on training output'

slope, intercept, r_val, p_val, std_err = stats.linregress(predicted, y_train)
print 'R squared:', r_val**2
print 'Intercept:', intercept
print 'Slope:', slope
print 'P value:', p_val

fig, ax = plt.subplots()
ax.scatter(y_test, predicted, marker = 'o') 
ax.set_xlabel('Log normalised population density', fontsize=20)
ax.set_ylim(0, max(predicted) )
ax.set_xlim(0, max(y_test))
ax.set_ylabel('Model prediction', fontsize=20)
x = np.linspace(0, max(y_test))
ax.plot(x, x, color='y', linewidth=3)
plt.savefig('scatter.png')
plt.cla()
plt.clf()

#pickle.dump( predicted, open( "predicted_normalised.p", "wb" ) )
#pickle.dump( y_test, open( "y_normalised.p", "wb" ) )
#pickle.dump( max_train, open( "max_train.p", "wb" ) )

#y_unnorm = un_normalise(y_test, max_train)
#pred_unnorm = un_normalise(predicted, max_train)
#pickle.dump( pred_unnorm, open( "predicted_unnormalised.p", "wb" ) )
#pickle.dump( y_unnorm, open( "y_unnormalised.p", "wb" ) )

fix, ax = plt.subplots()
ax.plot(history.history['loss'], label = 'Training loss')
ax.plot(history.history['val_loss'], label = 'Validation loss')
ax.set_xlabel('Epoch', fontsize=20)
ax.set_ylabel('RMSE (people per km$^2$)', fontsize=20)
plt.legend()
plt.savefig('loss.png')

print 'Printing History'
print history.history


