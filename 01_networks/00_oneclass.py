import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model

# the readings are huge, maybe just load a subportion of it
readings = pd.DataFrame(pd.read_pickle('/home/dlrc1/measurements/camera/00_firstcameraself.pkl'))
print(readings.shape)
print(readings.columns)
print(readings.realsense_rgbdata[0].shape)
print(readings.shape)

plt.imshow(readings.realsense_rgbdata[99])
plt.imshow(readings.realsense_depthdata[99])
plt.colorbar()

np.random.seed(1)   # initialize rng to keep things reproducible
training_frac = 0.9 # fraction of data to use for training
test_frac = 1-training_frac

index = np.arange(len(readings.realsense_depthdata))
np.random.shuffle(index)

# cutidx is the index at which to split the data into train/test sets
cutidx = (np.floor(training_frac*len(readings.realsense_depthdata))).astype(int)
x_train = readings.realsense_depthdata[:cutidx]
x_test = readings.realsense_depthdata[cutidx:]

# flatten the data
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# print(x_train.shape)
# print(x_test.shape)

# for images alone, the input has to be of the shape:
# num_images x height x width x layers
input_shape = (len(x_train),
               x_train[0].shape[0],
               x_train[0].shape[1],
               1)
print(f"Input shape: {input_shape}")

input_img = Input(shape=input_shape)
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)


autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


autoencoder.fit(x_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=x_test)