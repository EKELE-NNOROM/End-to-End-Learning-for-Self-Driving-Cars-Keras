#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 16:05:21 2019

@author: ekele
"""

! git clone https://github.com/rslim087a/track
! ls track

import os
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D, Activation
import cv2
import pandas as pd
import random
import ntpath
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg

datadir = 'track'
columns = 'center left right steering throttle reverse speed'.split()
data = pd.read_csv(os.path.join(datadir, 'driving_log.csv'), names = columns)
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)
data.head()

def path_leaf(path):
    top, bottom = ntpath.split(path)
    return ntpath.basename(bottom)

data['center'] = data['center'].apply(path_leaf)
data['left'] = data['left'].apply(path_leaf)
data['right'] = data['right'].apply(path_leaf)
data.head()

# for every set of images plot the steering angle in a histogram
num_bins = 25
hist, bins = np.histogram(data['steering'], num_bins)

# centering the values
center = (bins[:-1] + bins[1:]) * 0.5
samples_per_bin = 200
data['steering'].plot.hist(center, width=0.05)
plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))
print(bins.shape)
print(data['steering'].shape)

# Balancing data 
# Flatten our distribution
# Cut off extraneous samples for specific bins whose frequency exceed 200
print('total data: ', len(data))
samples_to_remove = []

for j in range(num_bins):
    list_ = []
    for i in range(len(data['steering'])):
        if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j+1]:
            list_.append(i)
    list_ = shuffle(list_)
    # cutting values above samples_per_bin of 200
    list_ = list_[samples_per_bin:]
    samples_to_remove.extend(list_)
print('removed', len(samples_to_remove))

data.drop(data.index[samples_to_remove], inplace=True)
print('remaining', len(data))

hist, bins = np.histogram(data['steering'], num_bins)
plt.bar(center, hist, width=0.5)
plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))
        
print(data.iloc[1])    
def load_img_steering(datadir, df):
    image_path = []
    steering = []
    for i in range(len(data)):
        indexed_data = data.iloc[i]
        center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
        image_path.append(os.path.join(datadir, center.strip()))
        steering.append(indexed_data[3])
    image_paths = np.asarray(image_path)
    steerings = np.asarray(steering)
    return image_paths, steerings

# Our image paths correspond to the features while the steerings correspond to our labels
image_paths, steerings = load_img_steering(datadir + '/IMG', data)

# Split into train and test data
X_train, X_valid, y_train, y_valid = train_test_split(image_paths, steerings, test_size=0.2, random_state=6)
print('Training samples {}\nTest/Validation samples {}'.format(len(X_train), len(X_valid)))
print(X_train.shape)

fig, axes = plt.subplots(1, 2, figsize=(12, 8))
axes[0].hist(y_train, bins=num_bins, width=0.05, color='blue')
axes[0].set_title('Training set')
axes[1].hist(y_valid, bins=num_bins, width=0.05, color='orange')
axes[1].set_title('Validation set')

# Preprocessing
def img_processing(img):
    img = mpimg.imread(img)
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # this matches the size used by the Nvidia Model architecture
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img

image = image_paths[200]
original = mpimg.imread(image)
preprocessed_image = img_processing(image)

fig, axes = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
axes[0].imshow(original)
axes[0].set_title('Original Image')
axes[1].imshow(preprocessed_image)
axes[1].set_title('Processed Image')

X_train = np.asarray(list(map(img_processing, X_train)))
X_valid = np.asarray(list(map(img_processing, X_valid)))

# plot a random image
plt.imshow(X_train[random.randint(0, len(X_train) - 1)])
plt.axis('off')
print(X_train.shape)

def nvidia_model():
    model = Sequential()
    model.add(Conv2D(24, (5, 5), strides=(2, 2), input_shape=X_train.shape[1:]))
    model.add(Activation('elu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2)))
    model.add(Activation('elu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2)))
    model.add(Activation('elu'))
    # Remove subsampling as the size of image has reduced significantly
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('elu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(50))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(10))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    # Output is a steering angle so output layer has only one neuron
    model.add(Dense(1))
    
    model.compile(loss='mse', optimizer=Adam(lr = 0.001))
    return model

model = nvidia_model()
print(model.summary())

history = model.fit(X_train, y_train, epochs = 50, validation_data = (X_valid, y_valid), batch_size = 100, verbose = 1, shuffle = 1)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')

model.save('TrainedModel/nvidia_model.h5')

from google.colab import files
files.download('nvidia_model.h5')


















