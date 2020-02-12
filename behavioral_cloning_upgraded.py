#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 19:56:31 2019

@author: ekele
"""


! git clone https://github.com/rslim087a/track
! ls track
!pip3 install imgaug

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
from imgaug import augmenters as iaa



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

# Data augmentation with imgaug
def zoom(image):
    zoom = iaa.Affine(scale=(1, 1.3))
    image = zoom.augment_image(image)
    return image

def pan(image):
    pan = iaa.Affine(translate_percent= {'x': (-0.1, 0.1), 'y': (-0.1, 0.1)})
    image = pan.augment_image(image)
    return image

def img_random_brightness(image):
    brightness = iaa.Multiply((0.2, 1.2))
    image = brightness.augment_image(image)
    return image

def flip(image, steering_angle):
    image = cv2.flip(image, 1)
    steering_angle = -steering_angle
    return image, steering_angle

def random_augment(image, steering_angle):
    image = mpimg.imread(image)
    if np.random.rand() < 0.5:
        image = pan(image)
    if np.random.rand() < 0.5:
        image = zoom(image)
    if np.random.rand() < 0.5:
        image = img_random_brightness(image)
    if np.random.rand() < 0.5:
        image, steering = flip(image, steering_angle)
    return image, steering_angle


def img_processing(img):
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

def batch_generator(image_paths, steering_angle, batch_size, istraining):
    
    while True:
        batch_img = []
        batch_steering = []
        
        for i in range(batch_size):
            random_index = random.randint(0, len(image_paths) - 1)
            
            # differentiate between training and validation arguments
            if istraining:
                im, steering = random_augment(image_paths[random_index], steering_angle[random_index])
            else:
                im = mpimg.imread(image_paths[random_index])
                steering = steering_angle[random_index]
            im = img_processing(im)
            batch_img.append(im)
            batch_steering.append(steering)
        yield(np.asarray(batch_img), np.asarray(batch_steering))

print(X_train.shape)            
X_train_gen, y_train_gen = next(batch_generator(X_train, y_train, 1, 1))    
X_valid_gen, y_valid_gen = next(batch_generator(X_valid, y_valid, 1, 0))

fig, axes = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()

axes[0].imshow(X_train_gen[0])
axes[0].set_title('Training Image')
axes[1].imshow(X_valid_gen[0])
axes[1].set_title('Validation Image')
    
ncol = 2
nrow = 10

fig, axs = plt.subplots(nrow, ncol, figsize=(15, 50))
fig.tight_layout()

for i in range(10):
    randnum = random.randint(0, len(image_paths) - 1)
    random_image = image_paths[randnum]
    random_steering = steerings[randnum]
    
    original_image = mpimg.imread(random_image)
    augmented_image, steering = random_augment(random_image, random_steering)
    
    axs[i][0].imshow(augmented_image)
    axs[i][0].set_title('Original Image')
    
    axs[i][1].imshow(augmented_image)
    axs[i][1].set_title('Augmented Image')


image = image_paths[random.randint(0, 500)]
random_index = random.randint(0, 500)
steering_angle = steerings[random_index]

original = mpimg.imread(image)
panned_image = pan(original)
brightness_aug_img = img_random_brightness(original)
flipped_image, flipped_steering_angle = flip(original, steering_angle)

fig, axes = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
axes[0].imshow(original)
axes[0].set_title('Original Image')
axes[1].imshow(flipped_image)
axes[1].set_title('Processed Image')

# plot a random image
plt.imshow(X_train[random.randint(0, len(X_train) - 1)])
plt.axis('off')
# Dimensions of input tensor
print(X_train_shape.shape)

def nvidia_model():
    model = Sequential()
    model.add(Conv2D(24, (5, 5), strides=(2, 2), input_shape=(66, 200, 3)))
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
    
    
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation('elu'))
     
    model.add(Dense(50))
    model.add(Activation('elu'))
    
    model.add(Dense(10))
    model.add(Activation('elu'))  
    # Output is a steering angle so output layer has only one neuron
    model.add(Dense(1))
    
    model.compile(loss='mse', optimizer=Adam(lr = 0.0001))
    return model

model = nvidia_model()
print(model.summary())

history = model.fit_generator(batch_generator(X_train, y_train, batch_size=100, istraining=1), 
                              steps_per_epoch=300, 
                              epochs = 10, 
                              validation_data = batch_generator(X_valid, y_valid, batch_size=100, istraining=0), 
                              validation_steps = 200,
                              verbose = 1,
                              shuffle = 1)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')

model.save('TrainedModel/nvidia_model.h5')



















