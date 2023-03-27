import os
import random
from glob import glob
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras import preprocessing

from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

# set size for images
width  = 100
height = 100

# function for loading images from path
def load_images(base_path):
    images = []
    path = os.path.join(base_path, '*.png')
    for image_path in glob(path):
        image = preprocessing.image.load_img(image_path,
                                             target_size=(width, height))
        x = preprocessing.image.img_to_array(image)

        images.append(x)
    return images

# loaded images 
A = load_images('./data/training_set/cancer')
B = load_images('./data/training_set/healthy')

# convert images into arrays
A = np.array(A)
B = np.array(B)

# print array shapes
print(A.shape)
print(B.shape)

# joining of the arrays by columns
X = np.concatenate((A,B), axis=0)

# divide array by 255
X = X / 255.0

# enumerating arrays
YA = [0 for item in enumerate(A)]
YB = [1 for item in enumerate(B)]

# final joining of arrays
y = np.concatenate((YA,YB), axis=0)


y = tf.keras.utils.to_categorical(y, num_classes=2) # 0 for cancer and 1 for healthy

print(y.shape)

