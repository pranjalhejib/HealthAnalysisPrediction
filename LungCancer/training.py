import os
import random
from tkinter import Y
import numpy
from glob import glob
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
from keras.utils import load_img, img_to_array

width = 100
height = 100
##############


def load_images(base_path):

    images = []
    path = os.path.join(base_path, '*.png')
    for image_path in glob(path):
        img = load_img(image_path,
                       target_size=(width, height))
        x = img_to_array(img)
        ret, thresh1 = cv2.threshold(x, 120, 255, cv2.THRESH_BINARY)
        images.append(thresh1)
    return images
###############


def load_images_random(base_path):

    images = []
    path = os.path.join(base_path, '*.png')
    for image_path in glob(path):
        image = preprocessing.image.load_img(image_path,
                                             target_size=(width, height))
        x = preprocessing.image.img_to_array(image)
        images.append(x)
    return images


def trainingProcess(cancer, healthy):
    A = load_images(cancer)
    B = load_images(healthy)
    ###################
    A = np.array(A)
    B = np.array(B)
    print(A.shape)
    print(B.shape)
    X = np.concatenate((A, B), axis=0)
    ##############
    X = X / 255.
    ###################
    YA = [0 for item in enumerate(A)]
    YB = [1 for item in enumerate(B)]

    y = np.concatenate((YA, YB), axis=0)
    y = tf.keras.utils.to_categorical(y, num_classes=2)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return X_train, X_test, Y_train, Y_test


def buildNetwork():
    conv_1 = 16
    conv_1_drop = 0.2
    conv_2 = 32
    conv_2_drop = 0.2
    dense_1_n = 1024
    dense_1_drop = 0.2
    dense_2_n = 512
    dense_2_drop = 0.2
    lr = 0.001

    epochs = 75
    batch_size = 32
    color_channels = 3

    model = Sequential()

    model.add(Convolution2D(conv_1, (5, 5),
                            input_shape=(width, height, color_channels),
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conv_1_drop))

    model.add(Convolution2D(conv_2, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conv_2_drop))

    model.add(Flatten())

    model.add(Dense(dense_1_n, activation='relu'))
    model.add(Dropout(dense_1_drop))

    model.add(Dense(dense_2_n, activation='relu'))
    model.add(Dropout(dense_2_drop))

    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=lr),
                  metrics=['accuracy'])

    return model


def buildVgg():
    model = VGG16(weights='imagenet')

    return model
#######################
