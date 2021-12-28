import keras,os
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


class RCNNModel(Sequential):

    def __init__(self, shape=(244, 244, 244), filters=64, kernel_size=(3, 3), padding="same", activation="relu"):
        super().__init__()
        self.add(Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, activation=activation))
        self.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        self.add(Conv2D(filters=filters * 2, kernel_size=kernel_size, padding=padding, activation=activation))
        self.add(Conv2D(filters=filters * 2, kernel_size=kernel_size, padding=padding, activation=activation))
        self.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        self.add(Conv2D(filters=filters * 4, kernel_size=kernel_size, padding=padding, activation=activation))
        self.add(Conv2D(filters=filters * 4, kernel_size=kernel_size, padding=padding, activation=activation))
        self.add(Conv2D(filters=filters * 4, kernel_size=kernel_size, padding=padding, activation=activation))
        self.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        self.add(Conv2D(filters=filters * 8, kernel_size=kernel_size, padding=padding, activation=activation))
        self.add(Conv2D(filters=filters * 8, kernel_size=kernel_size, padding=padding, activation=activation))
        self.add(Conv2D(filters=filters * 8, kernel_size=kernel_size, padding=padding, activation=activation))
        self.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        self.add(Conv2D(filters=filters * 8, kernel_size=kernel_size, padding=padding, activation=activation))
        self.add(Conv2D(filters=filters * 8, kernel_size=kernel_size, padding=padding, activation=activation))
        self.add(Conv2D(filters=filters * 8, kernel_size=kernel_size, padding=padding, activation=activation))
        self.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
