# -*- coding: utf-8 -*-
"""
Apr 30, 2020
CS 540
P10: CNNs in Keras

@author: Jamal Moussa
"""
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

def get_dataset(training=True):
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    
    train_images = np.asarray(train_images).reshape(60000, 28, 28, 1)
    test_images = np.asarray(test_images).reshape(10000, 28, 28, 1)
    
    if training is False:
        return (test_images, test_labels)
    else:
        return (train_images, train_labels)
    
def build_model():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(64, (3,3), input_shape=(28,28, 1), activation='relu'))
    model.add(keras.layers.Conv2D(32, (3,3), activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    
    return model

def train_model(model, train_images, train_labels, test_images, test_labels, T):
    train_labels = keras.utils.to_categorical(train_labels)
    test_labels = keras.utils.to_categorical(test_labels)
    
    model.fit(train_images, train_labels, epochs=T, validation_data=(test_images, test_labels))
    
def predict_label(model, images, index):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    pred = model.predict(images)
    
    pred = np.argmax(pred)
    return pred