# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 22:12:50 2020

@author: Jamal Moussa
"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

def get_dataset(training=True):
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    
    if training is False:
        return (test_images, test_labels)
    else:
        return (train_images, train_labels)
    
def print_stats(images, labels):
    (total, m, n) = images.shape
    dim = str(m) + 'x' + str(n)
    
    counter = [0]*10
    
    for label in labels:
        if label == 1:
            counter[label] += 1
        if label == 2:
            counter[label] += 1
        if label == 3:
            counter[label] += 1
        if label == 4:
            counter[label] += 1
        if label == 5:
            counter[label] += 1
        if label == 6:
            counter[label] += 1
        if label == 7:
            counter[label] += 1
        if label == 8:
            counter[label] += 1
        if label == 9:
            counter[label] += 1
        if label == 0:
            counter[label] += 1
            
    print(total)
    print(dim)
    print('0. T-shirt/top - ' + str(counter[0]))
    print('1. Trouser - ' + str(counter[1]))
    print('2. Pullover - ' + str(counter[2]))
    print('3. Dress - ' + str(counter[3]))
    print('4. Coat - ' + str(counter[4]))
    print('5. Sandal - ' + str(counter[5]))
    print('6. Shirt - ' + str(counter[6]))
    print('7. Sneaker - ' + str(counter[7]))
    print('8. Bag - ' + str(counter[8]))
    print('9. Ankle boot - ' + str(counter[9]))
        
def view_image(image, label):
    plt.figure()
    plt.title(label)
    plt.imshow(image)
    plt.colorbar()
    plt.grid()
    plt.show()
    
def build_model():
    
    model = keras.Sequential()
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, input_shape=(28,28), activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.add(keras.layers.Softmax())
    
    sce = keras.losses.sparse_categorical_crossentropy

    model.compile(loss=sce,optimizer='adam', metrics=['accuracy'])
   
    return model

def train_model(model, images, labels, T):
    model.fit(images, labels, epochs=T)
    
def evaluate_model(model,images,labels,show_loss=True):
    test_loss, test_accuracy = model.evaluate(images,labels,verbose=0)
    test_accuracy = str(round(test_accuracy*100,2)) + '%'
    
    if show_loss is True:
        print("Loss: " + str(round(test_loss,2)))
        
    print("Accuracy: " + test_accuracy)
    
def predict_label(model,images,index):
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    pred = model.predict(images[index])
    
    return pred
    
    