import os
import tensorflow as tf
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sn
from tensorflow.keras.models import Sequential
from keras.layers import Dense,Flatten, Convolution2D, MaxPooling2D

def Build_Model():
    model = Sequential()
    model.add(Convolution2D(32, 4, activation='relu', input_shape=(28, 28, 3)))
    model.add(Convolution2D(32, 4, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(900, activation='relu'))
    model.add(Dense(24, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
  
def Dataset(train_path, test_path):
  train_imgs = []
  train_labels = []
  test_imgs = []
  test_labels = []
  
  for imgname in os.listdir(f'{train_path}/images'):
    img = cv.imread(os.path.join(f'{train_path}/images', imgname))
    train_imgs.append(img)

  for imglabel in os.listdir(f'{train_path}/labels'):
    f = open(os.path.join(f'{train_path}/labels', imglabel), 'r')
    line = f.readline()
    first_number = line.split()[0]
    train_labels.append(int(first_number))

  for imgname in os.listdir(f'{test_path}/images'):
    img = cv.imread(os.path.join(f'{test_path}/images', imgname))
    test_imgs.append(img)

  for imglabel in os.listdir(f'{test_path}/labels'):
    f = open(os.path.join(f'{test_path}/labels', imglabel), 'r')
    line = f.readline()
    first_number = line.split()[0]
    test_labels.append(int(first_number))

  test_images = np.array(test_imgs)
  test_labels = np.array(test_labels)
  train_images = np.array(train_imgs)
  train_labels = np.array(train_labels)

  test_images = test_images / 255.0
  train_images = train_images / 255.0
  return train_images, train_labels, test_images, test_labels

def Confusion_Matrix(model, test_images, test_labels):
  predictions = model.predict(test_images)
  y_predicted_labels = [np.argmax(i) for i in predictions]
  cm = tf.math.confusion_matrix(labels=test_labels, predictions=y_predicted_labels)
  plt.figure(figsize = (10,7))
  sn.heatmap(cm, annot=True, fmt='d')
  plt.xlabel('Predicted')
  plt.ylabel('Truth')