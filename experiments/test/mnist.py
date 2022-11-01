import sys
sys.path.append(r'D:\projects\tensorflow_gpu\experiments')
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import metrics, preprocessing
import numpy as np
import pandas as pd
from utils import load_data, packetloss_threshold, energyconsumption_threshold
from network import create_model
from operator import mod
from torch import softmax


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
X_train, X_test = x_train/255.0, x_test/255.0
X_train = X_train.reshape((X_train.shape[0], 28*28)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28*28)).astype('float32')

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

num_labels = len(np.unique(y_train))
batch_size = 128
hidden_units = 256
dropout = 0.45


# Sequential type
model1 = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='sigmoid')
])

model1.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

model1.fit(X_train, y_train, epochs=5, batch_size=64, verbose=True)

model1.summary()


# functional type
image_size = x_train.shape[1]
input_size = image_size * image_size


inputs = tf.keras.Input(shape=(input_size,))
x = tf.keras.layers.Dense(64, activation=tf.nn.relu)(inputs)
outputs = tf.keras.layers.Dense(10, activation='sigmoid')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
optim = tf.keras.optimizers.Adam()

model.summary()
model.compile(optimizer=optim, loss=loss, metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=True)