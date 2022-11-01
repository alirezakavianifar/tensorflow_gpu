import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import metrics, preprocessing
import numpy as np
import pandas as pd
from utils import load_data, packetloss_threshold, energyconsumption_threshold
from network import create_model
from sklearn.model_selection import train_test_split, GridSearchCV
import sys
from unicodedata import name
sys.path.append(r'D:\projects\tensorflow_gpu\experiments')


path = r'D:\projects\papers\Deep Learning for Effective and Efficient  Reduction of Large Adaptation Spaces in Self-Adaptive Systems\experiment_results\simulation_results\DeltaIoTv1'

data = load_data(load_all=True)

X = data[0]
X = np.array(X)
y_packet = data[1][0]
y_latency = data[1][1]
y_packet = np.array(y_packet)
y_latency = np.array(y_latency)


def discretize_y(y):
    if y < 10:
        return 1
    else:
        return 0


discrete = np.vectorize(discretize_y)

y_packet = discrete(y_packet)
y_latency = discrete(y_latency)


X_train, X_test, y_train_p, y_test_p, y_train_l, y_test_l = train_test_split(
    X, y_packet, y_latency, test_size=0.30, random_state=42)


input_size = X.shape[1]
# input_size = image_size * image_size


inputs = tf.keras.Input(shape=(input_size,))
x = tf.keras.layers.Dense(128, activation=tf.nn.relu)(inputs)
output1 = tf.keras.layers.Dense(1, activation='sigmoid', name='packetloss')(x)
output2 = tf.keras.layers.Dense(1, activation='sigmoid', name='latency')(x)


model = tf.keras.Model(inputs=inputs, outputs=[output1, output2])
loss1 = tf.keras.losses.BinaryCrossentropy()
loss2 = tf.keras.losses.BinaryCrossentropy()
optim = tf.keras.optimizers.Adam()

losses = {'packetloss': loss1, 'latency': loss2}
y = {'packetloss': y_train_p, 'latency': y_train_l}
y_test = {'packetloss': y_test_p, 'latency': y_test_l}

model.summary()
model.compile(optimizer=optim, loss=losses, metrics=['accuracy'])


batch_size = [32, 64, 128, 256]
epochs = [10, 20, 30, 40, 50]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', n_jobs=-1, cv=3)
grid_result = grid.fit(X_train, y=y_train_p)



model.fit(X_train, y=y, epochs=15, batch_size=64, verbose=True)

model.evaluate(X_test, y=y_test)
