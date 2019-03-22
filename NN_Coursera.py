# Libraries
from os import getcwd, chdir
import keras
import numpy as np

## Setting WD
current_dir = getcwd()
chdir(current_dir)

## Basic model (1 layer, 1 input)
model = keras.Sequential([keras.layers.Dense(units = 1, input_shape = [1])])

## Optimizer and Loss function
model.compile(optimizer = "sgd", loss = "mean_squared_error")

## Importing data
xs = np.array([-1, 0, 1, 2, 3, 4], dtype = float)
ys = np.array([-3, -1, 1, 3, 5, 7], dtype = float)

## Running model
model.fit(xs, ys, epochs = 5)

## Looking at prediction
print(model.predict([10]))
