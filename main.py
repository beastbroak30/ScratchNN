import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from NN import *


data = pd.read_csv('dataset/train.csv').values

data = np.array(data)
m, n = data.shape
np.random.shuffle(data) 
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.


X_train[:,0].shape

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500, m, n)