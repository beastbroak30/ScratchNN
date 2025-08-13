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

# --- Model save/load utilities ---
def save_model(W1, b1, W2, b2, filename_prefix):
    np.savetxt(filename_prefix + '_W1.txt', W1)
    np.savetxt(filename_prefix + '_b1.txt', b1)
    np.savetxt(filename_prefix + '_W2.txt', W2)
    np.savetxt(filename_prefix + '_b2.txt', b2)

def load_model(filename_prefix):
    W1 = np.loadtxt(filename_prefix + '_W1.txt').reshape(10, 784)
    b1 = np.loadtxt(filename_prefix + '_b1.txt').reshape(10, 1)
    W2 = np.loadtxt(filename_prefix + '_W2.txt').reshape(10, 10)
    b2 = np.loadtxt(filename_prefix + '_b2.txt').reshape(10, 1)
    return W1, b1, W2, b2

# Example usage:
#W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.15, 400, m, n)
#save_model(W1, b1, W2, b2, 'model')
W1, b1, W2, b2 = load_model('model')

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
    
test_prediction(6, W1, b1, W2, b2)