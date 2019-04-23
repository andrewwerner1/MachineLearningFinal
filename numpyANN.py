# -*- coding: utf-8 -*-
import numpy as np
import metrics_testing as mt
import math
import GetData as dataHandler

#Used this as a reference https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

#def load_data(filename):
#    dataset = np.loadtxt(filename, delimiter=",")
#    X = dataset[:, 1:56]
#    Y_data = dataset[:, 0]
#   #Create a container array around Y
#    Y = np.array(np.array(Y_data))
#    return X,Y


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def make_predictions(x_input, y_output, w1, w2, classification):
    # Forward pass: compute predicted y
    h = x_input.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)
    return y_pred
    #accuracy = 1 - ((np.square(y_pred - y_output).sum()) / len(y_pred))




def train_model(x_train, y_train, w1, w2, learning_rate, classification ):
    for t in range(200000):
        # Forward pass: compute predicted y
        h = x_train.dot(w1)
        h_relu = np.maximum(h, 0)
        y_pred = h_relu.dot(w2)
        #if(classification):
        #    y_pred = sigmoid(y_pred)

        # Compute and print loss
        loss = np.square(y_pred[0] - y_train).sum()
        #loss =
        print(t, loss)

        # Backprop to compute gradients of w1 and w2 with respect to loss
        grad_y_pred = 2.0 * (y_pred - y_train)
        grad_w2 = h_relu.T.dot(grad_y_pred)
        grad_h_relu = grad_y_pred.dot(w2.T)
        grad_h = grad_h_relu.copy()
        grad_h[h < 0] = 0
        grad_w1 = x_train.T.dot(grad_h)

        # Update weights
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2

    #return weights
    return w1, w2



# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 159, 4, 100, 159

# Create random input and output data
x = np.random.randn(N, D_in)
#print(x)
y = np.random.randn( D_out)
#print(y)

filename = "C:/Users/andre/PycharmProjects/MachineLearningFinal/Data/Iris.csv"
dataset = dataHandler.get_data_from_file(filename)
x_test, y_test = dataHandler.split_data_into_XY(dataset, class_index=0, first_attribute_index=1, last_attribute_index=56)

# Randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6

w1, w2 = train_model(x_test, y_test, w1, w2, learning_rate, classification=True)

y_pred = make_predictions(x, y_test, w1, w2, classification=True)
print('predictions')
print(y_pred)
print('actual output')
print(y_test)
accuracy = mt.find_accuracy(y_pred, y_test, classification=True)
print('accuracy')
print(accuracy)