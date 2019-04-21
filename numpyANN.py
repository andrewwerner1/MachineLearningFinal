# -*- coding: utf-8 -*-
import numpy as np

#Used this as a reference https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

def load_data(filename):
    dataset = np.loadtxt(filename, delimiter=",")
    X = dataset[:, 1:56]
    Y_data = dataset[:, 0]
    #Create a container array around Y
    Y = np.array(np.array(Y_data))
    return X,Y


def find_accuracy(y_pred, y_output_data, classification):
    #take last vector of y_pred
    y_hat_data = y_pred[31]
    sum = 0
    for i in range(0,len(y_pred)):
        y_hat = y_hat_data[i]
        y = y_output_data[i]
        if(classification):
            if(round(y_hat) == round(y)):
                sum+= 1
        else:
            val = 1 - (np.abs(y_output_data - y_pred).sum() / len(y_output_data))
            return val
    return sum / len(y_hat_data)


def make_predictions(x_input, y_output, w1, w2, classification):
    # Forward pass: compute predicted y
    h = x_input.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)
    #print predictions, actual outputs, and accuracy rate
    print('predictions')
    print(y_pred)
    print('actual outputs')
    print(y_output)
    print ('computed accuracy')
    #accuracy = 1 - ((np.square(y_pred - y_output).sum()) / len(y_pred))
    accuracy = find_accuracy(y_pred, y_output, classification)
    print(accuracy)



def train_model(x_train, y_train, w1, w2, learning_rate ):
    for t in range(100000):
        # Forward pass: compute predicted y
        h = x_train.dot(w1)
        h_relu = np.maximum(h, 0)
        y_pred = h_relu.dot(w2)

        # Compute and print loss
        loss = np.square(y_pred - y_train).sum()
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
N, D_in, H, D_out = 32, 55, 100, 32

# Create random input and output data
x = np.random.randn(N, D_in)
#print(x)
y = np.random.randn( D_out)
#print(y)

filename = "C:/Users/andre/PycharmProjects/AndyANN/Data/lungcancer.csv"
x, y = load_data(filename)

# Randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6

w1, w2 = train_model(x, y, w1, w2, learning_rate)

make_predictions(x, y, w1, w2, classification=True)