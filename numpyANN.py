import numpy as np
import common
from random import shuffle
import metrics_testing as mt
# N is batch size; D_in is input dimension;


def split_into_x_y(data, class_index):
    x = []
    y = []
    for point in data:
        x_row = []
        y_row = []
        for j in range(len(point)):
            val = point[j]
            if j == class_index:
                y_row.append(val)
            else:
                x_row.append(val)
        x.append(x_row)
        y.append(y_row)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return x,y

def train_model(x, y):
    # Randomly initialize weights
    w1 = np.random.randn(D_in, H)
    w2 = np.random.randn(H, D_out)
    for t in range(500):
        # Forward pass: compute predicted y
        h = x.dot(w1)
        h_relu = np.maximum(h, 0)
        y_pred = h_relu.dot(w2)

        # Compute and print loss
        #loss = np.square(y_pred - y).sum()
        loss = 0
        for i in range(len(y)):
            y_i = y[i]
            y_pred_i = y_pred[i]
            if(np.argmax(y_pred_i) == 0 and y_i != 0):
                loss +=1
            elif(np.argmax(y_pred_i) == 1 and y_i != 1):
                loss +=1
            elif(np.argmax(y_pred_i) == 2 and y_i != 2):
                loss +=1

        print(t, loss)

        # Backprop to compute gradients of w1 and w2 with respect to loss
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.T.dot(grad_y_pred)
        grad_h_relu = grad_y_pred.dot(w2.T)
        grad_h = grad_h_relu.copy()
        grad_h[h < 0] = 0
        grad_w1 = x.T.dot(grad_h)

        # Update weights
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2
    return w1, w2

def predict_output(w1, w2, test_set, actual_output):
    # Forward pass: compute predicted y
    predicted_output = []
    for i in range(len(test_set)):
        point = test_set[i]
        h = point.dot(w1)
        h_relu = np.maximum(h, 0)
        y_pred = h_relu.dot(w2)
        y_pred = y_pred[0]
        max_index = np.argmax(y_pred)
        #print('predicted' + str(max_index))
        #print('actual' + str(actual_output[i]))
        predicted_output.append(max_index)
    return predicted_output


# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 9, 100, 2


learning_rate = 1e-6

data = common.read_csv("C:/Users/andre/PycharmProjects/MachineLearningFinal/Data/breastCancer.csv")


class_index = 9


#update class lables 2=>0 and 4=> 1
for point in data:
    if point[class_index] == '2':
        point[class_index] = '0'
    elif point[class_index] == '4':
        point[class_index] = '1'


#remove data points with missing attributes (since there are only 16 out of over 600 data points)
common.remove_points_with_missing_attributes(data)




shuffle(data)
print(len(data))
x,y = split_into_x_y(data, class_index)
w1, w2 = train_model(x, y)

output = predict_output(w1, w2, x, y)

accuracy = mt.find_accuracy(output, y)
print(accuracy)