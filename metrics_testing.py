import numpy as np


def find_accuracy(y_pred, y_actual, classification):
    #take last vector of y_pred
    y_hat_data = y_pred[31]
    sum = 0
    for i in range(0,len(y_pred)):
        y_hat = y_hat_data[i]
        y = y_actual[i]
        if(classification):
            if(round(y_hat) == round(y)):
                sum+= 1
        else:
            val = 1 - (np.abs(y_actual - y_pred).sum() / len(y_actual))
            return val
    return sum / len(y_hat_data)


def find_total_correct_labels(y_pred, y_actual):
    sum = 0
    for i in range(0, len(y_actual)):
        y_hat = y_pred[i]
        y = y_actual[i]
        if(round(y) == round(y_hat)):
            sum += 1
    return sum / len(y_actual)


def find_accuracy(y_pred, y_actual, classification):
    if y_pred.shape[0] > 1:
        y_pred = y_pred[0]
    if classification:
        total_correct_labels = find_total_correct_labels(y_pred, y_actual)
        return total_correct_labels / len(y_actual)
    else:
        val = 1 - (np.abs(y_actual - y_pred).sum() / len(y_actual))
        return val

def recall(y_pred, y_actual, classification):
    pass
