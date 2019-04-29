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




def find_accuracy(y_pred, y_actual):
    correct = 0
    for i in range(0, len(y_pred)):
        estimated_class_val = y_pred[i]
        class_val = y_actual[i]
        if float(estimated_class_val) == float(class_val):
            correct += 1
    return correct / len(y_pred)

#only use for binary classification problems
def find_precision(y_pred, y_actual):
    #make very small value to avoid divide by zero
    true_positive = 1E-15
    false_positive = 0
    for i in range(0, len(y_pred)):
        estimated_class_val = y_pred[i]
        class_val = y_actual[i]
        if float(estimated_class_val) == float(1):
            if(float(class_val) == float(1)):
                true_positive += 1
            else:
                false_positive += 1
    return true_positive / (true_positive + false_positive)


def find_precision_multiclass(y_pred, y_actual, class_values):
    sum = 0
    for class_val in class_values:
        sum += find_precision_multiclass_helper(y_pred, y_actual, class_val)
    numb_of_class_vals = len(class_values)
    return sum / numb_of_class_vals

def find_precision_multiclass_helper(y_pred, y_actual, correct_class_val):
    #make very small value to avoid divide by zero
    true_positive = 1E-15
    false_positive = 0
    for i in range(0, len(y_pred)):
        estimated_class_val = y_pred[i]
        class_val = y_actual[i]
        if float(estimated_class_val) == float(correct_class_val):
            if(float(class_val) == float(correct_class_val)):
                true_positive += 1
            else:
                false_positive += 1
    return true_positive / (true_positive + false_positive)

#only use for binary classification problems
def find_recall(y_pred, y_actual):
    true_positive = 1E-15
    false_negative = 0
    for i in range(0, len(y_pred)):
        estimated_class_val = y_pred[i]
        class_val = y_actual[i]
        if (float(class_val) == float(1)):
            if(float(estimated_class_val) == float(1)):
                true_positive += 1
            else:
                false_negative += 1
    return true_positive / (true_positive + false_negative)


def find_recall_multiclass(y_pred, y_actual, class_values):
    sum = 0
    for class_val in class_values:
        sum += find_recall_multiclass_helper(y_pred, y_actual, class_val)
    numb_of_classes = len(class_values)
    return sum / numb_of_classes

def find_recall_multiclass_helper(y_pred, y_actual, correct_class_val):
    true_positive = 1E-15
    false_negative = 0
    for i in range(0, len(y_pred)):
        estimated_class_val = y_pred[i]
        class_val = y_actual[i]
        if (float(class_val) == float(correct_class_val)):
            if(float(estimated_class_val) == float(correct_class_val)):
                true_positive += 1
            else:
                false_negative += 1
    return true_positive / (true_positive + false_negative)


