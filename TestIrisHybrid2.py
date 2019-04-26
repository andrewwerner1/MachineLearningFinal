import common
import copy
import math
import backpropogation as b
import random
from random import shuffle
import metrics_testing as mt
import GetData as data_handler
import KNearestNeighbor as Knn
import numpy as np


#URL https://archive.ics.uci.edu/ml/datasets/Iris


data = common.read_csv('C:/Users/andre/PycharmProjects/MachineLearningFinal/Data/iris.csv')

class_index = 4
first_attribute_index = 0
last_attribute_index = 3

#update class lables
for point in data:
    if point[class_index] == 'Iris-setosa':
        point[class_index] = '0'
    elif point[class_index] == 'Iris-versicolor':
        point[class_index] = '1'
    elif point[class_index] == 'Iris-virginica':
        point[class_index] = '2'


#remove data points with missing attributes (since there are only 16 out of over 600 data points)
common.remove_points_with_missing_attributes(data)

shuffle(data)

def split_data_in_ten_parts(data,  class_index):
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []
    list6 = []
    list7 = []
    list8 = []
    list9 = []
    list10 = []
    listClass0 = []
    listClass1 = []
    listClass2 = []
    data_copy = copy.deepcopy(data)

    for point in data_copy:
        class_val = point[class_index]
        if(float(class_val) == float('0')):
            listClass0.append(point)
        elif(float(class_val) == float('1')):
            listClass1.append(point)
        else:
            listClass2.append(point)
    for i in range(0, len(listClass0)):
        point = listClass0[i]
        if((i % 10) == 0):
            list1.append(point)
        elif((i % 10) == 1):
            list2.append(point)
        elif((i % 10) == 2):
            list3.append(point)
        elif((i % 10) == 3):
            list4.append(point)
        elif((i % 10) == 4):
            list5.append(point)
        elif((i % 10) == 5):
            list6.append(point)
        elif((i % 10) == 6):
            list7.append(point)
        elif((i % 10) == 7):
            list8.append(point)
        elif((i % 10) == 8):
            list9.append(point)
        elif((i % 10) == 9):
            list10.append(point)
    for i in range(0, len(listClass1)):
        point = listClass1[i]
        if((i % 10) == 0):
            list1.append(point)
        elif((i % 10) == 1):
            list2.append(point)
        elif((i % 10) == 2):
            list3.append(point)
        elif((i % 10) == 3):
            list4.append(point)
        elif((i % 10) == 4):
            list5.append(point)
        elif((i % 10) == 5):
            list6.append(point)
        elif((i % 10) == 6):
            list7.append(point)
        elif((i % 10) == 7):
            list8.append(point)
        elif((i % 10) == 8):
            list9.append(point)
        elif((i % 10) == 9):
            list10.append(point)
    for i in range(0, len(listClass2)):
        point = listClass2[i]
        if((i % 10) == 0):
            list1.append(point)
        elif((i % 10) == 1):
            list2.append(point)
        elif((i % 10) == 2):
            list3.append(point)
        elif((i % 10) == 3):
            list4.append(point)
        elif((i % 10) == 4):
            list5.append(point)
        elif((i % 10) == 5):
            list6.append(point)
        elif((i % 10) == 6):
            list7.append(point)
        elif((i % 10) == 7):
            list8.append(point)
        elif((i % 10) == 8):
            list9.append(point)
        elif((i % 10) == 9):
            list10.append(point)
    return list1, list2, list3, list4, list5, list6, list7, list8, list9, list10


set1, set2, set3, set4, set5, set6, set7, set8, set9, set10 = split_data_in_ten_parts(data,  class_index)
shuffle(set1)
shuffle(set2)
shuffle(set3)
shuffle(set4)
shuffle(set5)
shuffle(set6)
shuffle(set7)
shuffle(set8)
shuffle(set9)
shuffle(set10)


#define tunable parameters
numb_hidden_nodes = 5
numb_iterations = 50
numb_outputs = 3
learning_rate = 0.1
k=3

print('Test 1')
training_set_1 = set1 + set2 + set3 + set4 + set5
training_set_2 = set6 + set7 + set8 + set9
test_set = set10
#convert sets to numpy arrays
training_set_1 = np.asarray(training_set_1, dtype=float)
training_set_2 = np.asarray(training_set_2, dtype=float)
x_train_1, y_train_1 = data_handler.split_data_into_XY(training_set_1, class_index, first_attribute_index, last_attribute_index )
x_train_2, y_train_2 = data_handler.split_data_into_XY(training_set_2, class_index, first_attribute_index, last_attribute_index)
avg_features, updated_output = Knn.K_Nearest_Neighbor_with_feature_averaging(x_train_1, x_train_2, y_train_1, y_train_2, k, True)
training_set = data_handler.concatenate_feature_vals_with_labels(avg_features, updated_output)
v, w = b.find_model_1_hidden_layer(training_set, class_index, numb_hidden_nodes, numb_iterations, numb_outputs, learning_rate)
print('v weights found: ' + str(v) )
print('w weights found: ' + str(w))
estimated_output = b.get_estimated_output(test_set, class_index, numb_hidden_nodes, numb_outputs, v, w)
actual_output = data_handler.get_class_labels(test_set, class_index)
accuracy = mt.find_accuracy(estimated_output, actual_output)
error = b.test_model(test_set, class_index, numb_hidden_nodes, numb_outputs, v, w)
print('measured accuracy: ' + str(accuracy))
print('error: ' + str(error))
print('\n')

print('Test 2')
training_set_1 = set1 + set2 + set3 + set4 + set5
training_set_2 = set6 + set7 + set8 + set10
test_set = set9
#convert sets to numpy arrays
training_set_1 = np.asarray(training_set_1, dtype=float)
training_set_2 = np.asarray(training_set_2, dtype=float)
x_train_1, y_train_1 = data_handler.split_data_into_XY(training_set_1, class_index, first_attribute_index, last_attribute_index )
x_train_2, y_train_2 = data_handler.split_data_into_XY(training_set_2, class_index, first_attribute_index, last_attribute_index)
avg_features, updated_output = Knn.K_Nearest_Neighbor_with_feature_averaging(x_train_1, x_train_2, y_train_1, y_train_2, k, True)
training_set = data_handler.concatenate_feature_vals_with_labels(avg_features, updated_output)
v, w = b.find_model_1_hidden_layer(training_set, class_index, numb_hidden_nodes, numb_iterations, numb_outputs, learning_rate)
print('v weights found: ' + str(v) )
print('w weights found: ' + str(w))
estimated_output = b.get_estimated_output(test_set, class_index, numb_hidden_nodes, numb_outputs, v, w)
actual_output = data_handler.get_class_labels(test_set, class_index)
accuracy = mt.find_accuracy(estimated_output, actual_output)
error = b.test_model(test_set, class_index, numb_hidden_nodes, numb_outputs, v, w)
print('measured accuracy: ' + str(accuracy))
print('error: ' + str(error))
print('\n')

print('Test 3')
training_set_1 = set1 + set2 + set3 + set4 + set5
training_set_2 = set6 + set7 + set9 + set10
test_set = set8
#convert sets to numpy arrays
training_set_1 = np.asarray(training_set_1, dtype=float)
training_set_2 = np.asarray(training_set_2, dtype=float)
x_train_1, y_train_1 = data_handler.split_data_into_XY(training_set_1, class_index, first_attribute_index, last_attribute_index )
x_train_2, y_train_2 = data_handler.split_data_into_XY(training_set_2, class_index, first_attribute_index, last_attribute_index)
avg_features, updated_output = Knn.K_Nearest_Neighbor_with_feature_averaging(x_train_1, x_train_2, y_train_1, y_train_2, k, True)
training_set = data_handler.concatenate_feature_vals_with_labels(avg_features, updated_output)
v, w = b.find_model_1_hidden_layer(training_set, class_index, numb_hidden_nodes, numb_iterations, numb_outputs, learning_rate)
print('v weights found: ' + str(v) )
print('w weights found: ' + str(w))
estimated_output = b.get_estimated_output(test_set, class_index, numb_hidden_nodes, numb_outputs, v, w)
actual_output = data_handler.get_class_labels(test_set, class_index)
accuracy = mt.find_accuracy(estimated_output, actual_output)
error = b.test_model(test_set, class_index, numb_hidden_nodes, numb_outputs, v, w)
print('measured accuracy: ' + str(accuracy))
print('error: ' + str(error))
print('\n')

print('Test 4')
training_set_1 = set1 + set2 + set3 + set4 + set5
training_set_2 = set6 + set8 + set9 + set10
test_set = set7
#convert sets to numpy arrays
training_set_1 = np.asarray(training_set_1, dtype=float)
training_set_2 = np.asarray(training_set_2, dtype=float)
x_train_1, y_train_1 = data_handler.split_data_into_XY(training_set_1, class_index, first_attribute_index, last_attribute_index )
x_train_2, y_train_2 = data_handler.split_data_into_XY(training_set_2, class_index, first_attribute_index, last_attribute_index)
avg_features, updated_output = Knn.K_Nearest_Neighbor_with_feature_averaging(x_train_1, x_train_2, y_train_1, y_train_2, k, True)
training_set = data_handler.concatenate_feature_vals_with_labels(avg_features, updated_output)
v, w = b.find_model_1_hidden_layer(training_set, class_index, numb_hidden_nodes, numb_iterations, numb_outputs, learning_rate)
print('v weights found: ' + str(v) )
print('w weights found: ' + str(w))
estimated_output = b.get_estimated_output(test_set, class_index, numb_hidden_nodes, numb_outputs, v, w)
actual_output = data_handler.get_class_labels(test_set, class_index)
accuracy = mt.find_accuracy(estimated_output, actual_output)
error = b.test_model(test_set, class_index, numb_hidden_nodes, numb_outputs, v, w)
print('measured accuracy: ' + str(accuracy))
print('error: ' + str(error))
print('\n')

print('Test 5')
training_set_1 = set1 + set2 + set3 + set4 + set5
training_set_2 = set7 + set8 + set9 + set10
test_set = set6
#convert sets to numpy arrays
training_set_1 = np.asarray(training_set_1, dtype=float)
training_set_2 = np.asarray(training_set_2, dtype=float)
x_train_1, y_train_1 = data_handler.split_data_into_XY(training_set_1, class_index, first_attribute_index, last_attribute_index )
x_train_2, y_train_2 = data_handler.split_data_into_XY(training_set_2, class_index, first_attribute_index, last_attribute_index)
avg_features, updated_output = Knn.K_Nearest_Neighbor_with_feature_averaging(x_train_1, x_train_2, y_train_1, y_train_2, k, True)
training_set = data_handler.concatenate_feature_vals_with_labels(avg_features, updated_output)
v, w = b.find_model_1_hidden_layer(training_set, class_index, numb_hidden_nodes, numb_iterations, numb_outputs, learning_rate)
print('v weights found: ' + str(v) )
print('w weights found: ' + str(w))
estimated_output = b.get_estimated_output(test_set, class_index, numb_hidden_nodes, numb_outputs, v, w)
actual_output = data_handler.get_class_labels(test_set, class_index)
accuracy = mt.find_accuracy(estimated_output, actual_output)
error = b.test_model(test_set, class_index, numb_hidden_nodes, numb_outputs, v, w)
print('measured accuracy: ' + str(accuracy))
print('error: ' + str(error))
print('\n')


print('Test 6')
training_set_1 = set1 + set2 + set3 + set4 + set6
training_set_2 = set7 + set8 + set9 + set10
test_set = set5
#convert sets to numpy arrays
training_set_1 = np.asarray(training_set_1, dtype=float)
training_set_2 = np.asarray(training_set_2, dtype=float)
x_train_1, y_train_1 = data_handler.split_data_into_XY(training_set_1, class_index, first_attribute_index, last_attribute_index )
x_train_2, y_train_2 = data_handler.split_data_into_XY(training_set_2, class_index, first_attribute_index, last_attribute_index)
avg_features, updated_output = Knn.K_Nearest_Neighbor_with_feature_averaging(x_train_1, x_train_2, y_train_1, y_train_2, k, True)
training_set = data_handler.concatenate_feature_vals_with_labels(avg_features, updated_output)
v, w = b.find_model_1_hidden_layer(training_set, class_index, numb_hidden_nodes, numb_iterations, numb_outputs, learning_rate)
print('v weights found: ' + str(v) )
print('w weights found: ' + str(w))
estimated_output = b.get_estimated_output(test_set, class_index, numb_hidden_nodes, numb_outputs, v, w)
actual_output = data_handler.get_class_labels(test_set, class_index)
accuracy = mt.find_accuracy(estimated_output, actual_output)
error = b.test_model(test_set, class_index, numb_hidden_nodes, numb_outputs, v, w)
print('measured accuracy: ' + str(accuracy))
print('error: ' + str(error))
print('\n')

print('Test 7')
training_set_1 = set1 + set2 + set3 + set5 + set6
training_set_2 = set7 + set8 + set9 + set10
test_set = set4
#convert sets to numpy arrays
training_set_1 = np.asarray(training_set_1, dtype=float)
training_set_2 = np.asarray(training_set_2, dtype=float)
x_train_1, y_train_1 = data_handler.split_data_into_XY(training_set_1, class_index, first_attribute_index, last_attribute_index )
x_train_2, y_train_2 = data_handler.split_data_into_XY(training_set_2, class_index, first_attribute_index, last_attribute_index)
avg_features, updated_output = Knn.K_Nearest_Neighbor_with_feature_averaging(x_train_1, x_train_2, y_train_1, y_train_2, k, True)
training_set = data_handler.concatenate_feature_vals_with_labels(avg_features, updated_output)
v, w = b.find_model_1_hidden_layer(training_set, class_index, numb_hidden_nodes, numb_iterations, numb_outputs, learning_rate)
print('v weights found: ' + str(v) )
print('w weights found: ' + str(w))
estimated_output = b.get_estimated_output(test_set, class_index, numb_hidden_nodes, numb_outputs, v, w)
actual_output = data_handler.get_class_labels(test_set, class_index)
accuracy = mt.find_accuracy(estimated_output, actual_output)
error = b.test_model(test_set, class_index, numb_hidden_nodes, numb_outputs, v, w)
print('measured accuracy: ' + str(accuracy))
print('error' + str(error))
print('\n')

print('Test 8')
training_set_1 = set1 + set2 + set4 + set5 + set6
training_set_2 = set7 + set8 + set9 + set10
test_set = set3
#convert sets to numpy arrays
training_set_1 = np.asarray(training_set_1, dtype=float)
training_set_2 = np.asarray(training_set_2, dtype=float)
x_train_1, y_train_1 = data_handler.split_data_into_XY(training_set_1, class_index, first_attribute_index, last_attribute_index )
x_train_2, y_train_2 = data_handler.split_data_into_XY(training_set_2, class_index, first_attribute_index, last_attribute_index)
avg_features, updated_output = Knn.K_Nearest_Neighbor_with_feature_averaging(x_train_1, x_train_2, y_train_1, y_train_2, k, True)
training_set = data_handler.concatenate_feature_vals_with_labels(avg_features, updated_output)
v, w = b.find_model_1_hidden_layer(training_set, class_index, numb_hidden_nodes, numb_iterations, numb_outputs, learning_rate)
print('v weights found: ' + str(v) )
print('w weights found: ' + str(w))
estimated_output = b.get_estimated_output(test_set, class_index, numb_hidden_nodes, numb_outputs, v, w)
actual_output = data_handler.get_class_labels(test_set, class_index)
accuracy = mt.find_accuracy(estimated_output, actual_output)
error = b.test_model(test_set, class_index, numb_hidden_nodes, numb_outputs, v, w)
print('measured accuracy: ' + str(accuracy))
print('error: ' + str(error))
print('\n')

print('Test 9')
training_set_1 = set1 + set3 + set4 + set5 + set6
training_set_2 = set7 + set8 + set9 + set10
test_set = set2
#convert sets to numpy arrays
training_set_1 = np.asarray(training_set_1, dtype=float)
training_set_2 = np.asarray(training_set_2, dtype=float)
x_train_1, y_train_1 = data_handler.split_data_into_XY(training_set_1, class_index, first_attribute_index, last_attribute_index )
x_train_2, y_train_2 = data_handler.split_data_into_XY(training_set_2, class_index, first_attribute_index, last_attribute_index)
avg_features, updated_output = Knn.K_Nearest_Neighbor_with_feature_averaging(x_train_1, x_train_2, y_train_1, y_train_2, k, True)
training_set = data_handler.concatenate_feature_vals_with_labels(avg_features, updated_output)
v, w = b.find_model_1_hidden_layer(training_set, class_index, numb_hidden_nodes, numb_iterations, numb_outputs, learning_rate)
print('v weights found: ' + str(v) )
print('w weights found: ' + str(w))
estimated_output = b.get_estimated_output(test_set, class_index, numb_hidden_nodes, numb_outputs, v, w)
actual_output = data_handler.get_class_labels(test_set, class_index)
accuracy = mt.find_accuracy(estimated_output, actual_output)
error = b.test_model(test_set, class_index, numb_hidden_nodes, numb_outputs, v, w)
print('measured accuracy: ' + str(accuracy))
print('error: ' + str(error))
print('\n')

print('Test 10')
training_set_1 =  set2 + set3 + set4 + set5 + set6
training_set_2 = set7 + set8 + set9 + set10
test_set = set1
#convert sets to numpy arrays
training_set_1 = np.asarray(training_set_1, dtype=float)
training_set_2 = np.asarray(training_set_2, dtype=float)
x_train_1, y_train_1 = data_handler.split_data_into_XY(training_set_1, class_index, first_attribute_index, last_attribute_index )
x_train_2, y_train_2 = data_handler.split_data_into_XY(training_set_2, class_index, first_attribute_index, last_attribute_index)
avg_features, updated_output = Knn.K_Nearest_Neighbor_with_feature_averaging(x_train_1, x_train_2, y_train_1, y_train_2, k, True)
training_set = data_handler.concatenate_feature_vals_with_labels(avg_features, updated_output)
v, w = b.find_model_1_hidden_layer(training_set, class_index, numb_hidden_nodes, numb_iterations, numb_outputs, learning_rate)
print('v weights found: ' + str(v) )
print('w weights found: ' + str(w))
estimated_output = b.get_estimated_output(test_set, class_index, numb_hidden_nodes, numb_outputs, v, w)
actual_output = data_handler.get_class_labels(test_set, class_index)
accuracy = mt.find_accuracy(estimated_output, actual_output)
error = b.test_model(test_set, class_index, numb_hidden_nodes, numb_outputs, v, w)
print('measured accuracy: ' + str(accuracy))
print('error: ' + str(error))
print('\n')


