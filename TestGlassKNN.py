import common
import copy
import math
import backpropogation as b
import random
from random import shuffle
import metrics_testing as mt
import GetData as data_handler
import numpy as np
import KNearestNeighbor as Knn

#URL for dataset https://archive.ics.uci.edu/ml/datasets/glass+identification


data = common.read_csv('C:/Users/andre/PycharmProjects/MachineLearningFinal/Data/glass.csv')

# since first feature is just an id number, this doesn't provide any useful information
common.remove_nth_column(data, 0)

class_index = 9
first_attribute_index = 0
last_attribute_index = 8

#update class lables 1=>0, 2=> 1, 3=>2, 4=>3, 5=>4, 6=>5, 7=>6
for point in data:
    val = float(point[class_index]) - 1
    point[class_index] = str(val)

class_values = [0, 1, 2, 3, 4, 5, 6]

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
    listClass3 = []
    listClass4 = []
    listClass5 = []
    listClass6 = []
    data_copy = copy.deepcopy(data)

    for point in data_copy:
        class_val = point[class_index]
        if(float(class_val) == float('0')):
            listClass0.append(point)
        elif(float(class_val) == float('1')):
            listClass1.append(point)
        elif(float(class_val) == float('2')):
            listClass2.append(point)
        elif(float(class_val) == float('3')):
            listClass3.append(point)
        elif(float(class_val) == float('4')):
            listClass4.append(point)
        elif(float(class_val) == float('5')):
            listClass5.append(point)
        elif(float(class_val) == float('6')):
            listClass6.append(point)
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
    for i in range(0, len(listClass3)):
        point = listClass3[i]
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
    for i in range(0, len(listClass4)):
        point = listClass4[i]
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
    for i in range(0, len(listClass5)):
        point = listClass5[i]
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
    for i in range(0, len(listClass6)):
        point = listClass6[i]
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
k=3


print('Test 1')
training_set = set1 + set2 + set3 + set4 + set5 + set6 + set7 + set8 + set9
test_set = set10
#convert sets to numpy arrays
training_set = np.asarray(training_set, dtype=float)
test_set = np.asarray(test_set, dtype=float)
x_train, y_train = data_handler.split_data_into_XY(training_set, class_index, first_attribute_index, last_attribute_index )
x_test, y_test = data_handler.split_data_into_XY(test_set, class_index, first_attribute_index, last_attribute_index)
estimated_output = Knn.K_Nearest_Neighbor(x_train, x_test, y_train, y_test, k, True)
actual_output = data_handler.get_class_labels(test_set, class_index)
accuracy = mt.find_accuracy(estimated_output, actual_output)
precision = mt.find_precision_multiclass(estimated_output, actual_output, class_values)
recall = mt.find_recall_multiclass(estimated_output, actual_output, class_values)
print('measured accuracy: ' + str(accuracy))
print('measured precision: ' + str(precision))
print('measured recall: ' + str(recall))
print('\n')

print('Test 2')
training_set = set1 + set2 + set3 + set4 + set5 + set6 + set7 + set8 + set10
test_set = set9
#convert sets to numpy arrays
training_set = np.asarray(training_set, dtype=float)
test_set = np.asarray(test_set, dtype=float)
x_train, y_train = data_handler.split_data_into_XY(training_set, class_index, first_attribute_index, last_attribute_index )
x_test, y_test = data_handler.split_data_into_XY(test_set, class_index, first_attribute_index, last_attribute_index)
estimated_output = Knn.K_Nearest_Neighbor(x_train, x_test, y_train, y_test, k, True)
actual_output = data_handler.get_class_labels(test_set, class_index)
accuracy = mt.find_accuracy(estimated_output, actual_output)
precision = mt.find_precision_multiclass(estimated_output, actual_output, class_values)
recall = mt.find_recall_multiclass(estimated_output, actual_output, class_values)
print('measured accuracy: ' + str(accuracy))
print('measured precision: ' + str(precision))
print('measured recall: ' + str(recall))
print('\n')

print('Test 3')
training_set = set1 + set2 + set3 + set4 + set5 + set6 + set7 + set9 + set10
test_set = set8
#convert sets to numpy arrays
training_set = np.asarray(training_set, dtype=float)
test_set = np.asarray(test_set, dtype=float)
x_train, y_train = data_handler.split_data_into_XY(training_set, class_index, first_attribute_index, last_attribute_index )
x_test, y_test = data_handler.split_data_into_XY(test_set, class_index, first_attribute_index, last_attribute_index)
estimated_output = Knn.K_Nearest_Neighbor(x_train, x_test, y_train, y_test, k, True)
actual_output = data_handler.get_class_labels(test_set, class_index)
accuracy = mt.find_accuracy(estimated_output, actual_output)
precision = mt.find_precision_multiclass(estimated_output, actual_output, class_values)
recall = mt.find_recall_multiclass(estimated_output, actual_output, class_values)
print('measured accuracy: ' + str(accuracy))
print('measured precision: ' + str(precision))
print('measured recall: ' + str(recall))
print('\n')

print('Test 4')
training_set = set1 + set2 + set3 + set4 + set5 + set6 + set8 + set9 + set10
test_set = set7
#convert sets to numpy arrays
training_set = np.asarray(training_set, dtype=float)
test_set = np.asarray(test_set, dtype=float)
x_train, y_train = data_handler.split_data_into_XY(training_set, class_index, first_attribute_index, last_attribute_index )
x_test, y_test = data_handler.split_data_into_XY(test_set, class_index, first_attribute_index, last_attribute_index)
estimated_output = Knn.K_Nearest_Neighbor(x_train, x_test, y_train, y_test, k, True)
actual_output = data_handler.get_class_labels(test_set, class_index)
accuracy = mt.find_accuracy(estimated_output, actual_output)
precision = mt.find_precision_multiclass(estimated_output, actual_output, class_values)
recall = mt.find_recall_multiclass(estimated_output, actual_output, class_values)
print('measured accuracy: ' + str(accuracy))
print('measured precision: ' + str(precision))
print('measured recall: ' + str(recall))
print('\n')

print('Test 5')
training_set = set1 + set2 + set3 + set4 + set5 + set7 + set8 + set9 + set10
test_set = set6
#convert sets to numpy arrays
training_set = np.asarray(training_set, dtype=float)
test_set = np.asarray(test_set, dtype=float)
x_train, y_train = data_handler.split_data_into_XY(training_set, class_index, first_attribute_index, last_attribute_index )
x_test, y_test = data_handler.split_data_into_XY(test_set, class_index, first_attribute_index, last_attribute_index)
estimated_output = Knn.K_Nearest_Neighbor(x_train, x_test, y_train, y_test, k, True)
actual_output = data_handler.get_class_labels(test_set, class_index)
accuracy = mt.find_accuracy(estimated_output, actual_output)
precision = mt.find_precision_multiclass(estimated_output, actual_output, class_values)
recall = mt.find_recall_multiclass(estimated_output, actual_output, class_values)
print('measured accuracy: ' + str(accuracy))
print('measured precision: ' + str(precision))
print('measured recall: ' + str(recall))
print('\n')


print('Test 6')
training_set = set1 + set2 + set3 + set4 + set6 + set7 + set8 + set9 + set10
test_set = set5
#convert sets to numpy arrays
training_set = np.asarray(training_set, dtype=float)
test_set = np.asarray(test_set, dtype=float)
x_train, y_train = data_handler.split_data_into_XY(training_set, class_index, first_attribute_index, last_attribute_index )
x_test, y_test = data_handler.split_data_into_XY(test_set, class_index, first_attribute_index, last_attribute_index)
estimated_output = Knn.K_Nearest_Neighbor(x_train, x_test, y_train, y_test, k, True)
actual_output = data_handler.get_class_labels(test_set, class_index)
accuracy = mt.find_accuracy(estimated_output, actual_output)
precision = mt.find_precision_multiclass(estimated_output, actual_output, class_values)
recall = mt.find_recall_multiclass(estimated_output, actual_output, class_values)
print('measured accuracy: ' + str(accuracy))
print('measured precision: ' + str(precision))
print('measured recall: ' + str(recall))
print('\n')

print('Test 7')
training_set = set1 + set2 + set3 + set5 + set6 + set7 + set8 + set9 + set10
test_set = set4
#convert sets to numpy arrays
training_set = np.asarray(training_set, dtype=float)
test_set = np.asarray(test_set, dtype=float)
x_train, y_train = data_handler.split_data_into_XY(training_set, class_index, first_attribute_index, last_attribute_index )
x_test, y_test = data_handler.split_data_into_XY(test_set, class_index, first_attribute_index, last_attribute_index)
estimated_output = Knn.K_Nearest_Neighbor(x_train, x_test, y_train, y_test, k, True)
actual_output = data_handler.get_class_labels(test_set, class_index)
accuracy = mt.find_accuracy(estimated_output, actual_output)
precision = mt.find_precision_multiclass(estimated_output, actual_output, class_values)
recall = mt.find_recall_multiclass(estimated_output, actual_output, class_values)
print('measured accuracy: ' + str(accuracy))
print('measured precision: ' + str(precision))
print('measured recall: ' + str(recall))
print('\n')

print('Test 8')
training_set = set1 + set2 + set4 + set5 + set6 + set7 + set8 + set9 + set10
test_set = set3
#convert sets to numpy arrays
training_set = np.asarray(training_set, dtype=float)
test_set = np.asarray(test_set, dtype=float)
x_train, y_train = data_handler.split_data_into_XY(training_set, class_index, first_attribute_index, last_attribute_index )
x_test, y_test = data_handler.split_data_into_XY(test_set, class_index, first_attribute_index, last_attribute_index)
estimated_output = Knn.K_Nearest_Neighbor(x_train, x_test, y_train, y_test, k, True)
actual_output = data_handler.get_class_labels(test_set, class_index)
accuracy = mt.find_accuracy(estimated_output, actual_output)
precision = mt.find_precision_multiclass(estimated_output, actual_output, class_values)
recall = mt.find_recall_multiclass(estimated_output, actual_output, class_values)
print('measured accuracy: ' + str(accuracy))
print('measured precision: ' + str(precision))
print('measured recall: ' + str(recall))
print('\n')

print('Test 9')
training_set = set1 + set3 + set4 + set5 + set6 + set7 + set8 + set9 + set10
test_set = set2
#convert sets to numpy arrays
training_set = np.asarray(training_set, dtype=float)
test_set = np.asarray(test_set, dtype=float)
x_train, y_train = data_handler.split_data_into_XY(training_set, class_index, first_attribute_index, last_attribute_index )
x_test, y_test = data_handler.split_data_into_XY(test_set, class_index, first_attribute_index, last_attribute_index)
estimated_output = Knn.K_Nearest_Neighbor(x_train, x_test, y_train, y_test, k, True)
actual_output = data_handler.get_class_labels(test_set, class_index)
accuracy = mt.find_accuracy(estimated_output, actual_output)
precision = mt.find_precision_multiclass(estimated_output, actual_output, class_values)
recall = mt.find_recall_multiclass(estimated_output, actual_output, class_values)
print('measured accuracy: ' + str(accuracy))
print('measured precision: ' + str(precision))
print('measured recall: ' + str(recall))
print('\n')

print('Test 10')
training_set =  set2 + set3 + set4 + set5 + set6 + set7 + set8 + set9 + set10
test_set = set1
#convert sets to numpy arrays
training_set = np.asarray(training_set, dtype=float)
test_set = np.asarray(test_set, dtype=float)
x_train, y_train = data_handler.split_data_into_XY(training_set, class_index, first_attribute_index, last_attribute_index )
x_test, y_test = data_handler.split_data_into_XY(test_set, class_index, first_attribute_index, last_attribute_index)
estimated_output = Knn.K_Nearest_Neighbor(x_train, x_test, y_train, y_test, k, True)
actual_output = data_handler.get_class_labels(test_set, class_index)
accuracy = mt.find_accuracy(estimated_output, actual_output)
precision = mt.find_precision_multiclass(estimated_output, actual_output, class_values)
recall = mt.find_recall_multiclass(estimated_output, actual_output, class_values)
print('measured accuracy: ' + str(accuracy))
print('measured precision: ' + str(precision))
print('measured recall: ' + str(recall))
print('\n')
