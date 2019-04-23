import common
import copy
import math
import backpropogation as b
import random
from random import shuffle
import metrics_testing as mt
import GetData as data_handler
import KNearestNeighbor as Knn


# DataSet URL: https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/ (breast-cancer-wisconsin.data)
#Class 2 for benign and 4 for malignant



data = common.read_csv("C:/Users/andre/PycharmProjects/MachineLearningFinal/Data/breastCancer.csv")

# since first feature is just an id number, this doesn't provide any useful information
common.remove_nth_column(data, 0)

class_index = 9
first_attribute_index = 0
last_attribute_index = 8

#update class lables 2=>0 and 4=> 1
for point in data:
    if point[class_index] == '2':
        point[class_index] = '0'
    elif point[class_index] == '4':
        point[class_index] = '1'


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
    data_copy = copy.deepcopy(data)

    for point in data_copy:
        class_val = point[class_index]
        if(float(class_val) == float('0')):
            listClass0.append(point)
        else:
            listClass1.append(point)
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
k=5

print('Test 1')
training_set = set1 + set2 + set3 + set4 + set5 + set6 + set7 + set8 + set9
test_set = set10
x_train, y_train = data_handler.split_data_into_XY(training_set, class_index, first_attribute_index, last_attribute_index )
x_test, y_test = data_handler.split_data_into_XY(test_set, class_index, first_attribute_index, last_attribute_index)
estimated_output = Knn.K_Nearest_Neighbor(x_train, x_test, y_train, y_test, k, True)
actual_output = data_handler.get_class_labels(test_set, class_index)
accuracy = mt.find_accuracy(estimated_output, actual_output)
precision = mt.find_precision(estimated_output, actual_output)
recall = mt.find_recall(estimated_output, actual_output)
print('measured accuracy: ' + str(accuracy))
print('measured precision: ' + str(precision))
print('measured recall: ' + str(recall))
print('\n')

print('Test 2')
training_set = set1 + set2 + set3 + set4 + set5 + set6 + set7 + set8 + set10
test_set = set9
x_train, y_train = data_handler.split_data_into_XY(training_set, class_index, first_attribute_index, last_attribute_index )
x_test, y_test = data_handler.split_data_into_XY(test_set, class_index, first_attribute_index, last_attribute_index)
estimated_output = Knn.K_Nearest_Neighbor(x_train, x_test, y_train, y_test, k, True)
actual_output = data_handler.get_class_labels(test_set, class_index)
accuracy = mt.find_accuracy(estimated_output, actual_output)
precision = mt.find_precision(estimated_output, actual_output)
recall = mt.find_recall(estimated_output, actual_output)
print('measured accuracy: ' + str(accuracy))
print('measured precision: ' + str(precision))
print('measured recall: ' + str(recall))
print('\n')

print('Test 3')
training_set = set1 + set2 + set3 + set4 + set5 + set6 + set7 + set9 + set10
test_set = set8
x_train, y_train = data_handler.split_data_into_XY(training_set, class_index, first_attribute_index, last_attribute_index )
x_test, y_test = data_handler.split_data_into_XY(test_set, class_index, first_attribute_index, last_attribute_index)
estimated_output = Knn.K_Nearest_Neighbor(x_train, x_test, y_train, y_test, k, True)
actual_output = data_handler.get_class_labels(test_set, class_index)
accuracy = mt.find_accuracy(estimated_output, actual_output)
precision = mt.find_precision(estimated_output, actual_output)
recall = mt.find_recall(estimated_output, actual_output)
print('measured accuracy: ' + str(accuracy))
print('measured precision: ' + str(precision))
print('measured recall: ' + str(recall))
print('\n')

print('Test 4')
training_set = set1 + set2 + set3 + set4 + set5 + set6 + set8 + set9 + set10
test_set = set7
x_train, y_train = data_handler.split_data_into_XY(training_set, class_index, first_attribute_index, last_attribute_index )
x_test, y_test = data_handler.split_data_into_XY(test_set, class_index, first_attribute_index, last_attribute_index)
estimated_output = Knn.K_Nearest_Neighbor(x_train, x_test, y_train, y_test, k, True)
actual_output = data_handler.get_class_labels(test_set, class_index)
accuracy = mt.find_accuracy(estimated_output, actual_output)
precision = mt.find_precision(estimated_output, actual_output)
recall = mt.find_recall(estimated_output, actual_output)
print('measured accuracy: ' + str(accuracy))
print('measured precision: ' + str(precision))
print('measured recall: ' + str(recall))
print('\n')

print('Test 5')
training_set = set1 + set2 + set3 + set4 + set5 + set7 + set8 + set9 + set10
test_set = set6
x_train, y_train = data_handler.split_data_into_XY(training_set, class_index, first_attribute_index, last_attribute_index )
x_test, y_test = data_handler.split_data_into_XY(test_set, class_index, first_attribute_index, last_attribute_index)
estimated_output = Knn.K_Nearest_Neighbor(x_train, x_test, y_train, y_test, k, True)
actual_output = data_handler.get_class_labels(test_set, class_index)
accuracy = mt.find_accuracy(estimated_output, actual_output)
precision = mt.find_precision(estimated_output, actual_output)
recall = mt.find_recall(estimated_output, actual_output)
print('measured accuracy: ' + str(accuracy))
print('measured precision: ' + str(precision))
print('measured recall: ' + str(recall))
print('\n')


print('Test 6')
training_set = set1 + set2 + set3 + set4 + set6 + set7 + set8 + set9 + set10
test_set = set5
x_train, y_train = data_handler.split_data_into_XY(training_set, class_index, first_attribute_index, last_attribute_index )
x_test, y_test = data_handler.split_data_into_XY(test_set, class_index, first_attribute_index, last_attribute_index)
estimated_output = Knn.K_Nearest_Neighbor(x_train, x_test, y_train, y_test, k, True)
actual_output = data_handler.get_class_labels(test_set, class_index)
accuracy = mt.find_accuracy(estimated_output, actual_output)
precision = mt.find_precision(estimated_output, actual_output)
recall = mt.find_recall(estimated_output, actual_output)
print('measured accuracy: ' + str(accuracy))
print('measured precision: ' + str(precision))
print('measured recall: ' + str(recall))
print('\n')

print('Test 7')
training_set = set1 + set2 + set3 + set5 + set6 + set7 + set8 + set9 + set10
test_set = set4
x_train, y_train = data_handler.split_data_into_XY(training_set, class_index, first_attribute_index, last_attribute_index )
x_test, y_test = data_handler.split_data_into_XY(test_set, class_index, first_attribute_index, last_attribute_index)
estimated_output = Knn.K_Nearest_Neighbor(x_train, x_test, y_train, y_test, k, True)
actual_output = data_handler.get_class_labels(test_set, class_index)
accuracy = mt.find_accuracy(estimated_output, actual_output)
precision = mt.find_precision(estimated_output, actual_output)
recall = mt.find_recall(estimated_output, actual_output)
print('measured accuracy: ' + str(accuracy))
print('measured precision: ' + str(precision))
print('measured recall: ' + str(recall))
print('\n')

print('Test 8')
training_set = set1 + set2 + set4 + set5 + set6 + set7 + set8 + set9 + set10
test_set = set3
x_train, y_train = data_handler.split_data_into_XY(training_set, class_index, first_attribute_index, last_attribute_index )
x_test, y_test = data_handler.split_data_into_XY(test_set, class_index, first_attribute_index, last_attribute_index)
estimated_output = Knn.K_Nearest_Neighbor(x_train, x_test, y_train, y_test, k, True)
actual_output = data_handler.get_class_labels(test_set, class_index)
accuracy = mt.find_accuracy(estimated_output, actual_output)
precision = mt.find_precision(estimated_output, actual_output)
recall = mt.find_recall(estimated_output, actual_output)
print('measured accuracy: ' + str(accuracy))
print('measured precision: ' + str(precision))
print('measured recall: ' + str(recall))
print('\n')

print('Test 9')
training_set = set1 + set3 + set4 + set5 + set6 + set7 + set8 + set9 + set10
test_set = set2
x_train, y_train = data_handler.split_data_into_XY(training_set, class_index, first_attribute_index, last_attribute_index )
x_test, y_test = data_handler.split_data_into_XY(test_set, class_index, first_attribute_index, last_attribute_index)
estimated_output = Knn.K_Nearest_Neighbor(x_train, x_test, y_train, y_test, k, True)
actual_output = data_handler.get_class_labels(test_set, class_index)
accuracy = mt.find_accuracy(estimated_output, actual_output)
precision = mt.find_precision(estimated_output, actual_output)
recall = mt.find_recall(estimated_output, actual_output)
print('measured accuracy: ' + str(accuracy))
print('measured precision: ' + str(precision))
print('measured recall: ' + str(recall))
print('\n')

print('Test 10')
training_set =  set2 + set3 + set4 + set5 + set6 + set7 + set8 + set9 + set10
test_set = set1
x_train, y_train = data_handler.split_data_into_XY(training_set, class_index, first_attribute_index, last_attribute_index )
x_test, y_test = data_handler.split_data_into_XY(test_set, class_index, first_attribute_index, last_attribute_index)
estimated_output = Knn.K_Nearest_Neighbor(x_train, x_test, y_train, y_test, k, True)
actual_output = data_handler.get_class_labels(test_set, class_index)
accuracy = mt.find_accuracy(estimated_output, actual_output)
precision = mt.find_precision(estimated_output, actual_output)
recall = mt.find_recall(estimated_output, actual_output)
print('measured accuracy: ' + str(accuracy))
print('measured precision: ' + str(precision))
print('measured recall: ' + str(recall))
print('\n')




