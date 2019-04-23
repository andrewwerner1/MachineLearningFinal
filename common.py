from __future__ import division
import csv
import random
from random import shuffle
from numbers import Number
import copy
import math

# This file contains functionality shared by multiple algorithms
# and used for testing


#File contains helper methods (some methods common to multiple projects)
#and methods used by multiple files

def read_csv(file_name):
    file = []
    with open (file_name) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            file.append(row)
            #print(row)
    return file



def find_class_freq_dict(data, class_index):
    dict = {}
    for row in data:
        label = row[class_index]
        if label in dict:
            dict[label] += 1
        else:
            dict[label] = 1
    return dict


def read_file_space_delimeter(file_name):
    file = read_csv(file_name)
    new_file = []
    for line in file:
        row = line[0].split()
        new_file.append(row)
    return new_file


def remove_nth_item(point, n):
    point.pop(n)
    return point

def remove_nth_column(data, n):
    [row.pop(n) for row in data]

def retrieve_nth_column(data, n):
    data_copy = copy.deepcopy(data)
    col = []
    for row in data_copy:
        col.append(row[n])
    return col

def find_distance(v1, v2, non_numeric_distance_factor):
    distance = 0
    for x,y in zip(v1,v2):
        try:
            float1 = float(x)
            float2 = float(y)
            difference_squared = math.pow((float1 - float2), 2)
            distance = distance + math.sqrt(difference_squared)
        except ValueError:
            if x != y:
                # Not sure what to do here for non-numeric attributes so arbitrarily added 5 to distance
                distance = distance + non_numeric_distance_factor
    return distance


#Finds euclidean distance between two vectors
def find_distance_with_numeric_vectors(v1, v2):
    total_sum = sum([math.pow((float(x) - float(y)),2) for (x,y) in zip(v1, v2)])
    distance = math.sqrt(total_sum)
    return distance

#Finds distance between two vectors only considering dimensions given by
# indices vector
def find_distance_with_feature_indices(v1, v2, indices):
    sum = 0
    for index in indices:
        v1_index = v1[index]
        v2_index = v2[index]
        sum = sum + math.pow((float(v1_index) - float(v2_index)), 2)
    distance = math.sqrt(sum)
    return distance


def extract_columns_by_indices(data, indices):
    new_data = []
    for row in data:
        new_row = []
        for i in range(len(row)-1):
            if i in indices:
                new_row.append(row[i])
        new_data.append(new_row)
    return new_data


def split_data_in_half(data):
    half = len(data) / 2
    return data[:half], data[half:]


def find_numb_points_with_class_label(list_points, key, label_index):
    count = 0
    for point in list_points:
        class_label = point[label_index]
        if class_label == key:
            count = count + 1
    return count


def add_item_to_lists(list1, list2, list3, list4, list5, item, key, max_freq, label_index):
    count_list_1 = find_numb_points_with_class_label(list1, key, label_index)
    count_list_2 = find_numb_points_with_class_label(list2, key, label_index)
    count_list_3 = find_numb_points_with_class_label(list3, key, label_index)
    count_list_4 = find_numb_points_with_class_label(list4, key, label_index)
    count_list_5 = find_numb_points_with_class_label(list5, key, label_index)
    if count_list_1 < max_freq:
        list1.append(item)
    elif count_list_2 < max_freq:
        list2.append(item)
    elif count_list_3 < max_freq:
        list3.append(item)
    elif count_list_4 < max_freq:
        list4.append(item)
    elif count_list_5 < max_freq:
        list5.append(item)



def add_item_to_ten_lists(list1, list2, list3, list4, list5, list6,list7,list8,list9,list10, item, key, max_freq, label_index):
    count_list_1 = find_numb_points_with_class_label(list1, key, label_index)
    count_list_2 = find_numb_points_with_class_label(list2, key, label_index)
    count_list_3 = find_numb_points_with_class_label(list3, key, label_index)
    count_list_4 = find_numb_points_with_class_label(list4, key, label_index)
    count_list_5 = find_numb_points_with_class_label(list5, key, label_index)
    count_list_6 = find_numb_points_with_class_label(list6, key, label_index)
    count_list_7 = find_numb_points_with_class_label(list6, key, label_index)
    count_list_8 = find_numb_points_with_class_label(list6, key, label_index)
    count_list_9 = find_numb_points_with_class_label(list6, key, label_index)
    count_list_10 = find_numb_points_with_class_label(list6, key, label_index)
    if count_list_1 < max_freq:
        list1.append(item)
    elif count_list_2 < max_freq:
        list2.append(item)
    elif count_list_3 < max_freq:
        list3.append(item)
    elif count_list_4 < max_freq:
        list4.append(item)
    elif count_list_5 < max_freq:
        list5.append(item)
    elif count_list_6 < max_freq:
        list6.append(item)
    elif count_list_7 < max_freq:
        list7.append(item)
    elif count_list_8 < max_freq:
        list8.append(item)
    elif count_list_9 < max_freq:
        list9.append(item)
    elif count_list_10 < max_freq:
        list10.append(item)

def add_item_to_nth_list(list1, list2, list3, list4, list5, random_int, point):
    if random_int == 0:
        list1.append(point)
    elif random_int == 1:
        list2.append(point)
    elif random_int == 2:
        list3.append(point)
    elif random_int == 3:
        list4.append(point)
    elif random_int == 4:
        list5.append(point)

#Very helpful method for 5 way cross validation
#Creates 5 subsets of data with class label occurences
# given by dict parameter
def split_data_in_five_parts(data, dict, class_index):
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []
    numb_of_lists = 5
    data_copy = copy.deepcopy(data)
    for key, val in dict.items():
        for point in data:
            class_label = point[class_index]
            if class_label == key:
                data_copy.remove(point)
                add_item_to_lists(list1, list2, list3, list4, list5, point, key, val, class_index)
    # remove extra data points
    while len(data_copy) > 0:
        point = data_copy.pop()
        random_int = random.randint(0, 4)
        add_item_to_nth_list(list1, list2, list3, list4, list5, random_int, point)
    return list1, list2, list3, list4, list5


#Very helpful method for 5 way cross validation
#Creates 5 subsets of data with class label occurences
# given by dict parameter




#Creates a subset of the data with class label frequencies given by dict
#useful for creating pruning set
def create_sublist_with_same_class_distrubtion(data, dict, class_index):
    new_list = []
    data_copy = copy.deepcopy(data)
    for key,val in dict.iteritems():
        numb_points_moved = 0
        for point in data_copy:
            if numb_points_moved == val:
                break
            class_label = point[class_index]
            if class_label == key:
                new_list.append(point)
                data.remove(point)
                numb_points_moved = numb_points_moved + 1
    return new_list


#Note Not used for this project
def split_data_in_five_random_parts(data):
    shuffle(data)
    fifth = int(math.floor(len(data)/5))
    return data[0: fifth], data[fifth: 2*fifth], data[2*fifth : 3*fifth], data[3*fifth: 4*fifth], data[4*fifth : ]

#for continous values attributes, applies binary split depending on if attribute
# value is less than or equal to mean or greater than mean
# if less than or equal to mean, then sets attribute value to LEM otherwise
# if greater than mean, sets value to GM
def apply_discretizing_to_continuous_attr_values(data, dict_averages):
    for row in data:
        for j in range(len(row)):
            if j not in dict_averages:
                continue
            attr_value = float(row[j])
            mean_value = float(dict_averages[j])
            if attr_value <= mean_value / 4:
                row[j] = '<=1/4M'
            elif attr_value <= mean_value / 2:
                row[j] = '<=1/2M'
            elif attr_value <= ((3*mean_value)/4):
                row[j] = '<=3/4M'
            elif attr_value <= mean_value:
                row[j] = '<=M'
            elif attr_value <= (5/4) * mean_value:
                row[j] = '<=5M/4'
            elif attr_value <= (3/2) * mean_value:
                row[j] = '<=3M/2'
            elif attr_value <= (7/4) * mean_value:
                row[j] = '<=7M/4'
            elif attr_value <= 2 * mean_value:
                row[j] = '<=2M'
            else:
                row[j] = 'LARGE'

#Finds subset of a data set where class label equals preferred class label
def find_subset_data_with_class_label(training_set, preferred_class_label, class_index):
    subset = []
    for point in training_set:
        if point[class_index] == preferred_class_label:
            subset.append(point)
    return subset


def remove_points_with_missing_attributes(data):
    for point in data:
        if '?' in point:
            data.remove(point)

def remove_columns_that_dont_change(data):
    try:
        for i in range(len(data[0])):
            if i >= len(data[0]):
                return
            column = retrieve_nth_column(data, i)
            data_set = set(column)
            if len(data_set) == 1:
                remove_nth_column(data, i)
    except ValueError:
        return
