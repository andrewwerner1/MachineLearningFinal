# -*- coding: utf-8 -*-
import numpy as np
import copy


def find_k_nearest_neighbors(vector, training_data, k):
    training_data_copy = copy.deepcopy(training_data)
    for i in range(1, k):
        pass


def find_distance(vector1, vector2):
    difference = vector1 - vector2
    distance = (np.dot(difference ,difference)) ** .5
    return distance;


def K_Nearest_Neighbor(training_data, test_data, k):
    for vector in test_data:
        K_nearest_neihbors = find_k_nearest_neighbors(vector, training_data, k);