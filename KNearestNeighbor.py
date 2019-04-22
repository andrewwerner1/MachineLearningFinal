# -*- coding: utf-8 -*-
import numpy as np
import copy


def find_k_nearest_neighbors(test_example, training_data, k):
    distances = []
    neighbors = []
    number_of_training_examples = training_data.shape[0]
    for i in range(0, number_of_training_examples):
        training_example = training_data[i]
        dist = find_distance(training_example, test_example)
        distances.append((i, dist))
    distances.sort(key=operator.itemgetter(1))
    for x in range(k):
        neighbors.append()


def find_distance(vector1, vector2):
    difference = vector1 - vector2
    distance = (np.dot(difference ,difference)) ** .5
    return distance;


def K_Nearest_Neighbor(training_data, test_data, k):
    for vector in test_data:
        K_nearest_neihbors = find_k_nearest_neighbors(vector, training_data, k);
