import numpy as np
import operator
import GetData as dataHandler

#this seems like a good source  https://anujkatiyal.com/blog/2017/10/01/ml-knn/#.XL0GXOhKhPY

#resource used for this code https://anujkatiyal.com/blog/2017/10/01/ml-knn/#.XL26Z-hKjct

def find_k_nearest_neighbors(x_train, test_point, k):
    distances = []
    neighbors = []
    number_of_training_examples = x_train.shape[0]
    for i in range(0, number_of_training_examples):
        training_example = x_train[i]
        dist = find_distance(training_example, test_point)
        distances.append((i, dist))
    distances.sort(key=operator.itemgetter(1))
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def find_distance(vector1, vector2):
    difference = vector1 - vector2
    distance = (np.dot(difference ,difference)) ** .5
    return distance



def predictKnnClassForRegression(neighbor_indices, y_train):
    pass

def predictKnnClassForClassification(neighbor_indices, y_train):
    classVotes = {}
    for i in range(0, len(neighbor_indices)):
        neighbor_index = neighbor_indices[i]
        neighbor = y_train[neighbor_index][0]
        if neighbor in classVotes:
            classVotes[neighbor] += 1
        else:
            classVotes[neighbor] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def find_averaged_feature_vals(neighbor_indices, x_train, k):
    average_features = []
    numb_of_features = len(x_train[0])
    for feature_index in range(numb_of_features):
        feature_sum = 0
        for i in range(len(neighbor_indices)):
            neighbor_index = neighbor_indices[i]
            feature_val = x_train[neighbor_index][feature_index]
            feature_sum += feature_val
        feature_avg = feature_sum / k
        average_features.append(feature_avg)
    return average_features

def K_Nearest_Neighbor(x_train, x_test, y_train, y_test, k, classification=True):
    output_classes = []
    for i in range(0, x_test.shape[0]):
        neighbor_indices = find_k_nearest_neighbors(x_train, x_test[i], k)
        if(classification):
            predictedClass = predictKnnClassForClassification(neighbor_indices, y_train)
        else:
            predictedClass = predictKnnClassForRegression(neighbor_indices, y_train)
        output_classes.append((predictedClass))
    return output_classes





def K_Nearest_Neighbor_with_feature_averaging(x_train, x_test, y_train, y_test, k, classification=True):
    output_classes = []
    averaged_feature_vals = []
    for i in range(0, x_test.shape[0]):
        neighbor_indices = find_k_nearest_neighbors(x_train, x_test[i], k)
        if(classification):
            predictedClass = predictKnnClassForClassification(neighbor_indices, y_train)
            averaged_feature_vals_row = find_averaged_feature_vals(neighbor_indices, x_train, k)
        else:
            predictedClass = predictKnnClassForRegression(neighbor_indices, y_train)
        output_classes.append((predictedClass))
        averaged_feature_vals.append(averaged_feature_vals_row)
    return averaged_feature_vals, output_classes

#test
#print('test')
#dataset = dataHandler.get_data_from_file("C:/Users/andre/PycharmProjects/MachineLearningFinal/DataIris.csv")
#split_data_sets = dataHandler.get_k_folds(dataset, k=5)#

#test_set = split_data_sets[0]
#x_test, y_test = dataHandler.split_data_into_XY(test_set, 4, 0, 3)
#training_set = dataHandler.concatenate_sets(split_data_sets, 1, 5)
#x_train, y_train = dataHandler.split_data_into_XY(training_set, 4, 0, 3)


#k = 4
#K_Nearest_Neighbor(x_train, x_test, y_train, y_test, k, True)
