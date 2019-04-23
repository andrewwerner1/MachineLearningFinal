import numpy as np


def get_data_from_file(filename):
    #dataset = np.genfromtxt(filename, delimiter=",", dtype=None)
    dataset = np.loadtxt(filename, delimiter=",")
    #randomly shuffle data
    np.random.shuffle(dataset)
    return dataset

def get_class_labels(dataset, class_index):
    labels = []
    for point in dataset:
        label = point[class_index]
        labels.append(label)
    return labels


def split_data_into_XY(dataset, class_index, first_attribute_index, last_attribute_index):
    X = dataset[:, first_attribute_index:last_attribute_index]
    Y_data = dataset[:, class_index].reshape(len(X), 1)
    # Create a container array around Y
    Y = np.array(Y_data)
    return X, Y

def get_k_folds(dataset, k):
    #shuffle data (in case not already shuffled
    np.random.shuffle(dataset)
    split_data_sets = np.split(dataset, k)
    return split_data_sets

def concatenate_sets(training_sets, start_index, length):
    total_set = training_sets[start_index]
    for i in range(start_index+1, length):
        total_set = np.concatenate([total_set, training_sets[i]])
    return total_set


#test
#print('test')
#dataset = get_data_from_file("C:/Users/andre/PycharmProjects/MachineLearningFinal/Data/Iris.csv")
#split_data_sets = get_k_folds(dataset, k=5)

#k_fold_first_set = split_data_sets[0]
#X,Y = split_data_into_XY(k_fold_first_set, 4, 0, 3)

#reformed_dataset = concatenate_sets(split_data_sets, 0, 5)

#vowel = ''
