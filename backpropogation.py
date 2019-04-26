from __future__ import division
import copy
import random
import math
import common

#This file contains implementation of backpropogation algorithm
#For 2-layer multilayer network, uses diagram on page 281 of Introduction to Machine Learning by
#Ethem Alpaydin as a guide

#used to initialize a 2-d list of weights
def initialize_weights(dimension1, dimension2):
    weights = []
    for i in range(dimension1):
        list = []
        for j in range(dimension2):
            random_weight = random.uniform(-0.01, 0.01)
            list.append(random_weight)
        weights.append(list)
    return weights

#Computes dot product between two vectors
def fint_dot_product(vector1, vector2):
    dot_product_val = 0
    if len(vector1) != len(vector2):
        raise ValueError('Vectors are not of same length stupid!')
    for i in range(len(vector1)):
        val1 = vector1[i]
        val2 = vector2[i]
        dot_product_val += float(val1) * float(val2)
    return dot_product_val

#computes sigmoid function of two vectors
def sigmoid(w_h, point):
    dot_product_val = fint_dot_product(w_h, point)
    sigmoid_val = 1 / (1 + math.pow(math.e, -1 * dot_product_val))
    return sigmoid_val

#Computes weights that connect with output layer for 2-layer network
def compute_z_weights(numb_hidden_nodes, w, point):
    z = []
    for h in range(numb_hidden_nodes):
        # need to insert initial 1 in front of list of z values
        if h == 0:
            z.append(1)
        else:
            w_h = w[h]
            z_h = sigmoid(w_h, point)
            z.append(z_h)
    return z

#computes outputs for  2 layer network
def compute_y_outputs(v, z, numb_outputs):
    y = []
    for i in range(numb_outputs):
        v_i = v[i]
        y_i = fint_dot_product(v_i, z)
        y.append(y_i)
    return y

#computes outputs for  1 layer network
def compute_y_outputs_no_hidden_layer(w, input, numb_outputs):
    outputs = []
    for i in range(numb_outputs):
        w_i = w[i]
        O_i = fint_dot_product(w_i, input)
        try:
            output_i = math.pow(math.e, O_i)
        except OverflowError:
            output_i = 1E10
        outputs.append(output_i)
    normalizing_factor = find_estimate_normalizing_factor(numb_outputs, outputs)
    for i in range(numb_outputs):
        outputs[i] = outputs[i] / normalizing_factor
    return outputs

# returns class value and values to input features and appends a 1 to initial place of input features
def find_input_and_class_val(point, class_index):
    input = []
    for i in range(len(point)):
        if i == class_index:
            continue
        attr_val = point[i]
        input.append(attr_val)
    input.insert(0, '1')
    class_val = point[class_index]
    return input, class_val

# if class val corresponds to output node, then class val is 1, otherwise 0
def find_class_val(output_node_index, class_val_code):
    if float(output_node_index) == float(class_val_code):
        class_val = 1
    else:
        class_val = 0
    return class_val


#Finds the delta along the outer edges (connected to output nodes) of the network
def find_delta_v(numb_outputs, numb_hidden_nodes, learning_rate, y_estimate, class_val_code, z):
    delta_v = []
    for i in range(numb_outputs):
        class_val = find_class_val(i, class_val_code)
        delta_v_i = []
        for h in range(numb_hidden_nodes):
            delta_v_i_h = float(learning_rate) * (float(class_val) - float(y_estimate[i])) * float(z[h])
            delta_v_i.append(delta_v_i_h)
        delta_v.append(delta_v_i)
    return delta_v


# computes the delta of the weights for the edges connected to the input nodes
# for the 2-layer network only
def find_delta_w(numb_outputs, numb_hidden_nodes, learning_rate, y_estimate, class_val_code, z, v, input):
    delta_w = []
    for h in range(numb_hidden_nodes):
        back_sum = 0
        delta_w_h = []
        for i in range(numb_outputs):
            v_h_i = v[i][h]
            class_val = find_class_val(i, class_val_code)
            back_sum += (float(class_val) - float(y_estimate[i])) * float(v_h_i)
        for j in range(len(input)):
            x_j = input[j]
            delta_w_h_j = float(learning_rate) * float(back_sum) * float(z[h]) * (1 - float(z[h])) * float(x_j)
            delta_w_h.append(delta_w_h_j)
        delta_w.append(delta_w_h)
    return delta_w

#updates v weights using the delta_v's computed
def update_v_weights(numb_outputs, numb_hidden_nodes, delta_v, v):
    for i in range(numb_outputs):
        for h in range(numb_hidden_nodes):
            v[i][h] += delta_v[i][h]

#updates w weights using the delta_w's computed
def update_w_weights(numb_hidden_nodes, numb_features, delta_w, w):
    for h in range(numb_hidden_nodes):
        for j in range(numb_features):
            w[h][j] += delta_w[h][j]

#Finds two-level model
def find_model_1_hidden_layer(training_set, class_index, numb_hidden_nodes, numb_iterations, numb_outputs, learning_rate):
    numb_features = len(training_set[0])
    w = initialize_weights(numb_hidden_nodes, numb_features)
    v = initialize_weights(numb_outputs, numb_hidden_nodes)
    for i in range(numb_iterations):
        for point in training_set:
            input, class_val_code = find_input_and_class_val(point, class_index)
            z = compute_z_weights(numb_hidden_nodes, w, input)
            y_estimate = compute_y_outputs(v, z, numb_outputs)
            delta_v = find_delta_v(numb_outputs, numb_hidden_nodes, learning_rate, y_estimate, class_val_code, z)
            delta_w = find_delta_w(numb_outputs, numb_hidden_nodes, learning_rate, y_estimate, class_val_code, z, v, input)
            update_v_weights(numb_outputs, numb_hidden_nodes, delta_v, v)
            update_w_weights(numb_hidden_nodes, numb_features, delta_w, w)
    return v, w

def find_model_2_hidden_layer(training_set, class_index, numb_hidden_nodes1,numb_hidden_nodes2, numb_iterations, numb_outputs, learning_rate):
    numb_features = len(training_set[0])
    w = initialize_weights(numb_hidden_nodes1, numb_features)
    v1 = initialize_weights(numb_hidden_nodes2, numb_hidden_nodes1)
    v2 = initialize_weights(numb_outputs, numb_hidden_nodes2)
    for i in range(numb_iterations):
        for point in training_set:
            input, class_val_code = find_input_and_class_val(point, class_index)
            z1 = compute_z_weights(numb_hidden_nodes1, w, input)
            z2 = compute_z_weights(numb_hidden_nodes2, v1, z1)
            y_estimate = compute_y_outputs(v2, z2, numb_outputs)
            delta_v2 = find_delta_v(numb_outputs, numb_hidden_nodes2, learning_rate, y_estimate, class_val_code, z2)
            delta_v1 = find_delta_w(numb_hidden_nodes2, numb_hidden_nodes1, learning_rate, y_estimate, class_val_code, z2, v2, z1)
            delta_w = find_delta_w(numb_hidden_nodes2, numb_hidden_nodes1, learning_rate, y_estimate, class_val_code, z1, v1, input)
            update_v_weights(numb_outputs, numb_hidden_nodes2, delta_v2, v2)
            update_v_weights(numb_hidden_nodes2, numb_hidden_nodes1, delta_v1, v1)
            update_w_weights(numb_hidden_nodes1, numb_features, delta_w, w)
    return v1, v2, w

#Finds single-level model
def find_model_no_hidden_layer(training_set, class_index, numb_iterations, numb_outputs, learning_rate):
    numb_features = len(training_set[0])
    w = initialize_weights(numb_outputs, numb_features)
    for i in range(numb_iterations):
        for point in training_set:
            input, class_val_code = find_input_and_class_val(point, class_index)
            y_estimate = compute_y_outputs_no_hidden_layer(w, input, numb_outputs)
            delta_w = find_delta_v(numb_outputs, numb_features, learning_rate, y_estimate, class_val_code, input)
            update_w_weights(numb_outputs, numb_features, delta_w, w)
    return  w


#Region Below this point used for testing

def find_intermediate_outputs(numb_outputs, numb_hidden_nodes, v, w, input):
    outputs = []
    for i in range(numb_outputs):
        O_i = 0
        for h in range(numb_hidden_nodes):
            w_h = w[h]
            if h == 0:
                z_h = 1
            else:
                z_h = sigmoid(w_h, input)
            O_i += float(v[i][h]) * float(z_h)
        outputs.append(O_i)
    return outputs


def find_estimate_normalizing_factor(numb_outputs, intermediate_outputs):
    normalizing_factor = 0
    for i in range(numb_outputs):
        O_k = intermediate_outputs[i]
        try:
            val = math.pow(math.e, O_k)
        except OverflowError:
            val = 1E10
        normalizing_factor += val
    return normalizing_factor


def find_estimated_class_value_code(numb_outputs, numb_hidden_nodes, v, w, input):
    intermediate_outputs = find_intermediate_outputs(numb_outputs, numb_hidden_nodes, v, w, input)
    normalizing_factor = find_estimate_normalizing_factor(numb_outputs, intermediate_outputs)
    estimated_class_val_code = []
    for i in range(numb_outputs):
        O_i = intermediate_outputs[i]
        numerator = math.pow(math.e, O_i)
        estimated_class_val = numerator / normalizing_factor
        estimated_class_val_code.append(estimated_class_val)
    return estimated_class_val_code


def find_max_index(numb_outputs, class_val_code):
    max = class_val_code[0]
    max_index = 0
    for i in range(numb_outputs):
        val = class_val_code[i]
        if val > max:
            max = val
            max_index = i
    return max_index



def find_estimated_class_val(numb_outputs, numb_hidden_nodes, v, w, input):
    class_val_code = find_estimated_class_value_code(numb_outputs, numb_hidden_nodes, v, w, input)
    max_index = find_max_index(numb_outputs, class_val_code)
    return max_index


def find_estimated_class_val_no_hidden_layer(input, w, numb_outputs, class_index):
    class_val_code = compute_y_outputs_no_hidden_layer(w, input, numb_outputs)
    max_index = find_max_index(numb_outputs, class_val_code)
    return max_index


def test_model_no_hidden_layer(test_set, class_index, numb_outputs, w):
    errors = 0
    for point in test_set:
        input, class_val = find_input_and_class_val(point, class_index)
        estimated_class_val = find_estimated_class_val_no_hidden_layer(input, w, numb_outputs, class_index)
        if float(estimated_class_val) != float(class_val):
            errors += 1
    return errors / len(test_set)


def test_model(test_set, class_index, numb_hidden_nodes, numb_outputs, v, w):
    errors = 0
    for point in test_set:
        input, class_val = find_input_and_class_val(point, class_index)
        estimated_class_val = find_estimated_class_val(numb_outputs, numb_hidden_nodes, v, w, input)
        if float(estimated_class_val) != float(class_val):
            errors += 1
    return errors / len(test_set)

def test_model_accuracy(test_set, class_index, numb_hidden_nodes, numb_outputs, v, w):
    correct = 0
    for point in test_set:
        input, class_val = find_input_and_class_val(point, class_index)
        estimated_class_val = find_estimated_class_val(numb_outputs, numb_hidden_nodes, v, w, input)
        if float(estimated_class_val) == float(class_val):
            correct += 1
    return correct / len(test_set)


def get_estimated_output(test_set, class_index, numb_hidden_nodes, numb_outputs, v, w):
    estimates = []
    for point in test_set:
        input, class_val = find_input_and_class_val(point, class_index)
        estimated_class_val = find_estimated_class_val(numb_outputs, numb_hidden_nodes, v, w, input)
        estimates.append(estimated_class_val)
    return estimates


#gets actual output from ANN without modyfing it to represent class value
def get_estimated_output_code(test_set, class_index, numb_hidden_nodes, numb_outputs, v, w):
    output_codes = []
    for point in test_set:
        input, class_val = find_input_and_class_val(point, class_index)
        class_val_code = find_estimated_class_value_code(numb_outputs, numb_hidden_nodes, v, w, input)
        output_codes.append(class_val_code)
    return output_codes