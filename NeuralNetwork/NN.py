# -*- coding: utf-8 -*-
"""
NeuralNetwork.ipynb
"""

import numpy as np

def matrix_multiplication(matrix1, matrix2):
    """
    Given matrices matrix1 and matrix2, return the multiplication of them
    """

    # Initialize results to 0
    result = np.zeros((matrix1.shape[0], matrix2.shape[1]))

    # Get non-zero indices
    matrix1_non_zero_indices = matrix1.nonzero()

    # Compute matrix multiplication
    if matrix1.shape[0] == 1 and len(matrix1_non_zero_indices[1]) == 1:
        result = matrix2[matrix1_non_zero_indices[1], :]

    elif matrix1.shape[1] == 1 and len(matrix1_non_zero_indices[0]) == 1:
        result[matrix1_non_zero_indices[0], :] = matrix2 * matrix1[matrix1_non_zero_indices[0], 0]

    else:
        result = np.matmul(matrix1, matrix2)

    return result

def value_function(s, weights):
    """
    Compute value of input s given the weights of a neural network
    """

    psi = matrix_multiplication(s, weights[0]["W"]) + weights[0]["b"]
    x = np.maximum(np.zeros_like(psi), psi)
    v = matrix_multiplication(x, weights[1]["W"]) + weights[1]["b"]

    return v

# Test of neural network fragment, will be similar to applying to environment

#  num_states = ?, num_hidden_layer = 1, and num_hidden_units = num_states*2
num_hidden_layer = 1
s = np.array([[0, 0, 0, 1, 0]]) # FIX ME need num_states, currently 5

data = np.load("FIX ME!!!") # preloaded data or replace np.load() with data
weights = [dict() for i in range(num_hidden_layer+1)]
weights[0]["W"] = data["W0"]
weights[0]["b"] = data["b0"]
weights[1]["W"] = data["W1"]
weights[1]["b"] = data["b1"]

estimated_value = value_function(s, weights)
print ("Estimated value: {}".format(estimated_value))

assert(np.allclose(estimated_value, [[ "FIX ME: test values needed" ]]))