import math
import numpy as np
from typing import List, Callable

# function for popping a row from the matrix (not used after all)
def poprow(dataset, rowno:int):
    df_t = dataset.T
    popped_row = df_t.pop(rowno)
    return df_t.T

# poping rows that have Nan values (not)
def filterNaN(dataset):
    for i in range(0, len(dataset)):
        if math.isnan(dataset[i]):
            poprow(dataset, )

def categorical_list(array:np.ndarray) -> List:
    res = []
    return [res.append(element) for element in array if element not in res]

# a hot encoder that works in 1 dimension it's different from the standard hotencoders
# specifically it will not create a separate dimension for each category, rather than,
# numerically code them in the same vector (column)
def oneDimHotEncoder(matrix):    
    temp = matrix[0]
    
    keys = np.array([]) #storing the keys we want to hot encode
    encoders = np.array([]) #the hot encoders 
    encodedmatrix = np.array([]) #the hot encoded matrix

    encoders = np.append(encoders, 0)
    keys = np.append(keys, temp)
    encodedmatrix = np.append(encodedmatrix, 0)

    for j in range(1, len(matrix)):
        if matrix[j] != temp:
            temp = matrix[j]
            if temp not in keys:
                keys = np.append(keys, temp)
                encoders = np.append(encoders, encoders[-1] + 1)
        encodedmatrix = np.append(encodedmatrix, encoders[np.where(keys == temp)])
    return encodedmatrix, encoders, keys

#default sgn function (not used)
def sign(n:float):
    if n >+ 0:
        return 1
    else:
        return -1
class one_hot_encoder:

    def __init__(self, data, categories=None) -> None:
        """
        Custom one-hot encoder for a given list of data and categories.

        Parameters:
        - data: List or NumPy array containing categorical data.
        - categories: List of unique categories. If None, unique categories are inferred from the data.

        Returns:
        - One-hot encoded NumPy array.
        - List of unique categories.
        """
        self.categories = categories

        if self.categories is None:
            self.categories = np.unique(data)

        self.encoded = np.zeros((len(data), len(self.categories)), dtype=int)

        for i, category in enumerate(self.categories):
            self.encoded[:, i] = (data == category).astype(int)

    def reverse(self, encoded_data):
        """
        Reverse the one-hot encoding to obtain the original categorical data.

        Parameters:
        - encoded_data: One-hot encoded NumPy array.
        - categories: List of unique categories.

        Returns:
        - List of original categorical data.
        """
        reversed_data = np.array(self.categories)[np.argmax(encoded_data)]
        return reversed_data


#inner product function for vectors (not used)
def innerprod(vec1, vec2):
    prod = 0
    if len(vec1) != len(vec2):
        print(vec1, vec2)
        return print("Vectors must be of the same size")
    for i in range(0, len(vec1)):
        prod += vec1[i] * vec2[i]  
    return prod

def identity(x):
    return x

def d_identity(x):
    return 1

def binary_step(x):
    if x < 0: return 0
    else: return 1

def d_binary_step(x):
    if x != 0: return 0

def logistic(x):
    return 1/(1 + np.exp(x))

def d_logistic(x):
    return logistic(x) * (1 - logistic(x))

def tanh(x):
    return (2/(1 + np.exp(-2*x))) - 1

def d_tanh(x):
    return 1 - tanh(x)^2

def arctan(x):
    return np.arctan(x)

def d_arctain(x):
    return 1/(x^2 + 1)

def softplus(x):
    return np.log(1 + np.exp(x))

def d_softlpus(x):
    1/ (1 + np.exp^(-x))

# sigmoid neuron activation function
def sig(x):
 return 1/(1 + np.exp(-x))

# ReLU (rectified linear unit) neuron activation function
def ReLU(x):
    return np.maximum(x, 0)

def d_ReLU(x):
    if x < 0: return 0
    else: return 1

class k_fold:

    # last layer neuron activation function
    def softmax(Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A.tolist()
    
    # ReLU (rectified linear unit) neuron activation function
    def ReLU(Z):
        return np.maximum(Z, 0)



# method for sturges classification before. Will be used when making the hisotgramms
# sturges method is a safe and dedicated way to decide the size of each bin so that
# the classification of the whole data is balanced
def sturgesrule(data):
    q = round(1 + 3.32*math.log10(len(data)))
    R = math.ceil(max(data)) - math.floor(min(data))
    c = math.ceil(R/q)
    return range(int(math.floor(min(data))), int(math.ceil(max(data))), int(c))

def sturges_grouper(inp , strg_step:int):
    temp = np.zeros(len(inp), dtype=int)
    for i in range(len(inp)):
        temp[i] = int(inp[i] // strg_step)
    return temp

def ReLU_deriv(Z):
        return int(Z > 0)

def sig_deriv(Z):
    return sig(Z)/(1-sig(Z))

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size