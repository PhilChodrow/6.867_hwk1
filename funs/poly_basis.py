
# coding: utf-8

import numpy as np
from itertools import product

def pad(X):
    '''
    add a column of ones as the first column of the array X.
    params:
        X, an np.array() of dimension 2
    returns: 
        padded, an np.array() of dimension 2 with one more column than X. 
    '''
    padded = np.ones((X.shape[0], X.shape[1] + 1))
    padded[:,1:] = X
    return padded

def unique_rows(a):
    '''
    http://stackoverflow.com/questions/8560440/removing-duplicate-columns-and-rows-from-a-numpy-2d-array
    '''
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def column_products(X, Y):
    '''
    construct a matrix whose columns are the hadamard products of all unique pairs of columns in X and Y. 
    this implementation could probably be speeded up since it's not vectorized. 
    params:
        X, an np.array() of dimension 2
        Y, an np.array() of dimension 2 with the same number of columns as X
    returns:
        a matrix whose columns are the hadamard products of all unique pairs of columns in X and Y. Duplicate columns are discarded.
    '''
    a = np.array([col1 * col2 for (col1, col2) in product(X.T, Y.T)]).T
    return unique_rows(a.T).T

def polynomial_basis(X, m):
    '''
    construct a polynomial basis of degree m from a data set X. 
    params:
        X, an np.array() of dimension 2 in which each column represents a feature and each row an observation. 
    returns:
        phi, an np.array() of dimension 2 consisting of all degree m monomials in X, as well as a column of ones. 
    '''
    X = pad(X)
    return reduce(column_products, list(X for i in range(m)))
