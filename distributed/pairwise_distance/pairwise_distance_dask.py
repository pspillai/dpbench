# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import datetime
import dask.array as np

def pairwise_distance( X1, X2, schd=None):
        # Computing the first two terms (X1^2 and X2^2) of the Euclidean distance equation
        x1 = np.sum(np.square(X1), axis=1)
        x2 = np.sum(np.square(X2), axis=1)

        #Comnpute third term in equation
        D = -2 * np.dot(X1, X2.T)
        x3 = x1.reshape(x1.size,1)
        D = D + x3 #x1[:,None] Not supported by Numba
        D = D + x2

        #Compute square root for euclidean distance
        D = np.sqrt(D)
        return np.compute( np.stack(D), scheduler=schd)

def initialize(size, dims):
        np.random.seed(7777777)
        return (np.random.random((size, dims)), np.random.random((size, dims)))
        
def run_pairwise_distance(size, dims):
    start = datetime.datetime.now()
    X1, X2 = initialize(size, dims)
    D = pairwise_distance(X1, X2)
    delta = datetime.datetime.now() - start
    total = delta.total_seconds() * 1000.0
    print(f"Elapsed Time: {total} ms")
    print (D)
    return total
