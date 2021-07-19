# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import datetime
import heat as np

def l2_distance(X1, X2):
        sub = X1-X2
        sq = np.square(sub)
        sum = np.sum(sq)
        d = np.sqrt(sum)
        return d

def initialize(size, dims):
        np.random.seed(7777777)
        return (np.random.random((size, dims)), np.random.random((size, dims)))
        
def run_l2_distance(size, dims):
        start = datetime.datetime.now()
        X1, X2 = initialize(size, dims)
        d = l2_distance(X1, X2)
        delta = datetime.datetime.now() - start
        total = delta.total_seconds() * 1000.0
        print(f"Elapsed Time: {total} ms")
        print (d)
        return total
