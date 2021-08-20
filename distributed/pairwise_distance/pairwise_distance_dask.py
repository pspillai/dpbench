# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import datetime
import dask.array as np
from dask.distributed import Client
from os import getenv
import numpy


def pairwise_distance(X1, X2):
    # Computing the first two terms (X1^2 and X2^2) of the Euclidean distance equation
    x1 = np.sum(np.square(X1), axis=1)
    x2 = np.sum(np.square(X2), axis=1)

    #Compute third term in equation
    D = -2 * np.dot(X1, X2.T)
    x3 = x1.reshape(x1.size,1)
    D = D + x3
    D = D + x2

    #Compute square root for euclidean distance
    D = np.sqrt(D)
    return np.stack(D).compute()

def initialize(size, dims, nt):
    print(nt)
    np.random.seed(7777777)
    return (np.random.random_sample((size, dims)),#, chunks=(size//nt, dims)),
        np.random.random_sample((size, dims)))#, chunks=(size//nt, dims)))

def run_pairwise_distance(size, dims, timing):
    client = Client(scheduler_file=getenv("DASK_CFG"))
    nt = numpy.sum([x for x in client.nthreads().values()])
    start = datetime.datetime.now()
    X1, X2 = initialize(size, dims, nt)
    D = pairwise_distance(X1, X2)
    delta = datetime.datetime.now() - start
    total = delta.total_seconds() * 1000.0
    if timing:
        print(f"Elapsed Time: {total} ms")
    print(D[0])
    return total
