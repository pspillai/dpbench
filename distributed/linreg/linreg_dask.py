#!/usr/bin/env python

# Copyright 2021 NVIDIA Corporation
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

from __future__ import print_function

import datetime
import math

import dask.array as np
from dask.distributed import Client
from os import getenv
import numpy


def initialize(N, F, T, nt):
    # We'll generate some random inputs here
    # since we don't need it to converge
    np.random.seed(1)
    x = np.random.random_sample((N, F), chunks=(N//nt,F)).astype(T)
    y = np.random.random(N, chunks=(N//nt,)).astype(T)
    print(x.shape, y.shape)
    return x, y


def linear_regression(
    T, features, target, steps, learning_rate, sample, nt, add_intercept=False
):
    if add_intercept:
        intercept = np.ones((features.shape[0], 1), dtype=T)
        features = np.hstack((intercept, features))
    weights = np.zeros(features.shape[1], dtype=T)

    for step in range(steps):
        scores = np.matmul(features, weights)
        error = scores - target
        gradient = -(1.0 / len(features)) * np.matmul(error, features)
        weights += learning_rate * gradient
        if step % sample == 0:
            weights = weights.persist()
            error = error.persist()
            print(
                "Error of step "
                + str(step)
                + ": "
                + str(np.sum(np.power(error, 2)).compute())
            )

    return weights


def run_linear_regression(N, F, T, I, S, B):  # noqa: E741
    print("Running linear regression...")
    print("Number of data points: " + str(N) + "K")
    print("Number of features: " + str(F))
    print("Number of iterations: " + str(I))
    client = Client(scheduler_file=getenv("DASK_CFG"))
    nt = numpy.sum([x for x in client.nthreads().values()])
    if T == 16:
        T = numpy.float16
    elif T == 32:
        T = numpy.float32
    elif T == 64:
        T = numpy.float64
    start = datetime.datetime.now()
    features, target = initialize(N * 1000, F, T, nt)
    weights = linear_regression(T, features, target, I, 1e-5, S, nt, B).compute()
    # Check the weights for NaNs to synchronize before stopping timing
    assert not math.isnan(np.sum(weights))
    stop = datetime.datetime.now()
    delta = stop - start
    total = delta.total_seconds() * 1000.0
    print("Elapsed Time: " + str(total) + " ms")
    return total
