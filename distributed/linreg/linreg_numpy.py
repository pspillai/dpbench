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

import numpy as np

def initialize(N, F, T):
    # We'll generate some random inputs here
    # since we don't need it to converge
    np.random.seed(1)
    x = np.random.randn(N, F).astype(T)
    y = np.random.random(N).astype(T)
    print(x.shape, y.shape)
    return x, y


def linear_regression(
    T, features, target, steps, learning_rate, sample, add_intercept=False
):
    if add_intercept:
        intercept = np.ones((features.shape[0], 1), dtype=T)
        features = np.hstack((intercept, features))

    weights = np.zeros(features.shape[1], dtype=T)

    for step in range(steps):
        scores = np.dot(features, weights)
        error = scores - target
        gradient = -(1.0 / len(features)) * error.dot(features)
        weights += learning_rate * gradient

        if step % sample == 0:
            print(
                "Error of step "
                + str(step)
                + ": "
                + str(np.sum(np.power(error, 2)))
            )

    return weights


def run_linear_regression(N, F, T, I, S, B):  # noqa: E741
    print("Running linear regression...")
    print("Number of data points: " + str(N) + "K")
    print("Number of features: " + str(F))
    print("Number of iterations: " + str(I))
    if T == 16:
        T = np.float16
    elif T == 32:
        T = np.float32
    elif T == 64:
        T = np.float64
    start = datetime.datetime.now()
    features, target = initialize(N * 1000, F, T)
    weights = linear_regression(T, features, target, I, 1e-5, S, B)
    # Check the weights for NaNs to synchronize before stopping timing
    assert not math.isnan(np.sum(weights))
    stop = datetime.datetime.now()
    delta = stop - start
    total = delta.total_seconds() * 1000.0
    print("Elapsed Time: " + str(total) + " ms")
    return total
