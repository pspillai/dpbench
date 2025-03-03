# *****************************************************************************
# Copyright (c) 2020, Intel Corporation All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#     Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************

import numpy as np
import numba
import base_knn
from dpbench_decorators import jit


@jit(nopython=True)
def euclidean_dist(x1, x2):
    distance = 0

    for i in range(len(x1)):
        diff = x1[i] - x2[i]
        distance += diff * diff

    result = distance ** 0.5
    # result = np.sqrt(distance)
    return result


@jit(nopython=True)
def push_queue(queue_neighbors, new_distance, index=4):
    while index > 0 and new_distance[0] < queue_neighbors[index - 1][0]:
        queue_neighbors[index] = queue_neighbors[index - 1]
        index = index - 1
        queue_neighbors[index] = new_distance


@jit(nopython=True)
def sort_queue(queue_neighbors):
    for i in range(len(queue_neighbors)):
        push_queue(queue_neighbors, queue_neighbors[i], i)


@jit(nopython=True)
def simple_vote(neighbors, classes_num=3):
    votes_to_classes = np.zeros(classes_num)

    for neighbor in neighbors:
        votes_to_classes[neighbor[1]] += 1

    max_ind = 0
    max_value = 0

    for i in range(classes_num):
        if votes_to_classes[i] > max_value:
            max_value = votes_to_classes[i]
            max_ind = i

    return max_ind


@jit(nopython=True, parallel=True)
def run_knn(train, train_labels, test, k=5, classes_num=3):
    test_size = len(test)
    train_size = len(train)

    predictions = np.empty(test_size)

    for i in numba.prange(test_size):
        queue_neighbors = []

        for j in range(k):
            dist = euclidean_dist(train[j], test[i])
            # queue_neighbors[j] = (dist, train_labels[j])
            queue_neighbors.append((dist, train_labels[j]))

        sort_queue(queue_neighbors)

        for j in range(k, train_size):
            dist = euclidean_dist(train[j], test[i])
            new_neighbor = (dist, train_labels[j])

            if dist < queue_neighbors[k - 1][0]:
                queue_neighbors[k - 1] = new_neighbor
                push_queue(queue_neighbors, new_neighbor)

        predictions[i] = simple_vote(queue_neighbors, classes_num)

    return predictions


base_knn.run("K-Nearest-Neighbors Numba", run_knn)
