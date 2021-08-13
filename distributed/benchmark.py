#!/usr/bin/env python

# Copyright 2021 NVIDIA Corporation
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

import math
import sys
import time

if sys.version_info > (3, 0):
    from functools import reduce

def add_common_args(parser):
    parser.add_argument(
        "-b",
        "--benchmark",
        type=int,
        default=1,
        dest="benchmark",
        help="number of times to benchmark this application (default 1 "
        "- normal execution)",
    )
    parser.add_argument(
        "-u",
        "--use",
        default='numpy',
        choices=['numpy', 'dask', 'ramba', 'torch', 'heat', 'nums',],
        dest="use",
        help="use given numpy implementation",
    )
    parser.add_argument(
        "--no-nodes",
        default='0',
        type=int,
        help="Number of nodes this is running on (used for reporting only)",
    )
    parser.add_argument(
        "--worker",
        default=False,
        action="store_true",
        help="Is running process a worker process?",
    )


# A helper method for benchmarking applications
def run_benchmark(f, bargs, name, args):
    samples = bargs.benchmark
    nnodes = bargs.no_nodes
    if samples > 1:
        results = [f(*args) for s in range(samples)]
        if bargs.worker:
            return
        # Remove the largest and the smallest ones
        if samples >= 3:
            results.remove(max(results))
        if samples >= 2:
            results.remove(min(results))
        mean = sum(results) / len(results)
        variance = sum(map(lambda x: (x - mean) ** 2, results)) / len(results)
        stddev = math.sqrt(variance)
        print("-----------------------------------------------")
        print("BENCHMARK RESULTS: " + name)
        print("Total Samples: " + str(samples))
        print("Average Time: " + str(mean) + " ms")
        print("Variance: " + str(variance) + " ms")
        print("Stddev: " + str(stddev) + " ms")
        print(
            "All Results: "
            + reduce(lambda x, y: x + y, map(lambda x: str(x) + ", ", results))
        )
        print("-----------------------------------------------")

        ltime = time.localtime(time.time())
        fname = "bench-results.csv"
        with open(fname, "a") as csv:
            csv.write(','.join([
                str(ltime.tm_year),
                str(ltime.tm_mon),
                str(ltime.tm_mday),
                str(ltime.tm_hour),
                str(ltime.tm_min),
                str(ltime.tm_sec),
                name,
                str(nnodes),
                str(samples),
                str(mean),
                str(variance),
                str(stddev),
                *(str(x) for x in args),]))
            csv.write('\n')
    else:
        # Just run the application like normal
        f(*args)
