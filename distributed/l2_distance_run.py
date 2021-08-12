#!/usr/bin/env python

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

import argparse
from benchmark import run_benchmark, add_common_args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument(
        "-r",
        "--rows",
        type=int,
        dest='R',
        default=2**20,
        help="Rows of input matrices",
    )
    parser.add_argument(
        "-c",
        "--cols",
        type=int,
        dest='C',
        default=3,
        help="Cols of input matrices",
    )
    parser.add_argument(
        "-t",
        "--time",
        dest="timing",
        action="store_true",
        help="perform timing",
    )

    args = parser.parse_args()

    if args.use == 'numpy':
        from l2_distance.l2_distance_numpy import run_l2_distance
    elif args.use == 'dask':
        from l2_distance.l2_distance_dask import run_l2_distance
    elif args.use == 'ramba':
        from l2_distance.l2_distance_ramba import run_l2_distance
    elif args.use == 'torch':
        from l2_distance.l2_distance_torch import run_l2_distance
    elif args.use == 'heat':
        from l2_distance.l2_distance_heat import run_l2_distance
    elif args.use == 'nums':
        from l2_distance.l2_distance_nums import run_l2_distance
    elif args.use == 'legate':
        from l2_distance.l2_distance_legate import run_l2_distance

    run_benchmark(
        run_l2_distance,
        args.benchmark,
        f"L2_DISTANCE,{args.use}",
        args.no_nodes,
        (args.R, args.C, args.timing)
    )
