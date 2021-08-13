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

import argparse
from benchmark import run_benchmark, add_common_args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument(
        "-t",
        "--time",
        type=float,
        default=10.0,
        help="time",
    )
    parser.add_argument(
        "-d",
        "--delta-time",
        type=float,
        default=0.01,
        help="time-step",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        dest='N',
        default=100,
        help="number of bodies",
    )
    parser.add_argument(
        "-s",
        "--softening",
        type=float,
        default=0.1,
        help="softening",
    )
    args = parser.parse_args()

    if args.use == 'numpy':
        from nbody.nbody_numpy import run_nbody
    elif args.use == 'dask':
        from nbody.nbody_dask import run_nbody
    elif args.use == 'ramba':
        from nbody.nbody_ramba import run_nbody
    elif args.use == 'torch':
        from nbody.nbody_torch import run_nbody
    elif args.use == 'heat':
        from nbody.nbody_heat import run_nbody
    elif args.use == 'nums':
        from nbody.nbody_nums import run_nbody

    run_benchmark(
        run_nbody,
        args,
        f"NBODY,{args.use}",
        (args.N, args.time, args.delta_time, args.softening)
    )
