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

def _tuple(x):
    return tuple(map(int, x.split(",")))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument(
        "-d",
        "--dim",
        type=_tuple,
        default=(1000,1000),
        help="output image dimension (x,y)",
    )
    parser.add_argument(
        "-r",
        "--region",
        type=_tuple,
        default=(-2.25, -1.25, 0.75, 1.25),
        help="region to compute (xmin, ymin, xmax, ymax)",
    )
    parser.add_argument(
        "-i",
        "--itermax",
        type=int,
        default=200,
        help="maximum number of iterations",
    )
    parser.add_argument(
        "-z",
        "--horizon",
        type=float,
        default=2.0,
        help="horizon",
    )
    args = parser.parse_args()

    if args.use == 'numpy':
        from mandelbrot.mandelbrot_numpy import run_mandelbrot
    elif args.use == 'dask':
        from mandelbrot.mandelbrot_dask import run_mandelbrot
    elif args.use == 'ramba':
        from mandelbrot.mandelbrot_ramba import run_mandelbrot
    elif args.use == 'torch':
        from mandelbrot.mandelbrot_torch import run_mandelbrot
    elif args.use == 'heat':
        from mandelbrot.mandelbrot_heat import run_mandelbrot
    elif args.use == 'nums':
        from mandelbrot.mandelbrot_nums import run_mandelbrot

    run_benchmark(
        run_mandelbrot,
        args,
        f"MANDELBROT,{args.use}",
        (*args.region, *args.dim, args.itermax, args.horizon)
    )
