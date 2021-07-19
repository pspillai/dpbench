#!/usr/bin/env python

from __future__ import print_function

import argparse
from benchmark import run_benchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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

    run_benchmark(
        run_l2_distance,
        args.benchmark,
        f"L2_DISTANCE,{args.use}",
        (args.R, args.C,)
    )
