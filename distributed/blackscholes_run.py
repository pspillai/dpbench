#!/usr/bin/env python

from __future__ import print_function

import argparse
from benchmark import run_benchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--nopt",
        type=int,
        dest='N',
        default=2**20,
        help="number of options",
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
        from blackscholes.blackscholes_numpy import run_blackscholes
    elif args.use == 'dask':
        from blackscholes.blackscholes_dask import run_blackscholes
    elif args.use == 'ramba':
        from blackscholes.blackscholes_ramba import run_blackscholes
    elif args.use == 'torch':
        from blackscholes.blackscholes_torch import run_blackscholes
    elif args.use == 'heat':
        from blackscholes.blackscholes_heat import run_blackscholes
    elif args.use == 'nums':
        from blackscholes.blackscholes_nums import run_blackscholes

    run_benchmark(
        run_blackscholes,
        args.benchmark,
        f"BLACKSCHOLES,{args.use}",
        (args.N,)
    )
