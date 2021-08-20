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
        "-n",
        "--nopt",
        type=int,
        dest='N',
        default=2**20,
        help="number of options",
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
        from blackscholes.blackscholes_numpy import run_blackscholes
    elif args.use == 'dask':
        from blackscholes.blackscholes_dask import run_blackscholes
    elif args.use == 'ramba':
        from blackscholes.blackscholes_ramba import run_blackscholes
    elif args.use == 'torch':
        from blackscholes.blackscholes_torch import run_blackscholes
    elif args.use == 'heat':
        from blackscholes.blackscholes_heat import run_blackscholes
    elif args.use == 'cw4heat':
        from blackscholes.blackscholes_cw4heat import run_blackscholes
    elif args.use == 'nums':
        from blackscholes.blackscholes_nums import run_blackscholes
    elif args.use == 'legate':
        from blackscholes.blackscholes_legate import run_blackscholes

    run_benchmark(
        run_blackscholes,
        args,
        f"BLACKSCHOLES,{args.use}",
        (args.N, args.timing)
    )
