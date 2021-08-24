"""
Copyright 2021 Intel Corporation

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

# Use lbfgs/gen_Xy_data.py to generate input data

from __future__ import print_function

import argparse
from benchmark import run_benchmark, add_common_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser("\n\n!!!!!!!!!!!!!!!!!!!\nNote: use lbfgs/gen_Xy_data.py to generate input data\n!!!!!!!!!!!!!!!!!!!\n")
    add_common_args(parser)
    parser.add_argument(
        "-f",
        "--features",
        type=int,
        default=10,
        help="number of features for each input data point",
    )
    parser.add_argument(
        "-n",
        "--numrows",
        type=int,
        default=1000,
        help="number of elements in the data set",
    )
    args = parser.parse_args()

    if args.use == 'numpy':
        from lbfgs.lbfgs_numpy import run_lbfgs
    elif args.use == 'dask':
        from lbfgs.lbfgs_dask import run_lbfgs
    elif args.use == 'ramba':
        from lbfgs.lbfgs_ramba import run_lbfgs
    elif args.use == 'torch':
        from lbfgs.lbfgs_torch import run_lbfgs
    elif args.use == 'heat':
        from lbfgs.lbfgs_heat import run_lbfgs
    elif args.use == 'cw4heat':
        from lbfgs.lbfgs_cw4heat import run_lbfgs
    elif args.use == 'nums':
        from lbfgs.lbfgs_nums import run_lbfgs
    
    run_benchmark(
        run_lbfgs,
        args,
        f"LBFGS,{args.use}",
        (args.numrows, args.features),
    )
