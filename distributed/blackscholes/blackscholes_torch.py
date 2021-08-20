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

import datetime
import torch as np
from torch import erf

def black_scholes( nopt, price, strike, t, rate, vol, call, put ):
	mr = -rate
	sig_sig_two = vol * vol * 2

	P = price
	S = strike
	T = t

	a = np.log(P / S)
	b = T * mr

	z = T * sig_sig_two
	c = 0.25 * z
	y = 1./np.sqrt(z)

	w1 = (a - b + c) * y
	w2 = (a - b - c) * y

	d1 = 0.5 + 0.5 * erf(w1)
	d2 = 0.5 + 0.5 * erf(w2)

	Se = np.exp(b) * S

	r =  P * d1 - Se * d2
	call[:] = r  # temporary `r` is necessary for faster `put` computation
	put[:] = r - P + Se

def initialize(nopt):
        np.manual_seed(7777777)
        S0L = 10.0
        S0H = 50.0
        XL = 10.0
        XH = 50.0
        TL = 1.0
        TH = 2.0
    
        return (np.DoubleTensor(nopt).uniform_(S0L, S0H),
                np.DoubleTensor(nopt).uniform_(XL, XH),
                np.DoubleTensor(nopt).uniform_(TL, TH),
                np.zeros(nopt, dtype=np.float64),
                -np.ones(nopt, dtype=np.float64))
        
def run_blackscholes(N, timing):
        RISK_FREE = 0.1
        VOLATILITY = 0.2
        
        start = datetime.datetime.now()
        price, strike, t, call, put = initialize(N)
        black_scholes(N, price, strike, t, RISK_FREE, VOLATILITY, call, put)
        delta = datetime.datetime.now() - start
        total = delta.total_seconds() * 1000.0
        if timing:
                print(f"Elapsed Time: {total} ms")
        return total
