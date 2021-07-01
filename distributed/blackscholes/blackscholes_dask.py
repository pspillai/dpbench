# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import datetime
import dask.array as np
from scipy.special import erf

def black_scholes ( nopt, price, strike, t, rate, vol, schd=None):
	mr = -rate
	sig_sig_two = vol * vol * 2

	P = price.persist()
	S = strike.persist()
	T = t.persist()

	a = np.log(P / S)
	b = T * mr

	z = T * sig_sig_two
	c = 0.25 * z
	y = np.map_blocks( lambda x: np.true_divide(1.0, np.sqrt(x)) , z)

	w1 = (a - b + c) * y
	w2 = (a - b - c) * y

	d1 = 0.5 + 0.5 * np.map_blocks(erf, w1)
	d2 = 0.5 + 0.5 * np.map_blocks(erf, w2)

	Se = np.exp(b) * S

	call = P * d1 - Se * d2
	put = call - P + Se

	return np.compute( np.stack((put, call)), scheduler=schd )

def initialize(nopt):
    S0L = 10.0
    S0H = 50.0
    XL = 10.0
    XH = 50.0
    TL = 1.0
    TH = 2.0
    
    return (
        np.random.uniform(S0L, S0H, nopt),
        np.random.uniform(XL, XH, nopt),
        np.random.uniform(TL, TH, nopt)
    )
        
def run_blackscholes(N):
    RISK_FREE = 0.1
    VOLATILITY = 0.2
    
    start = datetime.datetime.now()
    price, strike, t = initialize(N)
    black_scholes(N, price, strike, t, RISK_FREE, VOLATILITY)
    delta = datetime.datetime.now() - start
    total = delta.total_seconds() * 1000.0
    print(f"Elapsed Time: {total} ms")
    return total
