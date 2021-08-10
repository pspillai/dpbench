# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import datetime
import nums.numpy as np
from scipy.special import erf

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

        d1 = 0.5 + 0.5 * erf(w1.get())
        d2 = 0.5 + 0.5 * erf(w2.get())

        Se = np.exp(b) * S

        r =  P * d1 - Se * d2
        call[:] = r  # temporary `r` is necessary for faster `put` computation
        put[:] = r - P + Se

        return (call.get(), put.get())

def initialize(nopt):
        S0L = 10.0
        S0H = 50.0
        XL = 10.0
        XH = 50.0
        TL = 1.0
        TH = 2.0
    
        return ((S0H-S0L)*np.random.random_sample(nopt) + S0L,
                (XH-XL)*np.random.random_sample(nopt) + XL,
                (TH-TL)*np.random.random_sample(nopt) + TL,
                np.zeros(nopt, dtype=np.float64),
                -np.ones(nopt, dtype=np.float64)
        )
        
def run_blackscholes(N):
        RISK_FREE = 0.1
        VOLATILITY = 0.2
    
        start = datetime.datetime.now()
        price, strike, t, call, put = initialize(N)
        call, put = black_scholes(N, price, strike, t, RISK_FREE, VOLATILITY, call, put)
        delta = datetime.datetime.now() - start
        total = delta.total_seconds() * 1000.0
        print(f"Elapsed Time: {total} ms")
        return total
