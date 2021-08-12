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
import heat as np

def l2_distance(X1, X2):
        sub = X1-X2
        sq = np.square(sub)
        sum = np.sum(sq)
        d = np.sqrt(sum)
        return d

def initialize(size, dims):
        np.random.seed(7777777)
        return (np.random.random_sample((size, dims)), np.random.random_sample((size, dims)))
        
def run_l2_distance(size, dims, timing):
        start = datetime.datetime.now()
        X1, X2 = initialize(size, dims)
        d = l2_distance(X1, X2)
        delta = datetime.datetime.now() - start
        total = delta.total_seconds() * 1000.0
        if timing:
                print(f"Elapsed Time: {total} ms")
        print (d)
        return total
