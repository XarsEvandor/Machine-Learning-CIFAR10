# ......................................................................................
# MIT License

# Copyright (c) 2020-2022 Pantelis I. Kaplanoglou

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ......................................................................................


import os
import random
import numpy as np
import tensorflow as tf


# --------------------------------------------------------------------------------------
# We are seeding the number generators to get some amount of determinism for the whole ML training process. 
# This is not ensuring 100% deterministic reproduction of an experiment in GPUs
def RandomSeed(p_nSeed=2022):
    random.seed(p_nSeed)
    os.environ['PYTHONHASHSEED'] = str(p_nSeed)
    np.random.seed(p_nSeed)    
    tf.compat.v1.reset_default_graph()
    
    #tf.compat.v1.set_random_seed(cls.SEED)
    tf.random.set_seed(p_nSeed)
    print("Random seed set to %d" % p_nSeed)
# --------------------------------------------------------------------------------------
