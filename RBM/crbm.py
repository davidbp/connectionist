from timeit import default_timer as timer
import numpy as np
from numpy import dot as npdot
import time
import matplotlib.pyplot as plt
import numexpr as ne
import sys


def sig(v, numexpr=False):
    if numexpr:
        return ne.evaluate( "1/(1 + exp(-v))")
    else:
        return 1/(1 + np.exp(-v))

def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


class CRBM:
    """
    Conditional Restricted Boltzmann Machine model.

    This class implements the CRBM model described in (1)



    ##### References

    (1): http://www.cs.toronto.edu/~fritz/absps/uai_crbms.pdf

    """
      def __init__(self, visible_dim, hidden_dim, seed=42, mu=0, sigma=0.3, monitor_time=True):
        np.random.seed(seed)
        self.previous_xneg = None
        W = np.random.normal(mu, sigma, [visible_dim, hidden_dim])
        self.W = np.array(W, dtype='float32')

        np.random.seed(seed)
        b = np.random.normal(mu, sigma, [visible_dim ])
        self.b = np.array(b, dtype='float32')

        np.random.seed(seed)
        c = np.random.normal(mu, sigma, [hidden_dim])
        self.c = np.array(c, dtype='float32')

        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim

        self.num_epochs_trained = 0
        self.lr = 0
        self.monitor_time = monitor_time