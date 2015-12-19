from timeit import default_timer as timer
import numpy as np
from numpy import dot as npdot
import time
import matplotlib.pyplot as plt
import numexpr as ne
import sys
# Author: David Buchaca Prats


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


class RBM:

    def __init__(self, visible_dim, hidden_dim, seed=42, mu=0, sigma=0.3, monitor_time=True):
        np.random.seed(seed)
        self.previous_xneg = None
        W = np.random.normal(mu, sigma, [ visible_dim, hidden_dim])
        self.W = np.array(W, dtype='float32')

        np.random.seed(seed)
        b = np.random.normal(mu, sigma, [visible_dim ])
        self.b = np.array(b, dtype='float32')

        np.random.seed(seed)
        c = np.random.normal(mu, sigma, [hidden_dim])
        self.c = np.array(c, dtype='float32')

        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim

        self.lr = 0
        self.monitor_time = monitor_time

        
    def plot_weights(self):
        plt.figure(figsize=(15.2, 15))

        for i, comp in enumerate(self.W.T):
            plt.subplot(15, 15, i + 1)
            plt.imshow(comp.reshape((28, 28)), cmap=plt.cm.gray_r, interpolation='nearest')
            plt.xticks(())
            plt.yticks(())

    def update_params_cdk(self, Xbatch, lr=0.1, K=1):

        batch_size = Xbatch.shape[0]

        Delta_W = 0
        Delta_b = 0
        Delta_c = 0

        for x in Xbatch:
            xneg = x
        
            for k in range(0, k):
                hneg = sig( npdot(xneg, self.W) + self.c) > np.random.random(self.hidden_dim).astype(np.float32)
                xneg = sig( npdot(hneg, self.W.T) + self.b) > np.random.random(self.visible_dim).astype(np.float32)
        
            ehp = sig( npdot(x, self.W) + self.c )
            ehn = sig( npdot(xneg, self.W) + self.c)

            Delta_W += lr * (np.outer(x, ehp) - np.outer(xneg, ehn))
            Delta_b += lr * (x - xneg)
            Delta_c += lr * (ehp - ehn)

        self.W += Delta_W * (1. / batch_size)
        self.b += Delta_b * (1. / batch_size)
        self.c += Delta_c * (1. / batch_size)


    def update_params_cdk_vectorized(self, Xbatch, lr=0.1, K=1):

        batch_size = Xbatch.shape[0]
        Xneg  = Xbatch

        for k in range(0,K):
            Hneg = sig( npdot(Xneg , self.W) + self.c) > np.random.random((batch_size, self.hidden_dim)).astype(np.float32)
            Xneg = sig( npdot(Hneg, self.W.T) + self.b) > np.random.random((batch_size, self.visible_dim)).astype(np.float32)

        Ehp = sig( npdot(Xbatch, self.W) + self.c)
        Ehn = sig( npdot(Xneg, self.W) + self.c)

        Delta_W = lr * ( npdot(Xbatch.T, Ehp) -  npdot(Xneg.T, Ehn))
        Delta_b =  np.sum(lr * (Xbatch - Xneg), axis=0)
        Delta_c = np.sum(lr * (Ehp - Ehn), axis=0)
        #error_epoch += np.sum(np.sum((Xbatch-Xneg)**2), axis = 0)
        
        self.W += Delta_W * (1. / batch_size)
        self.b += Delta_b * (1. / batch_size)
        self.c += Delta_c * (1. / batch_size)


    def update_params_cdk_vectorized_ne(self, Xbatch, lr=0.1, K=1):

        batch_size = Xbatch.shape[0]
        Xneg  = Xbatch

        for k in range(0,K):
            Hneg = sig( npdot(Xneg, self.W) + self.c, numexpr=True) > np.random.random((batch_size, self.hidden_dim)).astype(np.float32)
            Xneg = sig( npdot(Hneg, self.W.T) + self.b, numexpr=True) > np.random.random((batch_size, self.visible_dim)).astype(np.float32)

        Ehp = sig( npdot(Xbatch, self.W) + self.c, numexpr=True)
        Ehn = sig( npdot(Xneg, self.W) + self.c, numexpr=True)

        Delta_W = lr * ( npdot(Xbatch.T, Ehp) -  npdot(Xneg.T, Ehn))
        Delta_b = np.sum(lr * (Xbatch - Xneg), axis=0)
        Delta_c = np.sum(lr * (Ehp - Ehn), axis=0)
        #error_epoch += np.sum(np.sum((Xbatch-Xneg)**2), axis = 0)
        
        self.W += Delta_W * (1. / batch_size)
        self.b += Delta_b * (1. / batch_size)
        self.c += Delta_c * (1. / batch_size)


    def update_params_pcd(self, Xbatch, lr=0.1, K=1):

        batch_size = Xbatch.shape[0]

        Delta_W = 0
        Delta_b = 0
        Delta_c = 0

        if self.previous_xneg is None:
            xneg = Xbatch[0]

        for x in Xbatch:
            for k in range(0, K):
                hneg = sig( npdot(xneg, self.W) + self.c) > np.random.random(self.hidden_dim).astype(np.float32)
                xneg = sig( npdot(hneg, self.W.T) + self.b) > np.random.random(self.visible_dim).astype(np.float32)
            
            self.previous_xneg = xneg

            ehp = sig( npdot(x, self.W) + self.c )
            ehn = sig( npdot(xneg, self.W) + self.c)

            Delta_W += lr * (np.outer(x, ehp) - np.outer(xneg, ehn))
            Delta_b += lr * (x - xneg)
            Delta_c += lr * (ehp - ehn)

        self.W += Delta_W * (1. / batch_size)
        self.b += Delta_b * (1. / batch_size)
        self.c += Delta_c * (1. / batch_size)


    def fit_minibatch(self, Xbatch, method='CDK_vectorized', lr=0.2, K=5):
        '''
        Update the current weights with the given method for the given Xbatch
        '''

        if method == 'CDK':
            self.update_params_cdk(Xbatch=Xbatch, lr=lr, K=K)
        
        elif method == 'CDK_vectorized':
            self.update_params_cdk_vectorized(Xbatch=Xbatch, lr=lr, K=K)
        
        elif method == 'CDK_vectorized_numexpr':
            self.update_params_cdk_vectorized_ne(Xbatch=Xbatch, lr=lr, K=K)


    def fit(self, X, method='CDK_vectorized', K=1, lr=0.2, epochs=1, batch_size=10, plot_weights=False):
        '''
        Train the RBM 
        '''

        assert batch_size >0
        assert K>0, "K value" + K + " is not valid, K must be bigger than 0"
        assert method in ["CDK", "CDK_vectorized", 'CDK_vectorized_numexpr'], "method " + method + " is not valid, please choose valid method"

        self.lr = lr
        elements = np.array(range(X.shape[0]))

        for epoch in range(0, epochs):
            sys.stdout.write('\r')
            sys.stdout.write("epoch %d/ %d" %  (epoch,epochs))
            sys.stdout.flush()

            t0 = time.time()

            np.random.shuffle(elements)
            batches = list(chunks(elements, batch_size))

            for batch in batches:
                self.fit_minibatch(X[batch, ], method, lr, K)

            if plot_weights:
                save_plot(lr, epoch, W=self.W)

            if self.monitor_time:
                sys.stdout.write("\ttime per epoch: " + str(time.time() - t0) )


