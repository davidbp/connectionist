from timeit import default_timer as timer
import numpy as np
from numpy import dot as npdot
from numpy import outer as np_outer
import time
import matplotlib.pyplot as plt
import numexpr as ne
from numexpr import evaluate 
import sys
# Author: David Buchaca Prats
import numba
from numba import jit, autojit

def sig(v):
    return ne.evaluate("1/(1 + exp(-v))")

def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]

class RBM:

    def __init__(self, 
                 visible_dim,
                 hidden_dim, 
                 seed=42,
                 mu=0, 
                 sigma=0.3,
                 monitor_time=True):

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

        self.num_epochs_trained = 0
        self.lr = 0
        self.monitor_time = monitor_time

        
    def plot_weight(self, 
                    hidden_unit_j, 
                    min_max_scale = True,
                    min_ = False,
                    max_ = False):
    
        plt.figure(figsize=(4, 4))

        if type(min_) == bool:
            min_ = self.W.T[hidden_unit_j].min()
        if type(max_) == bool:
            max_ = self.W.T[hidden_unit_j].max()

        plt.imshow(self.W.T[hidden_unit_j].reshape((28, 28)), cmap= plt.get_cmap('gray'), vmin= min_, vmax = max_)
        plt.xticks(())
        plt.yticks(())

    def plot_weights(self, 
                     min_max_scale = True, 
                     min_ = None, 
                     max_ = False, 
                     folder = None):
                     
        plt.figure(figsize=(10, 10))

        if type(min_)== bool:
            min_ = self.W.min()
        if type(max_) == bool:
            max_ = self.W.max()

        for i, comp in enumerate(self.W.T):
            plt.subplot(15, 15, i + 1)
            if min_max_scale:
                plt.imshow(comp.reshape((28, 28)), cmap= plt.get_cmap('gray'), vmin=min_, vmax=max_)
            else:
                plt.imshow(comp.reshape((28, 28)), cmap= plt.get_cmap('gray'))

            plt.xticks(())
            plt.yticks(())
        
        if folder:
            plt.suptitle('Epoch =' + str(self.num_epochs_trained ), fontsize=20)
            plt.savefig(folder + 'epoch_' + str(self.num_epochs_trained)  + '.png', format='png')
            plt.close()

    def update_CDK(self, 
                   Xbatch, 
                   lr=0.1,
                   K=1):

        batch_size = Xbatch.shape[0]

        Delta_W = 0
        Delta_b = 0
        Delta_c = 0

        for x in Xbatch:
            xneg = x
        
            for k in range(0, K):
                hneg = sig( npdot(xneg, self.W) + self.c) > np.random.random(self.hidden_dim).astype(np.float32)
                xneg = sig( npdot(hneg, self.W.T) + self.b) > np.random.random(self.visible_dim).astype(np.float32)
        
            ehp = sig( npdot(x, self.W) + self.c )
            ehn = sig( npdot(xneg, self.W) + self.c)

            Delta_W += lr * (np_outer(x, ehp) - np_outer(xneg, ehn))
            Delta_b += lr * (x - xneg)
            Delta_c += lr * (ehp - ehn)

        self.W += Delta_W * (1. / batch_size)
        self.b += Delta_b * (1. / batch_size)
        self.c += Delta_c * (1. / batch_size)

    
    def update_vectorizedCDK(self, 
                             Xbatch,
                             lr=0.1,
                             K=1):

        batch_size = Xbatch.shape[0]
        Xneg  = Xbatch

        for k in range(0,K):
            Hneg = sig( npdot(Xneg, self.W) + self.c) > np.random.random((batch_size, self.hidden_dim)).astype(np.float32)
            Xneg = sig( npdot(Hneg, self.W.T) + self.b) > np.random.random((batch_size, self.visible_dim)).astype(np.float32)

        Ehp = sig( npdot(Xbatch, self.W) + self.c)
        Ehn = sig( npdot(Xneg, self.W) + self.c)

        Delta_W = lr * ( npdot(Xbatch.T, Ehp) -  npdot(Xneg.T, Ehn))
        Delta_b = np.sum(lr * (Xbatch - Xneg), axis=0)
        Delta_c = np.sum(lr * (Ehp - Ehn), axis=0)

        #error_epoch += np.sum(np.sum((Xbatch-Xneg)**2), axis = 0)
        
        self.W += Delta_W * (1. / batch_size)
        self.b += Delta_b * (1. / batch_size)
        self.c += Delta_c * (1. / batch_size)


    def update_weightedCDK(self,
                           Xbatch,
                           lr=0.1,
                           K=1):
        '''
        probs : 1d numpy.array
                probabilities assigned to each state of the batch
        '''
        batch_size = Xbatch.shape[0]
        Xneg  = Xbatch
        
        for k in range(0,K):
            Hneg = sig( npdot(Xneg, self.W) + self.c) > np.random.random((batch_size, self.hidden_dim)).astype(np.float32)
            Xneg = sig( npdot(Hneg, self.W.T) + self.b) > np.random.random((batch_size, self.visible_dim)).astype(np.float32)
        
        Ehp = sig( npdot(Xbatch, self.W) + self.c)
        Ehn = sig( npdot(Xneg, self.W) + self.c)
        
        Delta_W = lr * (npdot(np.dot(Xbatch.T, Diagonal) , Ehp) -  npdot(Xneg.T, Ehn))
        Delta_b = lr * npdot(Diagonal, (Xbatch - Xneg))
        Delta_c = lr * npdot(Diagonal, (Ehp - Ehn))
        #error_epoch += np.sum(np.sum((Xbatch-Xneg)**2), axis = 0)
        
        self.W += Delta_W * (1. / batch_size)
        self.b += Delta_b * (1. / batch_size)
        self.c += Delta_c * (1. / batch_size)


    def update_persistentCDK(self, 
                             Xbatch,
                             lr=0.1, 
                             K=1):

        batch_size = Xbatch.shape[0]

        Delta_W = 0
        Delta_b = 0
        Delta_c = 0

        #import pdb;pdb.set_trace()
        if self.previous_xneg is None:
            xneg = Xbatch[0]

        for x in Xbatch:
            for k in range(0, K):
                hneg = sig( npdot(xneg, self.W) + self.c) > np.random.random(self.hidden_dim).astype(np.float32)
                xneg = sig( npdot(hneg, self.W.T) + self.b) > np.random.random(self.visible_dim).astype(np.float32)
            
            self.previous_xneg = xneg

            ehp = sig( npdot(x, self.W) + self.c )
            ehn = sig( npdot(xneg, self.W) + self.c)

            Delta_W += lr * (np_outer(x, ehp) - np_outer(xneg, ehn))
            Delta_b += lr * (x - xneg)
            Delta_c += lr * (ehp - ehn)

        self.W += Delta_W * (1. / batch_size)
        self.b += Delta_b * (1. / batch_size)
        self.c += Delta_c * (1. / batch_size)


    def propup(self, visible_vector):
        '''
        This function propagates the visible units activation upwards to
        the hidden units.
        '''
        return sig(npdot(visible_vector, self.W) + self.c)

    def fit_minibatch(self,
                      Xbatch,
                      method='CDK_vectorized',
                      lr=0.2,
                      K=5):
        '''
        Update the current weights with the given method for the given Xbatch
        '''
        if method == 'CDK':
            self.update_CDK(Xbatch=Xbatch, lr=lr, K=K)
        
        elif method == 'vectorized_CDK':
            self.update_vectorizedCDK(Xbatch=Xbatch, lr=lr, K=K)

        elif method == 'persistent_CDK':
            self.update_persistentCDK(Xbatch=Xbatch, lr=lr, K=K)
        
        elif method == 'weighted_CDK':
            self.update_weightedCDK(Xbatch=Xbatch, lr=lr, K=K)
        

    def fit(self, 
            X,
            method='CDK_vectorized',
            K=1,
            lr=0.2,
            epochs=1, 
            batch_size=10, 
            plot_weights=False, 
            folder_plots=None):
        '''
        Train the RBM 
        '''
        assert batch_size >0
        assert K>0, "K value" + K + " is not valid, K must be bigger than 0"
        assert method in ["CDK", 'vectorized_CDK', 'persistent_CDK','weighted_CDK'], "method " + method + " is not valid, please choose valid method"
        self.previous_xneg = None

        t00 = time.time()
        self.lr = lr
        elements = np.array(range(X.shape[0]))

        for epoch in range(0, epochs):
            sys.stdout.write('\r')
            sys.stdout.write("epoch %d/ %d" %  (epoch+1,epochs))

            t0 = time.time()

            np.random.shuffle(elements)
            batches = list(chunks(elements, batch_size))

            for batch in batches:
                self.fit_minibatch(X[batch, :], method, lr, K)

            self.num_epochs_trained +=1

            if plot_weights:
                self.plot_weights(folder = folder_plots)

            if self.monitor_time:
                time_ep = time.time() - t0
                time_total = time.time() - t00
                sys.stdout.write("\ttime per epoch: " + "{0:.2f}".format(time_ep) + "\t total time: " + "{0:.2f}".format(time_total))
                sys.stdout.flush()

