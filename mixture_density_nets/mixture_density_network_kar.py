import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_array, check_X_y, column_or_1d
from sklearn.utils.extmath import safe_sparse_dot
import math 

def softmax(x):
    # softmaxes the columns of x
    #z = x - np.max(x, axis=0, keepdims=True) # for safety
    e = np.exp(x)
    en = e / np.sum(e, axis=0, keepdims=True)
    return en

class MDNRegressor(BaseEstimator, RegressorMixin):

    """
    Mixture density network regression. This version assumes
        - A single layer of hidden units.
        - Target variable to be 1-dimensional
    
    hidden_layer_sizes : tuple, length = n_layers - 2, default (100,)
 
       The ith element represents the number of neurons in the ith
       hidden layer.
       
    activation: {'tanh'}
    
    shuffle : bool, optional, default True
        Whether to shuffle samples in each iteration. 
        
    """
    
    def __init__(self,
                 hidden_layer_size,
                 n_components=5,
                 activation="tanh",
                 batch_size="auto",
                 shuffle=True,
                 n_epochs=200000):
        
        self.hidden_layer_size = hidden_layer_size
        self.n_components = n_components
        self.activation = activation
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_epochs = n_epochs

        #Initialize model
        self._initialize_in_fit(n_features=1,
                                n_hidden=hidden_layer_size,
                                n_outputs= 1,
                                n_components=self.n_components)

    def _initialize_in_fit(self, 
                           n_features,
                           n_hidden, 
                           n_outputs, 
                           n_components):
        """
        Initialize the model weights and biases
        """
        scaling_factor = 0.1
        
        # n_outputs = y.shape[1]
        self.n_outputs_ = n_outputs
        self.loss_per_epoch = []
        hidden_size = n_hidden
        input_size = n_features
        K = n_components

        # Initialize coefficient and intercept layers
        m = {}
        m['Wxh'] = np.random.randn(hidden_size, input_size) * 0.1 # input to hidden
        m['Whu'] = np.random.randn(K, hidden_size) * 0.1 # hidden to means
        m['Whs'] = np.random.randn(K, hidden_size) * 0.1 # hidden to log standard deviations
        m['Whp'] = np.random.randn(K, hidden_size) * 0.1 # hidden to mixing coefficients (cluster priors)
        m['bxh'] = np.random.randn(hidden_size, 1) * 0.01
        m['bhu'] = np.random.randn(K, 1) * 0.01
        m['bhs'] = np.random.randn(K, 1) * 0.01
        m['bhp'] = np.random.randn(K, 1) * 0.01

        print("hidden size:", hidden_size)
        print("input size:", input_size)
        print("K (num components):", K)


        self.m = m
    
    def predict_statistics(self, X):
        """
        For each of the K components predicts
            - the expected value (mean of the Gaussian) for a given x
            - the variance of the prediction (variance of the Gaussian)
            - the weight or coefficient of the component
        """
        # compute hidden activation
        return 
        
        
    def _validate_hyperparameters(self):
        """
        Ensures hyperparameters are set correctly
        """
        if not isinstance(self.shuffle, bool):
            raise ValueError("shuffle must be either True or False, got %s." %
                             self.shuffle)       
    

    def _validate_input(self, X, y, incremental):
    
        if y.ndim == 2 and y.shape[1] == 1:
            y = column_or_1d(y, warn=True)
        return X, y
    

    def _forward_pass(self, X):
        activations = [X]


    def _compute_loss(self, x, y):
        # data in X are columns
        
        m = self.m
        
        # forward pass
        h = np.tanh(np.dot(m['Wxh'], x) + m['bxh'])
        
        # predict mean
        mu = np.dot(m['Whu'], h) + m['bhu']
        
        # predict log variance
        logsig = np.dot(m['Whs'], h) + m['bhs']
        sig = np.exp(logsig)
        
        # predict mixture priors
        piu = np.dot(m['Whp'], h) + m['bhp'] # unnormalized pi
        pi = softmax(piu)
        
        # compute the loss: mean negative data log likelihood
        k,n = mu.shape # number of mixture components
        ps = np.exp(-((y - mu)**2)/(2*sig**2))/(sig*np.sqrt(2*math.pi))
        pin = ps * pi
        lp = -np.log(np.sum(pin, axis=0, keepdims=True))
        loss = np.sum(lp)/n
               
        stats = {}

        stats["loss"] = loss
        stats["elementwise_logprop"] = lp

        return stats

    def predict(self, x):

        # ferform forward pass and get only a point estimate

        # forward pass
        h = np.tanh(np.dot(self.m['Wxh'], x) + self.m['bxh'])
        
        # predict mean
        mu = np.dot(self.m['Whu'], h) + self.m['bhu']
        return mu.T


    def predict_best_estimate(self, x):
        mu, _,  pi = self.predict_distribution(x)

        preds = []
        for k in range(len(mu)):
            chosen_mode = np.argmax(pi[k])
            chosen_mean = means[k][chosen_mode]
            preds.append(chosen_mean)

        best_mus = np.array(best_mus)
        return best_mus


    def predict_distribution(self, x):
        # forward pass
        h = np.tanh(np.dot(self.m['Wxh'], x) + self.m['bxh'])
        
        # predict mean
        mu = np.dot(self.m['Whu'], h) + self.m['bhu']
        
        # predict log variance
        logsig = np.dot(self.m['Whs'], h) + self.m['bhs']
        sig = np.exp(logsig)
        
        # predict mixture priors
        piu = np.dot(self.m['Whp'], h) + self.m['bhp'] # unnormalized pi
        pi = softmax(piu)
        return (mu.T, sig.T, pi.T)

    def _compute_grads(self, x, y):
        
        # data in X are columns
        
        m = self.m

        # forward pass
        h = np.tanh(np.dot(m['Wxh'], x) + m['bxh'])
        
        # predict mean
        mu = np.dot(m['Whu'], h) + m['bhu']
        
        # predict log variance
        logsig = np.dot(m['Whs'], h) + m['bhs']
        sig = np.exp(logsig)
        
        # predict mixture priors
        piu = np.dot(m['Whp'], h) + m['bhp'] # unnormalized pi
        pi = softmax(piu)
        
        # compute the loss: mean negative data log likelihood
        k,n = mu.shape # number of mixture components
        ps = np.exp(-((y - mu)**2)/(2*sig**2))/(sig*np.sqrt(2*math.pi))
        pin = ps * pi
        lp = -np.log(np.sum(pin, axis=0, keepdims=True))
        loss = np.sum(lp)/n
        
        # compute the gradients on nn outputsmi
        grad = {}
        gammas = pin / np.sum(pin, axis=0, keepdims = True)
        dmu = gammas * ((mu - y)/sig**2) /n
        dlogsig = gammas * (1.0 - (y-mu)**2/(sig**2)) /n
        dpiu = (pi - gammas) /n
        
        # backprop to decoder matrices
        grad['bhu'] = np.sum(dmu, axis=1, keepdims=True)
        grad['bhs'] = np.sum(dlogsig, axis=1, keepdims=True)
        grad['bhp'] = np.sum(dpiu, axis=1, keepdims=True)
        grad['Whu'] = np.dot(dmu, h.T)
        grad['Whs'] = np.dot(dlogsig, h.T)
        grad['Whp'] = np.dot(dpiu, h.T)
        
        # backprop to h
        dh = np.dot(m['Whu'].T, dmu) + np.dot(m['Whs'].T, dlogsig) + np.dot(m['Whp'].T, dpiu)
        
        # backprop tanh
        dh = (1.0-h**2)*dh
        
        # backprop input to hidden
        grad['bxh'] = np.sum(dh, axis=1, keepdims=True)
        grad['Wxh'] = np.dot(dh, x.T)
        
        # misc stats
        stats = {}
        stats['lp'] = lp
        return loss, grad, stats
        
        
        
    def _fit(self, X, y, n_epochs = 20000, n_epochs_to_print=1000):
        """
        Train the model
        """
        ###########
        # Prepare #
        ###########
        
        # Do stuff here
        hidden_layer_size = self.hidden_layer_size
        n_epochs = self.n_epochs
      

        self.n_outputs_ = 1
        n_features = X.shape[0]
        
        # Initialize model
        self._initialize_in_fit(n_features,
                                hidden_layer_size,
                                self.n_outputs_,
                                self.n_components)
        
        ###########
        # Train   #
        ###########
        learning_rate = 0.01
        
        ### Initialize adagrad
        mem = {}
        for k in self.m.keys(): 
            mem[k] = np.zeros_like(self.m[k]) 


        print("X shape:", X.shape)
        print("y  shape:", y.shape)

        
        #nb = n_samples #full batch
        #xbatch = np.reshape(X[:nb], (1,nb))
        #ybatch = np.reshape(Y[:nb], (1,nb))
        
        for epoch in range(n_epochs):
            
            loss, grads , _ = self._compute_grads(X, y)

            if epoch % n_epochs_to_print == 0:
                print ("epoch: ", epoch, "loss: ", loss)

            for k,v in grads.items():
                mem[k] += grads[k]**2
                self.m[k] += -learning_rate * grads[k]  / np.sqrt(mem[k] + 1e-6)

            
    def fit(self, X, y):
        return self._fit(X, y)