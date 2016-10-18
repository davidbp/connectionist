import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_array, check_X_y, column_or_1d
from sklearn.utils.extmath import safe_sparse_dot

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
        
        # Initialize coefficient and intercept layers
        self.coefs_ = {'W_1': np.random.randn(n_features, n_hidden) * scaling_factor,
                       'W_variance': np.random.randn(n_hidden, n_components) * scaling_factor,
                       'W_mean': np.random.randn(n_hidden, n_components) * scaling_factor,
                       'W_mix_coeff':np.random.randn(n_hidden, n_components) * scaling_factor}
        
        self.intercepts_ = {'b_1':  np.zeros(n_hidden, ),
                            'b_variance': np.zeros(n_components, ),
                            'b_mean': np.zeros(n_components, ),
                            'b_mix_coeff':  np.zeros(n_components, )}
    
    
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


    def _compute_loss(self, X, y):
        """
        Returns the probability
        """
        
        # If y is a row vector rewrite it as column vector
        if y.ndim == 1:
            y = y.reshape((-1, 1))            
        
        ### Forward pass ###
        act_h1 = np.tanh( np.dot(X, self.coefs_['W_1']) + self.intercepts_['b_1']  )
        
        act_means = np.dot(act_h1, self.coefs_['W_mean']) + self.intercepts_['b_mean']
        act_variances = np.exp(np.dot(act_h1,self.coefs_['W_variance']) + self.intercepts_['b_variance'])
        act_mixing_coeff = softmax(np.dot(act_h1,self.coefs_['W_mix_coeff']) + self.intercepts_['b_mix_coeff'])
        
        ###
        ### Compute Loss (- mean log-likelihood)
        ###
        n_samples, n_components = act_means.shape
        
        # prob_per_sample has shape (n_components, n_samples)
        prob_per_sample = np.exp(-((y - act_means)**2)/(2*act_variances**2))/(act_variances*np.sqrt(2*np.pi))

        pin = act_mixing_coeff * prob_per_sample
        # logprob has shape (1,n_samples)
        logprob = -np.log(np.sum(pin, axis=1, keepdims=True))
        loss = np.sum(logprob)/n_samples
        #import pdb;pdb.set_trace()
        
        stats = {}
        stats["loss"] = loss
        stats["logprob"] = logprob
        return stats


    def _compute_loss(self, X, y):
        """
        Returns the probability
        """
             
        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape((-1, 1))            

        ### Forward pass ###
        act_h1 = np.tanh( np.dot(X, self.coefs_['W_1']) + self.intercepts_['b_1']  )
        
        act_means = np.dot(act_h1, self.coefs_['W_mean']) + self.intercepts_['b_mean']
        act_variances = np.exp(np.dot(act_h1,self.coefs_['W_variance']) + self.intercepts_['b_variance'])
        act_mixing_coeff = softmax(np.dot(act_h1,self.coefs_['W_mix_coeff']) + self.intercepts_['b_mix_coeff'])
        
        ###
        ### Compute Loss (- mean log-likelihood)
        ###
        n_samples, n_components = act_means.shape
        
        # prob_per_sample has shape (n_components, n_samples)
        prob_per_sample = np.exp(-((y - act_means)**2)/(2*act_variances**2))/(act_variances*np.sqrt(2*np.pi))

        pin = act_mixing_coeff * prob_per_sample
        # logprob has shape (1,n_samples)
        logprob = -np.log(np.sum(pin, axis=1, keepdims=True))

        loss = np.sum(logprob)/n_samples
        
        stats = {}
        stats["loss"] = loss
        stats["elementwise_logprop"] = logprob

        return stats


    def _compute_grads(self, X, y):
        
        ### Forward pass ###
        act_h1 = np.tanh( np.dot(X, self.coefs_['W_1']) + self.intercepts_['b_1']  )
        act_means = np.dot(act_h1, self.coefs_['W_mean']) + self.intercepts_['b_mean']
        act_variances = np.exp(np.dot(act_h1, self.coefs_['W_variance']) + self.intercepts_['b_variance'])
        act_mixing_coeff = softmax(np.dot(act_h1, self.coefs_['W_mix_coeff']) + self.intercepts_['b_mix_coeff'])
        
        ###
        ### Compute Loss (- mean log-likelihood)
        ###
        n_samples, n_components = act_means.shape
        
        # prob_per_sample has shape (n_components, n_samples)
        prob_per_sample = np.exp(-((y - act_means)**2)/(2*act_variances**2))/(act_variances*np.sqrt(2*np.pi))
        pin = act_mixing_coeff * prob_per_sample
        
        # logprob has shape (1,n_samples)
        logprob = -np.log(np.sum(pin, axis=1, keepdims=True))
        loss = np.sum(logprob)/n_samples

        ###
        ### Gradients 
        ###
        
        ### Gradients of the loss with respect to the parameters of the output layer
        gammas = pin / np.sum(pin, axis=0, keepdims = True)
        dmu = gammas * ((act_means - y)/act_variances**2) / n_samples
        dlogsig = gammas * (1.0 - (y - act_means)**2/(act_variances**2)) / n_samples
        dpiu = (act_mixing_coeff - gammas) / n_samples
    
        grads = {}
        grads['W_mean'] = np.dot(dmu.T, act_h1).T
        grads['W_variance'] = np.dot(dlogsig.T, act_h1).T
        grads['W_mix_coeff'] = np.dot(dpiu.T, act_h1).T

        grads['b_mean'] = np.sum(dmu, axis=0)
        grads['b_variance'] = np.sum(dlogsig, axis=0)
        grads['b_mix_coeff'] = np.sum(dpiu, axis=0)

        ### Gradients of the loss with respect to the parameters of the first layer
        dh = np.dot(self.coefs_['W_mean'], dmu.T) + \
             np.dot(self.coefs_['W_variance'], dlogsig.T) +\
             np.dot(self.coefs_['W_mix_coeff'], dpiu.T)

        dh = (1.0 - act_h1**2)*dh.T
        grads['W_1'] = np.sum(dh, axis=0)       
        grads['b_1'] = np.dot(dh.T, X).flatten()
        self.loss_per_epoch.append(loss)
        
        return grads
        
        
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


        # Validate input parameters.
        self._validate_hyperparameters()
        if np.any(np.array(hidden_layer_size) <= 0):
            raise ValueError("hidden_layer_sizes must be > 0, got %s." %
                             hidden_layer_size)
            
        # Validate input
        X, y = self._validate_input(X, y, incremental=True)
        
        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape((-1, 1))            

        self.n_outputs_ = y.shape[1]
        n_features = X.shape[1]
        
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
        for k in self.coefs_.keys(): 
            mem[k] = np.zeros_like(self.coefs_[k]) 
        for k in self.intercepts_.keys(): 
            mem[k] = np.zeros_like(self.intercepts_[k])
        
        
        #nb = n_samples #full batch
        #xbatch = np.reshape(X[:nb], (1,nb))
        #ybatch = np.reshape(Y[:nb], (1,nb))
        
        for epoch in range(n_epochs):
            
            grads = self._compute_grads(X, y)
            if epoch % n_epochs_to_print == 0:
                print ("epoch: ", epoch, "loss: ", self.loss_per_epoch[-1])

            for k,v in grads.items():
                mem[k] += grads[k]**2
                
                if k in self.coefs_:
                    self.coefs_[k] += -learning_rate * grads[k] # / np.sqrt(mem[k] + 1e-6)
                else:
                    self.intercepts_[k] += -learning_rate * grads[k] #/ np.sqrt(mem[k] + 1e-6)                
        
            
    def fit(self, X, y):
        return self._fit(X, y)