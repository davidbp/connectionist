
import theano
import numpy as np
import time
from theano.tensor.shared_randomstreams import RandomStreams
from theano import tensor as T
from sklearn.preprocessing import OneHotEncoder
srng = RandomStreams(seed=100)


 # https://gist.github.com/SercanKaraoglu/c39d472497a13f32c592
 
def softmax(X):
    # Use the following line (for numerical stability reasons)
    e_x = T.exp(X -X.max(axis=1).dimshuffle(0,'x'))
    # instead of e_x = T.exp(X)
    return e_x/e_x.sum(axis=1).dimshuffle(0,'x')
 
def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)
 
def relu(X):
    return T.maximum(X, 0.)

def identity(X):
    return X


class MLPRegression(object):
    """
    Class implementing a feedforward neuralnetwork (multilayer perceptron).
    This class assumes the objective of the model is to perform Regresison
    """
    
    def __init__(self, dims, activations, 
                             learning_rate=0.01, 
                             seed=1,
                             max_iter=200,
                             momentum=0.9,
                             batch_size=200,
                             dropout_prob = -1):

        # Save all hyperparameters
        self.learning_rate = learning_rate
        self.dims = dims                                       
        self.seed = seed                                       # must be an integer
        self.randomstate = np.random.RandomState(self.seed)    # RandomState numpy 
        self.max_iter = max_iter
        self.momentum = momentum
        self.batch_size = batch_size
        self.dropout_prob = dropout_prob

        self.sym_Xbatch = T.matrix("sym_Xbatch")
        self.sym_Ybatch = T.matrix("sym_Ybatch")
 
        # Initialize activation functions
        self.activations = self._init_activations(activations)

        # initialize the weights and biases of the mlp
        self._init_all_weights(dims)

         # Define how to find the output of the model given the parameters of the model during training
        self.output_for_sym_Xbatch_dropout = self.output_given_input_train(self.sym_Xbatch, self.W, self.b)

         # Define how to find the output of the model given the parameters of the model during training
        self.output_for_sym_Xbatch = self.output_given_input_evaluation(self.sym_Xbatch, self.W, self.b)

        # Define a cost function, must be defined using
        self.sym_cost = T.mean((self.output_for_sym_Xbatch_dropout - self.sym_Ybatch)**2)

        # Define a method for updating the parameters of the network
        self.sym_updates = self._updates_sgd(self.sym_cost, self.params)
        
        # Define a supervised training procedurea
        self.tfunc_fit_mini_batch = theano.function(inputs=[self.sym_Xbatch, self.sym_Ybatch], 
                                                    outputs=self.sym_cost,
                                                    updates=self.sym_updates,
                                                    allow_input_downcast = True)
        
        # Define a function to the the predicted class for a set of inputs
        self.tfunc_predict = theano.function(inputs=[self.sym_Xbatch],
                                             outputs=self.output_for_sym_Xbatch, 
                                             allow_input_downcast = True)

        # Define a function to the the predicted class for a set of inputs
        #self.tfunc_predict_evaluation = theano.function(inputs=[self.sym_Xbatch],
        #                                         outputs=self.output_for_sym_Xbatch, 
        #                                         allow_input_downcast = True)


    def _init_activations(self, activations):
        """
        Initialize the activation functions at each layer.
        """
        implemented_activations = {"relu": relu,
                                   "softmax": softmax,
                                   "identity": identity }
        
        # Check the given activations are allowed
        for activation in activations:
            assert(activation in implemented_activations, 'One of the activations was not allowed')

        activations_initialized = []
        for activation in activations:
            activations_initialized.append(implemented_activations[activation])

        return activations_initialized

    def _init_all_weights(self, dims, verbose=0):
        """
        Initialize the weights and biases of the network.
        - Weights are initialized using a normal centered around 0.
        - Biases are initialized to zero.
        """
        self.W = []
        self.b = []

        for input_dim, output_dim in zip(dims[:-1], dims[1:]):
            self.W.append(self._init_weights((input_dim,output_dim)))
            self.b.append(theano.shared(floatX(np.zeros(output_dim))))
        
        self.params = []
        for W_k, b_k in zip(self.W,self.b):
            self.params.append(W_k)
            self.params.append(b_k)

    def _init_weights(self, shape, scale=0.2):
        """
        Initialize the weights using numbers sampled from a normal distribution.
        """
        np.random.seed(self.seed)
        return theano.shared(floatX(np.random.normal(np.zeros(shape), scale=scale)/np.sqrt(shape[0])))
 

    def dropout(self, incoming_input, layer_size, p):
        srng = theano.tensor.shared_randomstreams.RandomStreams(self.seed)
        mask = srng.binomial(n=1, p=1-p, size=list([layer_size]), dtype= theano.config.floatX)

        output = incoming_input * T.cast(mask, theano.config.floatX)
        return output # / (1 - p)

    def _updates_sgd(self, cost, params):
        """
        Method used to define a list of symbolic updates for theano
        """
        grads = theano.tensor.grad(cost=cost, wrt=params)
        updates = []
        for param,grad in zip(params, grads):
            updates.append([param, param - grad * self.learning_rate ])

        return updates
     
    def output_given_input_train(self, X, Ws, bs):
        """
        Predicts the output of the network.
        """
        n_layers = len(Ws)
        current_layer = 0

        for W, b, activation in zip(Ws, bs, self.activations):
            X = activation(T.dot(X, W) + b)
            current_layer += 1
            #import pdb;pdb.set_trace()
            if current_layer < len(Ws) and self.dropout_prob>0 :
                X = self.dropout(X, layer_size = int(W.shape[1].eval()), p = self.dropout_prob)

        return X
     

    def output_given_input_evaluation(self, X, Ws, bs):
        """
        Predicts the output of the network.
        """
        n_layers = len(Ws)
        current_layer = 0

        for W, b, activation in zip(Ws, bs, self.activations):
            if self.dropout_prob>0:
                X = activation(T.dot(X, W) + b)*self.dropout_prob
            else:
                X = activation(T.dot(X, W) + b)

        return X
     

    def partial_fit(self, X, y):
        """
        Fit the model for a given minibatch
        """
        # Ensure y has ndim=2 (targets are passed as column vector)
        if y.ndim == 1:
            y = y.reshape((-1, 1))
        
        cost_minibatch = self.tfunc_fit_mini_batch(X, y)
        return cost_minibatch

    def fit(self, X, y, n_epochs = 100):
        """
        Fit the MLP.
        For each epoch and for each minibatch change the weights in the model.
        The function that changes the weights is the partial_fit function which
        calls self.tfunc_fit_mini_batch
        """

        # Get number of instances and number of features
        n_samples, n_features = X.shape

        # Ensure y has ndim=2 
        if y.ndim == 1:
            print("\ny has been reshaped because it had shape:", y.shape)
            y = y.reshape((-1, 1))

        y = OneHotEncoder(sparse=False).fit_transform(y)

        self.n_outputs_ = y.shape[1]

        #layer_units = ([n_features] + hidden_layer_sizes + [self.n_outputs_])
        for epoch in range(n_epochs):
            for i in range(0, n_samples - self.batch_size, self.batch_size):
                # WARNING: We can do this without slicing arrays
                # we can do it in the theano way passing only indicies
                self.partial_fit(X[i: i + self.batch_size], y[i: i + self.batch_size])

    def predict(self, X):
        """
        Returns the predicted target for each row in X.
        """
        return self.tfunc_predict(X)

    def compute_sym_cost(self, X, Y):
        """
        Returns the cost for a given set of data X,Y.
        """
        yhat_batch =  self.output_given_input_evaluation(X, self.W, self.b)
        return T.mean((yhat_batch - Y)**2)

    def compute_cost(self, X, Y):
        """º
        Returns the cost for a given set of data X,Y.
        """
        return self.compute_sym_cost(X, Y).eval()

