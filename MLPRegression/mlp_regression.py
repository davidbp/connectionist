
import theano
import numpy as np
import time
from theano.tensor.shared_randomstreams import RandomStreams
from theano import tensor as T
from sklearn.preprocessing import OneHotEncoder

srng = RandomStreams(seed=100)

# https://gist.github.com/SercanKaraoglu/c39d472497a13f32c592
# https://gist.github.com/kastnerkyle/816134462577399ee8b2#file-optimizers-py-L59
# https://github.com/lisa-lab/pylearn2/pull/1030

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

def sym_mean_squared_error(Ybatch_hat, Ybatch):
    return T.mean((Ybatch_hat - Ybatch)**2)

def sym_mean_absolute_percentage_error(Ybatch_hat, Ybatch):
    return T.mean(T.abs_((Ybatch - Ybatch_hat)/ Ybatch))


class MLPRegression(object):
    """
    Class implementing a feedforward neuralnetwork (multilayer perceptron).
    This class assumes the objective of the model is to perform Regresison
    """    
    def __init__(self, dims, activations, 
                             learning_rate=0.01, 
                             seed=1234,
                             max_iter=200,
                             batch_size=200,
                             dropout_prob = -1,
                             loss='mean_squared_error',
                             optimizer='SGD_momentum',
                             momentum=0.9,
                             random_state=1234):

        # Save all hyperparameters
        self.learning_rate = learning_rate
        self.dims = dims                                       
        self.seed = seed                                       # must be an integer
        self.randomstate = np.random.RandomState(self.seed)    # RandomState numpy 
        self.max_iter = max_iter
        self.batch_size = batch_size        
        self.dropout_prob = dropout_prob
        self.optimizer = optimizer
        self.momentum = momentum

        # Define other parameters
        self.velocity = None

        # Define lists for the train and validation curves
        self.loss_curve_ = None
        self.loss_curve_validation_ = None

        # Define symbolic minibatches for the data and the target
        self.sym_Xbatch = T.matrix("sym_Xbatch")
        self.sym_Ybatch = T.matrix("sym_Ybatch")
 
        # Initialize activation functions
        self.activations = self._init_activations(activations)

        # initialize the weights and biases of the mlp
        self._init_all_weights(dims)

        #### Symbolic computations train time (dropout makes forward propagation stochastic)

        # Define how to find the output of the model given the parameters of the model during training
        self.output_for_sym_Xbatch_traintime = self.output_given_input_traintime(self.sym_Xbatch, self.W, self.b)
        
        if loss == 'mean_squared_error':
            self.sym_cost_traintime = sym_mean_squared_error(self.output_for_sym_Xbatch_traintime, self.sym_Ybatch)
            self.cost = sym_mean_squared_error

        if loss == 'mean_absolute_percentage_error':
            self.sym_cost_traintime = sym_mean_absolute_percentage_error(self.output_for_sym_Xbatch_traintime, self.sym_Ybatch)
            self.cost = sym_mean_absolute_percentage_error

        # Define a method for updating the parameters of the network
        self.sym_updates = self.define_updates(self.sym_cost_traintime,
                                               self.params,
                                               optimizer=self.optimizer,
                                               velocity_coeff=self.momentum)
        
        # Define a supervised training procedurea
        self.tfunc_fit_mini_batch = theano.function(inputs=[self.sym_Xbatch, self.sym_Ybatch], 
                                                    outputs=self.sym_cost_traintime,
                                                    updates=self.sym_updates,
                                                    allow_input_downcast = True)

        #### Symbolic computations test time (dropout does not make predictions stochastic)
        # Define how to find the output of the model given the parameters of the model during training
        self.output_for_sym_Xbatch_testime = self.output_given_input_evaluation(self.sym_Xbatch, self.W, self.b)

        # Define a cost function, at test time
        self.sym_cost_testtime = T.mean((self.output_for_sym_Xbatch_testime - self.sym_Ybatch)**2)
        
        # Define a function to the the predicted class for a set of inputs
        self.tfunc_predict = theano.function(inputs=[self.sym_Xbatch],
                                             outputs=self.output_for_sym_Xbatch_testime, 
                                             allow_input_downcast = True)


    def _init_activations(self, activations):
        """
        Initialize the activation functions at each layer.
        """
        implemented_activations = {"relu": relu,
                                   "softmax": softmax,
                                   "identity": identity }
        
        # Check the given activations are allowed
        # for activation in activations:
        #    assert(activation in implemented_activations, 'One of the activations was not allowed')
        
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

    def dropout(self, activation_minibatch, p):
        """
        Dropout some units of the activation_minibatch (minibatch of activations of a particular layer).
        
        p: probability of dropout activations in the minibatch (sampling 0 in the binomial that generates the mask)
        THe implementation of  srng.binomial admits a p parameter that samples 1 with prob p
        Therefore probability of dropping out is 1-p
        """
        srng = theano.tensor.shared_randomstreams.RandomStreams(self.seed)
        mask = srng.binomial(n=1, p=1-p, size=activation_minibatch.shape, dtype= theano.config.floatX)
        output = activation_minibatch * T.cast(mask, theano.config.floatX)
        return output # / (1 - p)

    def define_updates(self, cost, params, optimizer ='SGD', velocity_coeff=0.8, rho=0.9, epsilon=1e-6):
        """
        Method used to define a list of symbolic updates for theano.
        
        Some optimizers must keep track of other auxiliar variables that change over time.
        For example, the velocity term in momentum needs to be changed every time
        a minibatch is been observed and the parameters of the model are changed.
        
        Delta_parameter = learning_rate * velocity
        """
        updates = []

        if optimizer == 'SGD':
            print('\tUsing SGD optimizer')
            grads = theano.tensor.grad(cost=cost, wrt=params)

            for param,grad in zip(params, grads):
                updates.append([param, param - grad * self.learning_rate ])
        
        elif optimizer == 'SGD_momentum':
            """
            Initialize (to zero) a velocity term for each parameter in the model.
            
            Update the velocity term using the previous velocity as follows: 
                 velocity = velocity * velocity_coeff - learning_rate * grad

            the velocity_coeff needs to verify  0 < velocity_coeff < 1

            Delta_parameter = velocity
            """
            print('\tUsing SGD_momentum optimizer')
            grads = theano.tensor.grad(cost=cost, wrt=params)

            for param, grad in zip(params, grads):
                velocity_param = theano.shared(param.get_value() * 0.)

                velocity_param_next = velocity_coeff * velocity_param  - self.learning_rate * grad 
                updates.append([velocity_param, velocity_param_next])
                updates.append([param, param + velocity_param_next])
                
        elif optimizer =='RMSprop':

            print('\tUsing SGD_momentum optimizer')
            grads = theano.tensor.grad(cost=cost, wrt=params)

            for param, grad in zip(params, grads):
                grad_avg = theano.shared(param.get_value() * 0.)

                grad_avg_new = rho * grad_avg + (1 - rho) * grad**2
                grad_scaling = T.sqrt(grad_avg_new + epsilon)
                grad = grad / grad_scaling

                updates.append((grad_avg, grad_avg_new))
                updates.append((param, param - self.learning_rate * grad))

        elif optimizer == 'SGD_nesterov':
            """
            WARNING > UPDATES DO NOT SEEM THE SAME AS THE ONES IN THE COMMENT
            """

            print('\tUsing SGD_nesterov optimizer')
            grads = theano.tensor.grad(cost=cost, wrt=params)

            for param, grad in zip(params, grads):
                # Not working because param_next is not from the computational graph of the cost
                # velocity_nesterov_param = theano.shared(param.get_value() * 0.)
                # param_next = param + velocity_coeff * velocity_nesterov_param
                # grad = T.grad(cost, param_next)
                # velocity_nesterov_param_next = velocity_coeff * velocity_nesterov_param - self.learning_rate * grad 
                velocity = theano.shared(param.get_value() * 0.)
                velocity_next = velocity_coeff * velocity - self.learning_rate * grad
                param_next = velocity_coeff**2 * velocity - (1 + velocity_coeff) * self.learning_rate * grad
                updates.append((velocity, velocity_next))
                updates.append((param, param + param_next))

        return updates
     
    def output_given_input_traintime(self, X, Ws, bs):
        """
        Predicts the output of the network for a minibatch at train time.
        """
        output_layer= len(Ws)
        current_layer = 0

        for W, b, activation in zip(Ws, bs, self.activations):
            X = activation(T.dot(X, W) + b)
            current_layer += 1
            if current_layer != output_layer and self.dropout_prob > 0 :
                X = self.dropout(X, p = self.dropout_prob)

        return X
     
    def output_given_input_evaluation(self, X, Ws, bs):
        """
        Predicts the output of the network for a minibatch at test time.
        """
        output_layer = len(Ws)
        current_layer  =  0

        for W, b, activation in zip(Ws, bs, self.activations):
            current_layer += 1
            if self.dropout_prob > 0 and current_layer != output_layer:
                X = activation(T.dot(X, W) + b) * (1 - self.dropout_prob)
            else:
                X = activation(T.dot(X, W) + b)

        return X
     
    def partial_fit(self, X, y):
        """
        Fit the model for a given minibatch
        """
        # Ensure y has ndim=2 (targets are passed as column vector)
        if y.ndim == 1:
            print("\nWarning: y has been reshaped because it had shape:", y.shape)
            y = y.reshape((-1, 1))
        
        cost_minibatch = self.tfunc_fit_mini_batch(X, y)
        return cost_minibatch

    def fit(self, X, y, X_val=None, y_val=None, n_epochs = 100):
        """
        Fit the MLP.
        For each epoch and for each minibatch change the weights in the model.

        The function that changes the weights is the partial_fit function which
        calls self.tfunc_fit_mini_batch
        """

        n_samples, n_features = X.shape
        np.random.RandomState(self.seed)
        permutation = np.random.permutation(n_samples)
        
        if not self.loss_curve_:
            self.loss_curve_ = []
            self.loss_curve_validation_ = []

        n_batches = len([permutation[x: x + self.batch_size] for x in range(0, n_samples, self.batch_size)])
        if n_batches == 0:
            print("\nWarning: batch_size bigger than number of examples", y.shape)
        
        if y.ndim == 1:
            print("\nWarning: y has been reshaped as column vector because it had shape:", y.shape)
            y = y.reshape((-1, 1))
        
        self.n_outputs_ = y.shape[1]
        
        for epoch in range(n_epochs):
            epoch_loss_aprox = 0
            for batch_indicies in [permutation[x: x + self.batch_size] for x in range(0, n_samples, self.batch_size)]:
                # WARNING: We can do this without slicing arrays we can do it in the theano way passing only indicies
                epoch_loss_aprox += self.partial_fit(X[batch_indicies], y[batch_indicies])
            
            if X_val is not None:
                yhat_validation =  self.output_given_input_evaluation(X_val, self.W, self.b)
                self.loss_curve_validation_.append(self.cost(yhat_validation, y_val).eval())

            np.random.RandomState(self.seed + epoch)
            permutation = np.random.permutation(n_samples)
            self.loss_curve_.append(epoch_loss_aprox/n_batches)


    def predict(self, X):
        """
        Returns the predicted target for each row in X.
        """
        return self.tfunc_predict(X)

    def compute_cost_(self, X, Y):
        """
        Returns the cost for a given set of data X,Y.
        """
        yhat_batch =  self.output_given_input_evaluation(X, self.W, self.b)
        return self.sym_cost_traintime

    def compute_cost(self, X, Y):
        """
        Returns the cost for a given set of data X,Y.
        """
        yhat_batch =  self.output_given_input_evaluation(X, self.W, self.b)
        return self.sym_cost_traintime.eval()
