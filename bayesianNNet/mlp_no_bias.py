
import theano
import numpy as np
import pymc3 as pm
from theano import tensor as T

def mlp_no_bias_binary_classification(input_dim, 
	                                  num_hidden,
	                                  X_train,
	                                  Y_train):

    n_hidden = num_hidden
    input_dim = input_dim
    output_dim = 1
    std_val = 0.5
    
    ann_input = theano.shared(X_train)
    ann_output = theano.shared(Y_train)
    
    n_hidden = 5
    
    # Initialize random weights.
    init_1 = np.random.randn(X_train.shape[1], n_hidden)
    init_2 = np.random.randn(n_hidden, n_hidden)
    init_out = np.random.randn(output_dim)
    
    with pm.Model() as neural_network:
        
        # Weights from input to hidden layer
        weights_in_1 = pm.Normal('w_in_1', 0, sd=std_val , 
                                 shape=(X_train.shape[1], n_hidden), 
                                 testval=init_1)
        
        # Weights from 1st to 2nd layer
        weights_1_2 = pm.Normal('w_1_2', 0, sd=std_val , 
                                shape=(n_hidden, n_hidden), 
                                testval=init_2)
        
        # Weights from hidden layer to output
        weights_2_out = pm.Normal('w_2_out', 0, sd=std_val , 
                                  shape=(n_hidden,), 
                                  testval=init_out)
        
        # Build neural-network
        act_1 = T.tanh(T.dot(ann_input, weights_in_1))
        act_2 = T.tanh(T.dot(act_1, weights_1_2))
        act_out = T.nnet.sigmoid(T.dot(act_2, weights_2_out))
        
        out = pm.Bernoulli('out', 
                           act_out,
                           observed=ann_output)


    with neural_network:    
        step = pm.Metropolis()
        trace = pm.sample(10000, step=step)

    return trace, neural_network