# Author: David Buchaca Prats

from timeit import default_timer as timer
import numpy as np
from numpy import outer as np_outer
import time
import matplotlib.pyplot as plt
import numexpr as ne
from numexpr import evaluate 
import sys
import os

class CRBM:
    
    def __init__(self, n_vis, n_hid, n_his,
                 seed=42, sigma=0.2, monitor_time=True, scale_factor = 0, dtype="Float32"):

        self.n_vis = n_vis
        self.n_hid = n_hid
        self.n_his = n_his
        self.seed  = seed
        self.sigma = sigma
        self.monitor_time = monitor_time
        self.scale_factor = scale_factor
        self.dtype = dtype
                    
        self.previous_xneg = None
        np.random.seed(seed)
        
        if scale_factor == 0:  #scale factor for the random initialization of the weights
            scale_factor = 1./(n_vis * n_his)
            
        if dtype == "Float32":
            dtype = np.float32
        elif dtype == "Float64":
            dtype = np.float64
        
        self.W = scale_factor * np.random.normal(0, sigma, [n_hid, n_vis]).astype(dtype)          # vis to hid
        self.A = scale_factor * np.random.normal(0, sigma, [n_vis, n_vis * n_his]).astype(dtype)  # cond to vis
        self.B = scale_factor * np.random.normal(0, sigma, [n_hid, n_vis * n_his]).astype(dtype)  # cond to hid
        self.v_bias    = np.zeros([n_vis, 1]).astype(dtype)
        self.h_bias    = np.zeros([n_hid, 1]).astype(dtype)
        self.dy_v_bias = np.zeros([n_vis, 1]).astype(dtype)
        self.dy_h_bias = np.zeros([n_hid, 1]).astype(dtype) 

        self.num_epochs_trained = 0
        self.lr = 0        
        
    def save(self, model_path, model_name):
        """
        Function to save the information contained in the class in a folder.
        The folder will contain 2 `.json` files.
        """
        
        ### Create a folder where to save models (if it does not exist) 
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        ### Create a folder for the current model with name `model_name`
        model_path = os.path.join(model_path, model_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        else:
            print("The model {} inside folder {} already exists!".format(model_name, model_path))
            return 0

        ### Save all the information to instanciate the same model again
        arguments_init = inspect.signature(CRBM)
        init_params = {k:self.__dict__[k] for k in arguments_init.parameters.keys()} 
        
        with open( os.path.join(model_path, "model_initializer") + '.json', 'w') as outfile:
            json.dump(init_params, outfile,  ensure_ascii=False)
        
        with open( os.path.join(model_path, "model_dict") + '.pickle', 'wb') as outfile:
            pickle.dump(self.__dict__, outfile, protocol=pickle.HIGHEST_PROTOCOL)
            
    @classmethod
    def load(self, model_path):

        if not os.path.exists(model_path):
            print("The model {} does not exist!".format(model_path))
            return
            
        if not os.path.exists( os.path.join(model_path, "model_initializer.json")):
            print( "File {} is not found.".format(os.path.join(model_path, "model_initializer.json")))
            return
            
        if not os.path.exists( os.path.join(model_path, "model_dict.pickle")):
            print( "File {} is not found.".format(os.path.join(model_path, "model_dict.pickle")))
            return
            
        with open( os.path.join(model_path, "model_initializer") + '.json', 'rb') as file:
            model_initializer  = json.load(file)
    
        with open( os.path.join(model_path, "model_dict") + '.pickle', 'rb') as file:
             model_dict = pickle.load(file)
        
        crbm = CRBM(**model_initializer)
        crbm.__dict__ = model_dict

        return crbm

def sig(v):
    return ne.evaluate("1/(1 + exp(-v))")

def split_vis(crbm: CRBM, vis: np.ndarray):
    n_his = vis.shape[0]
    cond = vis[0:(n_his-1), :].T
    x = vis[[n_his-1],:].T
    
    assert  crbm.n_vis == x.shape[0] and crbm.n_vis == cond.shape[0], \
            "crbm.n_vis = {}, is different from x.shape[0] = {} or cond.shape[0] = {}".format(crbm.n_vis,
                                                                                              x.shape[0],
                                                                                              cond.shape[0])
    return x, cond

def dynamic_biases_up(crbm: CRBM, cond: np.ndarray):
    crbm.dy_v_bias = np.dot(crbm.A, cond) + crbm.v_bias 
    crbm.dy_h_bias = np.dot(crbm.B, cond) + crbm.h_bias      
        
def hid_means(crbm: CRBM, vis: np.ndarray):
    p = np.dot(crbm.W, vis) + crbm.dy_h_bias
    return sig(p)
    
def vis_means(crbm: CRBM, hid: np.ndarray):   
    p = np.dot(crbm.W.T, hid) + crbm.dy_v_bias
    return sig(p)

def sample_hiddens(crbm: CRBM, v: np.ndarray, cond: np.ndarray):
    h_mean = sig( np.dot(crbm.W, v) +  np.dot(crbm.B, cond) + crbm.h_bias)
    h_sample = h_mean > np.random.random(h_mean.shape).astype(np.float32)
    return h_sample, h_mean

def sample_visibles(crbm: CRBM, h: np.ndarray, cond: np.ndarray):
    """
    Notice we don't sample or put the sigmoid here since visible units are Gaussian
    """
    v_mean = np.dot(crbm.W.T, h) + np.dot(crbm.A, cond) + crbm.v_bias  
    return v_mean

def CDK(crbm, vis,cond, K=1):
    v_pos_mean = vis
    h_pos_sample, h_pos_mean    = sample_hiddens(crbm,  v_pos_mean, cond)
    v_neg_mean                  = sample_visibles(crbm, h_pos_mean, cond)
    h_neg_sample, h_neg_mean    = sample_hiddens(crbm,  v_neg_mean, cond)

    for i in range(K-1):
        v_neg_mean           = sample_visibles(crbm, h_neg_mean, cond)
        h_neg, h_neg_mean    = sample_hiddens(crbm,  v_neg_mean, cond)
    
    return v_pos_mean, h_pos_mean , v_neg_mean, h_neg_mean

def update_history_as_vec(current_hist_vec, v_new):
    n_feat = v_new.shape[0]
    current_hist_vec[0:-n_feat] = current_hist_vec[n_feat:] 
    current_hist_vec[-n_feat:] = v_new
    return current_hist_vec

def history_mat_to_vec(cond):
    return np.array([cond.flatten('F')]).T


def compute_gradient(crbm, X):
    """
    Computes an approximated gradient of the likelihod (for a given minibatch X) with
    respect to the parameters. 
    """
    vis, cond = split_vis(crbm, X)
    cond = history_mat_to_vec(cond)
        
    v_pos, h_pos, v_neg, h_neg = CDK(crbm, vis, cond)
    n_obs = vis.shape[1]
    
    # for a sigle observation:  dW = h * v^T - h_hat * v_hat^T
    dW = ( np.dot(h_pos, v_pos.T) - np.dot(h_neg, v_neg.T) ) * (1./n_obs)
    dA = ( np.dot(v_pos, cond.T)  - np.dot(v_neg, cond.T)  ) * (1./n_obs)
    dB = ( np.dot(h_pos, cond.T)  - np.dot(h_neg, cond.T)  ) * (1./n_obs) 
    
    dv_bias = np.mean(v_pos - v_neg, axis=1, keepdims=True)
    dh_bias = np.mean(h_pos - h_neg, axis=1, keepdims=True)
    #print("n_obs:", n_obs)

    rec_error = np.linalg.norm(v_pos - v_neg)
    #print( np.sqrt(np.sum((v_pos - v_neg)**2)))
    
    return dW, dA, dB, dv_bias, dh_bias, rec_error

def update_weights_sgd(crbm, grads, learning_rate):
    
    dW, dA, dB, dv_bias, dh_bias = grads #rec_error = compute_gradient(crbm, X)
    crbm.W += dW * learning_rate
    crbm.A += dA * learning_rate
    crbm.B += dB * learning_rate
    
    crbm.v_bias += dv_bias * learning_rate
    crbm.h_bias += dh_bias * learning_rate

def update_weights_sgd_momentum(crbm, grads, learning_rate, ctx, momentum=0.9):
    
    dW, dA, dB, dv_bias, dh_bias = grads 
    
    ctx["W_vel"]        = ctx["W_vel"]      * momentum    +  dW      * learning_rate
    ctx["A_vel"]        = ctx["A_vel"]      * momentum    +  dA      * learning_rate
    ctx["B_vel"]        = ctx["B_vel"]      * momentum    +  dB      * learning_rate
    ctx["v_bias_vel"]   = ctx["v_bias_vel"] * momentum    +  dv_bias * learning_rate
    ctx["h_bias_vel"]   = ctx["h_bias_vel"] * momentum    +  dh_bias * learning_rate
    
    crbm.W += ctx["W_vel"]
    crbm.A += ctx["A_vel"]
    crbm.B += ctx["B_vel"]
    
    crbm.v_bias += ctx["v_bias_vel"]
    crbm.h_bias += ctx["h_bias_vel"]

def get_slice_at_position_k(X, k, n_his):
    """
    Returns a slice of shape  `(n_his + 1)` with the last column beeing the visible
    vector at the current time step `k`.
    """
    assert k > n_his, "Position k = {} is lower than n_his = {}".format(k, n_his)
    assert k <= X.shape[1], "Position k = {} is bigger than number of timesteps of X.shape[1] = {}".format(k, X.shape[0])
    return X[:, (k-(n_his+1)):k]

def build_slices_from_list_of_arrays(list_of_arrays, n_his, n_feat):
    """
    This function creates a list of slices of shape (n_his + 1, n_feat)
    """
    assert list_of_arrays[0].shape[1] == n_feat, "list_of_arrays[0].shape[1]={} but n_feat={}".format( list_of_arrays[0].shape[1], n_feat)
    
    X_slices = []
    
    for m, arr in enumerate(list_of_arrays):
        if arr.shape[0] < n_his + 1:
            print("Sequence {} has length {}".format(m, arr.shape[0])) 
        else:
            for k in range(n_his+1, arr.shape[0] + 1):
                X_slice = arr[(k-n_his-1):k, :]
                if X_slice.shape[0] != n_his+1:
                    print("error!")
                X_slices.append(X_slice)
                
    return X_slices

def CDK_sa(crbm, vis,cond, K=1):
    
    v_pos_mean = vis
    h_pos_sample, h_pos_mean    = sample_hiddens(crbm,  v_pos_mean, cond)
    v_neg_mean                  = sample_visibles(crbm, h_pos_sample, cond)
    h_neg_sample, h_neg_mean    = sample_hiddens(crbm,  v_neg_mean, cond)
    
        
    for i in range(K-1):
        v_neg_mean           = sample_visibles(crbm, h_neg_sample, cond)
        h_neg, h_neg_mean    = sample_hiddens(crbm,  v_neg_mean, cond)

    return v_pos_mean, h_pos_mean , v_neg_mean, h_neg_mean

def generate(crbm, vis, cond_as_vec, n_gibbs=10):
    """ 
    Given initialization(s) of visibles and matching history, generate a sample in the future.
    
        vis:  n_vis * 1 array
            
        cond_as_vec: n_hist * n_vis array
            
        n_gibbs : int
            number of alternating Gibbs steps per iteration
    """
    
    assert cond_as_vec.shape[1] ==1, "cond_as_vec has to be a column vector"
    
    n_seq = vis.shape[0]
    #import pdb; pdb.set_trace()
    #v_pos, h_pos, v_neg, h_neg = CDK(crbm, vis, cond_as_vec, n_gibbs)
    v_pos, h_pos, v_neg, h_neg = CDK_sa(crbm, vis, cond_as_vec, n_gibbs)
    
    return v_neg
    

def generate_n_samples(crbm, vis, cond_as_vec, n_samples, n_gibbs=100):
    """ 
    Given initialization(s) of visibles and matching history, generate a n_samples in the future.
    """
    
    assert cond_as_vec.shape[1] ==1, "cond_as_vec has to be a column vector"
    
    samples = []
    for i in range(n_samples):
        v_new = generate(crbm, vis, cond_as_vec, n_gibbs)
        
        # This should not be here
        #v_new = v_new/np.linalg.norm(v_new)      
        #print("i:", i, "\tv_new:", v_new.T)
        #print("cond_as_vec:", cond_as_vec[-8:].T, "\n\n")
        #v_new[v_new<0] = 0
        
        update_history_as_vec(cond_as_vec, v_new)
        
        samples.append(v_new)

    return samples