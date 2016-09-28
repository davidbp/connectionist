import numpy as np

def generate_2d_data(num_samples=500, seed_val = 123): 
    # Initialize random number generator
    np.random.seed(seed_val)
    
    # True parameter values
    alpha, sigma = 1, 1
    beta = [1, 2.5]
    
    # Size of dataset
    dataset_size = 500
    size = dataset_size
    
    # Predictor variable
    X1 = np.random.randn(size)
    X2 = np.random.randn(size) * 0.2
    
    # Simulate outcome variable
    Y = alpha + beta[0]*X1 + beta[1]*X2 + np.random.randn(size)*sigma

    X = np.vstack((X1, X2)).T

    return X, np.array(Y)