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


def generate_1dlinear_data(num_samples=500, seed_val = 123): 
    # Initialize random number generator
    np.random.seed(seed_val)
    
    # True parameter values
    alpha, sigma = 1, 1
    beta = 1
    
    # Size of dataset
    dataset_size = 500
    size = dataset_size
    
    # Predictor variable
    X = np.random.randn(size)
    
    # Simulate outcome variable
    Y = alpha + beta*X + np.random.randn(size)*sigma


    return X, np.array(Y)




def generate_1dsinusoidal_data(x_min = -5,
                               x_max = 10,
                               num_samples=500,
                               seed_val = 123): 
    # Initialize random number generator
    np.random.seed(seed_val)
    
    # True parameter values
    sigma = 1, 1
    
    # Size of dataset
    size = num_samples
    
    # Predictor variable
    X = np.random.uniform(x_min, x_max, size)
    
    ind_left = X< -1
    ind_right = X>1
    # ind middle is  -1<=X<=1
    ind_middle = ~ind_left * ~ind_right
    
    # noise
    Ynoise_small = np.random.normal(0, 0.2, ind_left.sum())
    Ynoise_verysmall =  np.random.normal(0, 0.05, ind_middle.sum())
    Ynoise_big = np.random.normal(0, 0.5, ind_right.sum())
    
    # Simulate outcome variable
    Y = np.sin(X)
    Y[ind_left] = Y[ind_left] + Ynoise_small
    Y[ind_middle] = Y[ind_middle] + Ynoise_verysmall
    Y[ind_right] =  Y[ind_right] + Ynoise_big
    
    return X, np.array(Y)



