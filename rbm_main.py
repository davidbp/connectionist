from timeit import default_timer as timer
import numpy as np
import time
import matplotlib.pyplot as plt
import numexpr  as ne
import profile
from RBM import rbm
import pandas

def save_plot(lr, epochs, W, saved_models_folder):
    #namefile = saved_models_folder + '/' + 'W_layer1' + '_epocs' + str(epochs[0]) + '_lr' + str(lr[0]) + '.npy'
    plt.figure(figsize=(15.2, 15))
    for i, comp in enumerate(W.T):
        plt.subplot(15, 15, i + 1)
        plt.imshow(comp.reshape((28, 28)), cmap=plt.cm.gray_r, interpolation='nearest')
        plt.xticks(())
        plt.yticks(())

    plt.suptitle('lr = ' + str(lr) + ', epoch =' + str(epochs), fontsize=10)
    plt.savefig('./plots/animation1000ep_CD15/lr_' + str(lr) + '_epoch_' + str(epochs) + '.png', format='png')
    plt.close()
    return

def main():
    start = timer()
    print('\n - Reading  raw data and binarizing ...')
    X = pandas.read_csv("Datasets/MNIST/testmnist.csv").values
    X_ = pandas.read_csv("Datasets/MNIST/train_mnist.csv")
    X_= X_[X_.columns[1:]].values
    X = np.concatenate((X, X_), axis=0) 
    del(X_)

    print('\t files read')
    X = np.array(X>4, dtype ='float32')

    saved_models_folder = "saved_models"
    visible_dim = X.shape[1]
    hidden_dim = 225
    epochs = 50
    K = 1
    lr = 0.1
    batch_size = 500
    
    Xaux = np.array(X, dtype='float32')
    
    rbm_ = rbm.RBM(visible_dim=visible_dim, hidden_dim=hidden_dim, seed=42, mu=0, sigma=0.3)

    rbm_.fit(Xaux, 
            method='CDK_vectorized',
            K=K,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            plot_weights=False)
    
    save_plot (lr, epochs, rbm.W, saved_models_folder)

    TOTAL_time = timer() - start
    print("took %f seconds" % TOTAL_time)


if __name__ == '__main__':
    main()
