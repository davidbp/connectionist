import numpy as np
from kmeans import Kmeans
from sklearn.base import BaseEstimator, ClassifierMixin
 
class RBF_normal_equation(BaseEstimator, ClassifierMixin):
    """
    Classical Radial basis function where the first layer weights are set via clustering
    and the output layer weights are learned using the "normal equation method".
    Therefore this RBF implementation minimizes the MSE error between the predictions given
    by the model, and the targets.
    """
    def __init__(self, nRBF, kmeans_flag=True):
        self.kmeans_flag = kmeans_flag
        self.nRBF = nRBF
 
    def one_hot_encoding(self, inputs, targets):
        """
        Generates a onehotencoding representation of the targets
        """
        unique_targets = np.unique(targets)
        number_targets = len(unique_targets)
        if list(unique_targets) != range(number_targets):
            raise ValueError("targets are not correct")
        one_hot_targets = np.zeros((np.shape(inputs)[0], number_targets))
        for clas in unique_targets:
            indices = np.where(targets == clas)
            one_hot_targets[indices, clas] = 1
 
        del number_targets,unique_targets
        return one_hot_targets
 
    def fit(self, inputs, targets, seed = 1):
        """
        Classical Radial basis function learning procedure:
            - the first layer weights are set via clustering
            - output layer weights are learned using the "normal equation method"
        """
 
        if len(inputs) != len(targets):
            raise ValueError("Number of inputs must equal number of targets")
 
        self.classes_ = np.unique(targets)
        self.n_classes = len(self.classes_)
        self.W = []
        self.W.append(np.zeros((inputs.shape[0], self.nRBF)))
        self.W.append(np.zeros((self.nRBF, self.n_classes )))
 
        if self.n_classes < 2:
            raise ValueError("This solver needs samples of at least 2 classes"
            " in the data, but the data contains only one"
            " class: %r" % self.classes_[0])
 
        one_hot_targets = self.one_hot_encoding(inputs, targets)
 
        nRBF = self.nRBF
        maxdist = np.max(np.max(inputs,axis=0)-np.min(inputs, axis=0))
        self.sigma = maxdist/np.sqrt(2*nRBF)
        self.hidden = np.zeros((inputs.shape[0],nRBF+1))
        self.hidden[:-1] = 1
 
        if self.kmeans_flag:
            # Set prototypes for the hidden unit RBFs as centroids of a kmeans
            self.Kmeans = Kmeans(nRBF, inputs)
            self.Kmeans.fit(inputs, centroid_initialization='from_data')
            self.W[0] = self.Kmeans.centroids
        else:
            # Set prototypes for the hidden unit RBFs to be input patterns
            np.random.seed(seed)
            indicies = np.random.shuffle(range(self.nRBF))
            self.W[0] = inputs[indicies, :]
 
        # set the net input of the first layer to be a zero matrix
        Z = np.zeros((inputs.shape[0], self.nRBF))
 
        for i,prototype in enumerate(self.W[0]):
            # Z[:,i] is a vector containing the distances between all inputs to prototype i
            Z[:,i] = np.linalg.norm(inputs - prototype,axis=1)**2/(2.*self.sigma**2)
 
        A1 = np.exp(-Z)
        # Second layer weights training
        self.W[1] = np.dot(np.linalg.pinv(A1), one_hot_targets)
        return self
 
    def forward_propagation(self,inputs):
        """
        Propagates through the the layers the inputs.
        Returns a matrix where row k is the output activations for the input pattern k
        """
        assert isinstance(inputs, np.ndarray), "inputs is not an array"
        assert len(inputs.shape) == 2, "inputs should be a 2D array"
 
        Z = np.zeros((inputs.shape[0],self.nRBF))
        for i,prototype in enumerate(self.W[0]):
            Z[:,i] = np.linalg.norm(inputs - prototype,axis=1)**2/(2.*self.sigma**2)
 
        A1 = np.exp(-Z);
        return np.dot( A1, self.W[1])
 
    def predict(self, inputs):
        """
        Return the output layer predictions for a given input matrix.
        Each row of the input should be a pattern.
        The function returns a matrix where row k corresponds to output activations of the
        RBF network for pattern k.
        """
        output_activations = self.forward_propagation(inputs)
        return np.argmax( output_activations, axis=1)
 
    def predict_proba(self, inputs):
        """
        Propagates through the the layers the inputs and returns the probability
        of each input to below to a each class
        """
        predictions = self.predict(inputs)
        return predictions/np.matrix(np.sum(predictions,axis=1)).T
 
    def score(self, inputs, targets):
        """
        Accuracy as standard score
        """
        return np.sum((self.predict(inputs) == targets))/float( len(targets))
