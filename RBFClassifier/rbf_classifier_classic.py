import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder

class RBFClassifier(BaseEstimator, ClassifierMixin):
    """
    Classical Radial basis function where the first layer weights are set via clustering
    and the output layer weights are learned using the "normal equation method".
    Therefore this RBF implementation minimizes the MSE error between the predictions given
    by the model, and the targets.
    """
    def __init__(self,
                 n_hidden_basis=50, 
                 KMeans_flag=True, 
                 stratified_basis_selection=True,
                 random_state=1234):

        self.KMeans_flag = KMeans_flag
        self.n_hidden_basis = n_hidden_basis
        self.stratified_basis_selection = True
        self.encoder_targets = None
        self.random_state = random_state
 
    def one_hot_encoding(self, targets):
        """
        Generates a onehotencoding representation of the targets 
        returns the non sparse matrix corresponding to the encoding.
        """
        if self.encoder_targets is  None:
            self.encoder_targets = OneHotEncoder()
            targets_encoded = self.encoder_targets.fit_transform(targets)
        else:
            targets_encoded = self.encoder_targets.transform(targets)

        return targets_encoded.toarray()
 
    def verify_fit_inputs(self, inputs, targets, seed, method):

        if len(inputs) != len(targets):
            raise ValueError("Number of inputs must equal number of targets")
        
        if targets.ndim == 1:
            raise ValueError("Targets have ndim=1, need columns vector of targets")

        self.n_classes = len(np.unique(targets))
        if self.n_classes < 2:
            raise ValueError("This solver needs samples of at least 2 classes"
            " in the data, but the data contains only one"
            " class: %r" % self.n_classes)


    def fit(self, inputs, targets, seed = 1, method="two_phase"):
        """
        method = two_phase
            Classical Radial basis function learning procedure:
                - the first layer weights are set via clustering
                - output layer weights are learned using the "normal equation method"
        """
        if targets.ndim == 1:
            targets = targets.reshape(-1,1)

        self.verify_fit_inputs(inputs, targets, seed, method)
        n_samples, n_features = inputs.shape

        self.W = []
        self.W.append(np.zeros((n_features, self.n_hidden_basis)))
        self.W.append(np.zeros((self.n_hidden_basis, self.n_classes )))
 
        one_hot_targets = self.one_hot_encoding(targets)

        maxdist = np.max(np.max(inputs, axis=0) - np.min(inputs, axis=0))
        self.sigma = maxdist / np.sqrt(2 * self.n_hidden_basis)
        self.hidden = np.zeros((inputs.shape[0], self.n_hidden_basis+1))
        self.hidden[:-1] = 1
 
        if self.KMeans_flag:
            # Set prototypes for the hidden unit RBFs as centroids of a kmeans
            self.KMeans = KMeans(self.n_hidden_basis, random_state=self.random_state)
            self.KMeans.fit(inputs)
            self.W[0] = self.KMeans.cluster_centers_.T
        else:
            # Set prototypes for the hidden unit RBFs to be input patterns
            np.random.seed(self.random_state)
            indicies = np.random.shuffle(range(self.n_hidden_basis))
            self.W[0] = inputs[indicies, :]
 
        # set the net input of the first layer to be a zero matrix
        Z = np.zeros((n_samples, self.n_hidden_basis))

        for i,prototype in enumerate(self.W[0].T):
            # Z[i, :] is a vector containing the distances between all inputs to prototype i
            Z[:, i] = np.linalg.norm(inputs - prototype, axis=1)**2 / (2. * self.sigma**2)
           
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
 
        Z = np.zeros((inputs.shape[0], self.n_hidden_basis))
        for i, prototype in enumerate(self.W[0].T):
            Z[:, i] = np.linalg.norm(inputs - prototype, axis=1)**2 / (2. * self.sigma**2)
 
        A1 = np.exp(-Z);
        return np.dot( A1, self.W[1])
 
    def predict(self, inputs):
        """
        Return the output layer predictions for a given input batch.
        Each row of the input should be a pattern.
        The function returns a 2ndim array where row k corresponds 
        to output activations of the RBF network for pattern k.
        """
        output_activations = self.forward_propagation(inputs)
        return np.argmax( output_activations, axis=1)
 
    def predict_proba(self, inputs):
        """
        Propagates through the the layers the inputs and returns the probability
        of each input to below to a each class
        """
        output_activations = self.forward_propagation(inputs)
        return output_activations/np.sum(output_activations, axis=1).reshape(-1,1)
