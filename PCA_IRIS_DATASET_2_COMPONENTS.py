import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

class PCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # mean centering
        self.mean = np.mean(X, axis=0)
        #X = X -  self.mean
        X=X.T

        U, S, Vh = np.linalg.svd(X)
        self.components = U[:,:self.n_components]
        self.variance = S

    def transform(self, X):
        # projects data
        #X = X - self.mean
        X = X.T
        return np.dot(self.components.T, X), self.variance

# Load Data
data = datasets.load_iris()
X = data.data
y = data.target

# Testing
if __name__ == "__main__":

    # Project the data onto the 2 primary principal components
    pca = PCA(2)
    pca.fit(X)
    X_projected, S = pca.transform(X)

    print("Shape of X:", X.shape)
    print("Shape of transformed X:", X_projected.shape)

    x1 = X_projected[0, :]
    x2 = X_projected[1, :]
    
    plt.scatter(
        x1, x2, c=y,  alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3)
    )
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar()
    plt.show()

    #Variance explained by different P.C.
    print('First P.C. explains', np.sum(S[0])/np.sum(S), 'of the total variance.')
    print('Second P.C. explains', np.sum(S[1])/np.sum(S), 'of the total variance.')