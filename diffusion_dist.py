import numpy as np
import random
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from scipy.linalg import eigh
import warnings
import scipy

warnings.filterwarnings('ignore')


def find_diffusion_matrix(X=None, alpha=0.15):
    """Function to find the diffusion matrix P
        
        >Parameters:
        alpha - to be used for gaussian kernel function
        X - feature matrix as numpy array
        
        >Returns:
        P_prime, P, Di, K, D_left
    """
    alpha = alpha
        
    dists = euclidean_distances(X, X)
    K = np.exp(-dists**2 / alpha)
    
    r = np.sum(K, axis=0)
    Di = np.diag(1/r)
    P = np.matmul(Di, K)
    
    D_right = np.diag((r)**0.5)
    D_left = np.diag((r)**-0.5)
    P_prime = np.matmul(D_right, np.matmul(P,D_left))

    return P_prime, P, Di, K, D_left


def find_diffusion_map(P_prime, D_left, n_eign = None):
    """Function to find the diffusion coordinates in the diffusion space
        
        >Parameters:
        P_prime - Symmetrized version of Diffusion Matrix P
        D_left - D^{-1/2} matrix
        n_eigen - Number of eigen vectors to return. This is effectively 
                    the dimensions to keep in diffusion space.
        
        >Returns:
        Diffusion_map as np.array object
    """   
    
    eigenValues, eigenVectors = eigh(P_prime)
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    
    diffusion_coordinates = np.matmul(D_left, eigenVectors)
    
    if n_eign == None:
        return diffusion_coordinates
    else:
        return diffusion_coordinates[:,:n_eign]

def diffusion_map(X, n_eign = None, alpha = 0.009, diffusion_time = 5):
    P_prime, P, Di, K, D_left = find_diffusion_matrix(X, alpha = alpha)
    P_prime = np.linalg.matrix_power(P_prime, n = diffusion_time)
    return find_diffusion_map(P_prime, D_left, n_eign = n_eign)    

def diffusion_similarity(diff_map):
    dists = euclidean_distances(diff_map, diff_map)
    return dists


def DPT_similarity(data, n_neign = None):
    '''Calculates DPT between all points in the data, directly ouput similarity matrix, which is the diffusion pseudotime matrix, a little better than diffusion map
    Parameters:
        data: feature matrix, numpy.array of the size [n_samples, n_features]
    
    Returns:
        DPT: similarity matrix calculated from diffusion pseudo-time
    '''
    import graphtools as gt
    from scipy.spatial.distance import pdist, squareform
    # Calculate from raw data would be too noisy, dimension reduction is necessary, construct graph adjacency matrix with n_pca 100
    G = gt.Graph(data, n_pca=100, use_pygsp=True)
    
    # Calculate eigenvectors of the diffusion operator
    # G.diff_op is a diffusion operator, return similarity matrix calculated from diffusion operation
    W, V = scipy.sparse.linalg.eigs(G.diff_op, k=1)
    
    # Remove first eigenspace
    T_tilde = G.diff_op.toarray() - (V[:,0] @ V[:,0].T)
    
    # Calculate M
    I = np.eye(T_tilde.shape[1])
    M = np.linalg.inv(I - T_tilde) - I
    M = np.real(M)

    eigenValues, eigenVectors = eigh(M)
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    
    DPT = squareform(pdist(M))

    # diffusion_coordinates = np.matmul(D_left, eigenVectors)

    # if n_neign == None:
    #     # Calculate DPT
    #     DPT = squareform(pdist(M))
    # else:
    #     # reduce dimensionality
    #     diffusion_coordinates = diffusion_coordinates[:,:n_eign]
    #     DPT = euclidean_distances(diff_map, diff_map)

    return DPT