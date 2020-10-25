import numpy as np
import random
import pandas as pd
import warnings
import scipy

warnings.filterwarnings('ignore')



def phate_similarity(data, n_neigh = 5, t = 5, use_potential = True):
    """\
    Description:
    ------------
        Calculate diffusion distance using Phate/Diffusion Map method
    
    Parameters:
    ------------
        data: 
            Feature matrix of dimension (n_samples, n_features)
        n_neigh:
            The number of neighbor in knn for graph construction
        t:
            The transition timestep t
        use_potential:
            Using potential distance or not, if use, the same as Phate; if not, the same as diffusion map

    Returns:
    -----------    
        dist:
            Similarity matrix
    """
    import graphtools as gt
    from scipy.spatial.distance import pdist, squareform
    
    # pairwise-distance graph
    G = gt.Graph(data, n_pca = 100, knn = n_neigh, use_pygsp=True)
    # obtain transition matrix
    T = G.diff_op.toarray()
    # T to the power of t
    T_t = np.linalg.matrix_power(T, t)
    # calculate potential distance used as feature vector for each cell
    if use_potential:
        U_t = - np.log(T_t + 1e-7)
    else:
        U_t = T_t
    # calculate pairwise feature vector distance
    dist = squareform(pdist(U_t))
    
    return dist
    




def DPT_similarity(data, n_neigh = 5, use_potential = False):
    '''\
    Description:
    -----------
        Calculates DPT between all points in the data, directly ouput similarity matrix, which is the diffusion pseudotime matrix, a little better than diffusion map
        
    Parameters:
    -----------
        data: 
            Feature matrix, numpy.array of the size [n_samples, n_features]
        n_neigh: 
            Larger correspond to slower decay
        use_potential:
            Expand shorter cell and compress distant cell
    
    Returns:
    -----------
        DPT: 
            Similarity matrix calculated from diffusion pseudo-time
    '''
    import graphtools as gt
    from scipy.spatial.distance import pdist, squareform
    # Calculate from raw data would be too noisy, dimension reduction is necessary, construct graph adjacency matrix with n_pca 100
    G = gt.Graph(data, n_pca=100, knn = n_neigh, use_pygsp=True)
    
    # Calculate eigenvectors of the diffusion operator
    # G.diff_op is a diffusion operator, return similarity matrix calculated from diffusion operation
    W, V = scipy.sparse.linalg.eigs(G.diff_op, k=1)
    
    # Remove first eigenspace
    T_tilde = G.diff_op.toarray() - (V[:,0] @ V[:,0].T)
    
    # Calculate M
    I = np.eye(T_tilde.shape[1])
    M = np.linalg.inv(I - T_tilde) - I
    M = np.real(M)    

    # log-potential
    if use_potential:
        M = M - np.min(M, axis = 1)[:,None]
        M = M / np.sum(M, axis = 1)[:,None]
        M = -np.log(M + 1e-7)
        
    DPT = squareform(pdist(M))

    
    return DPT

