import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.linalg import svd
import pandas as pd 
from sklearn.decomposition import PCA
# from torch_geometric.data import InMemoryDataset, Data
from sklearn.neighbors import kneighbors_graph
import diffusion_dist as diff

from sklearn import manifold


def lsi_ATAC(X, k = 100, use_first = False):
    """\
        Compute LSI with TF-IDF transform, i.e. SVD on document matrix, can do tsne on the reduced dimension

        Parameters:
            X: cell by feature(region) count matrix
            k: number of latent dimensions
            use_first: since we know that the first LSI dimension is related to sequencing depth, we just ignore the first dimension since, and only pass the 2nd dimension and onwards for t-SNE
        Returns:
            latent: cell latent matrix
    """    
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.decomposition import TruncatedSVD

    # binarize the scATAC-Seq count matrix
    bin_X = np.where(X < 1, 0, 1)
    
    # perform Latent Semantic Indexing Analysis
    # get TF-IDF matrix
    tfidf = TfidfTransformer(norm='l2', sublinear_tf=True)
    normed_count = tfidf.fit_transform(bin_X)

    # perform SVD on the sparse matrix
    lsi = TruncatedSVD(n_components = k, random_state=42)
    lsi_r = lsi.fit_transform(normed_count)
    
    # use the first component or not
    if use_first:
        return lsi_r
    else:
        return lsi_r[:, 1:]

def tsne_ATAC(X):
    """\
        Compute tsne

        Parameters:
            X: cell by feature(region) count matrix
        Returns:
            tsne: reduce-dimension matrix
    """       
    X_lsi = lsi_ATAC(X, k = 50, use_first = False)
    tsne = manifold.TSNE(n_components=2,
            learning_rate=200,
            early_exaggeration=20,
            n_iter=2000,
            random_state=42,
            init='pca',
            verbose=1).fit_transform(X_lsi)

    # return first two dimensions for visualization
    return tsne[:,:2]
    

def pca(data, n):
    p = PCA(n_components=n)
    return p.fit_transform(data)

def diff_map_dist(data, n_eign = 10, alpha = 100, diffusion_time = 5):
    diffu_atac = diff.diffusion_map(data, n_eign = 10, alpha = 100, diffusion_time = 5)
    diff_sim_atac = diff.diffusion_similarity(diffu_atac)
    return diff_sim_atac

def normalize(data):
    return data / np.linalg.norm(data)

def dpt_dist(data):
    return diff.DPT_similarity(data)