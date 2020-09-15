import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.linalg import svd
import pandas as pd 
from sklearn.decomposition import PCA
# from torch_geometric.data import InMemoryDataset, Data
from sklearn.neighbors import kneighbors_graph
from utils import *
import anndata
from scipy.sparse import csr_matrix
import scvelo as scv
import scanpy as sc

def latent_semantic_indexing(X, k = None):
    """\
        Compute LSI with TF-IDF transform, i.e. SVD on document matrix

        Parameters:
            X: cell by feature(region) count matrix
        Returns:
            latent: cell latent matrix
    """
    X = X.T
    count = np.count_nonzero(X, axis=1)
    count = np.log(X.shape[1] / count)
    X = X * count[:,None]

    
    U, S, Vh = svd(X, full_matrices = False, compute_uv = True)
    if k != None:
        latent = np.matmul(Vh[:k, :].T, np.diag(S[:k]))
    else:
        latent = np.matmul(Vh.T, np.diag(S))
    return latent

class scDataset_500(Dataset):

    def __init__(self, atac_seq_file = "./data/expr_rna_500.csv", rna_seq_file = "./data/expr_rna_500.csv", dim_reduction = False):
        self.expr_ATAC = pd.read_csv(atac_seq_file, index_col=0).to_numpy()
        self.expr_RNA = pd.read_csv(rna_seq_file, index_col=0).to_numpy()
        
        if dim_reduction:
            self.expr_RNA = StandardScaler().fit_transform(self.expr_RNA)
            self.expr_RNA = PCA(n_components=100).fit_transform(self.expr_RNA)
            self.expr_ATAC = latent_semantic_indexing(self.expr_ATAC, k=100)

        # self.transform = transform
        self.expr_ATAC *= 10
        self.expr_RNA *= 10
        self.expr_ATAC = torch.FloatTensor(self.expr_ATAC)
        self.expr_RNA = torch.FloatTensor(self.expr_RNA)

        
    def __len__(self):
        # number of cells
        return len(self.expr_RNA)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # index denote the index of the cell
        sample = {'ATAC': self.expr_ATAC[idx,:], 'RNA':self.expr_RNA[idx,:], 'index':idx}
        
        return sample

class scDataset(Dataset):

    def __init__(self, atac_seq_file = "./data/expr_atac_processed.csv", rna_seq_file = "./data/expr_rna_processed.csv", dim_reduction = False):
        self.expr_ATAC = pd.read_csv(atac_seq_file, index_col=0).to_numpy()
        self.expr_RNA = pd.read_csv(rna_seq_file, index_col=0).to_numpy()
        
        if dim_reduction:
            self.expr_RNA = StandardScaler().fit_transform(self.expr_RNA)
            self.expr_RNA = PCA(n_components=100).fit_transform(self.expr_RNA)
            self.expr_ATAC = latent_semantic_indexing(self.expr_ATAC, k=100)

        # self.transform = transform
        self.expr_ATAC = torch.FloatTensor(self.expr_ATAC)
        self.expr_RNA = torch.FloatTensor(self.expr_RNA)

        
    def __len__(self):
        # number of cells
        return len(self.expr_ATAC)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # index denote the index of the cell
        sample = {'ATAC': self.expr_ATAC[idx,:], 'RNA':self.expr_RNA[idx,:], 'index':idx}
        
        return sample


class testDataset(Dataset):

    def __init__(self):

        self.expr_ATAC = torch.FloatTensor(np.tile(np.arange(100), (100, 1)) + np.tile(np.arange(100)[:,None], 100))
        self.expr_RNA = torch.FloatTensor(np.tile(np.arange(200), (100, 1)) + np.tile(np.arange(100)[:,None], 200))
        
    def __len__(self):
        # number of cells
        return len(self.expr_ATAC)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # index denote the index of the cell
        sample = {'ATAC': self.expr_ATAC[idx,:], 'RNA':self.expr_RNA[idx,:], 'index':idx}
        
        # if self.transform:
        #     sample = self.transform(sample)
        
        return sample

class testCurveDataset(Dataset):

    def __init__(self):

        self.expr_ATAC = torch.FloatTensor(np.tile(np.arange(100), (100, 1)) + np.tile(np.arange(100)[:,None], 100))
        self.expr_RNA = torch.FloatTensor(np.tile(np.arange(200), (100, 1)) + np.tile(np.arange(100)[:,None], 200))
        
    def __len__(self):
        # number of cells
        return len(self.expr_ATAC)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # index denote the index of the cell
        sample = {'ATAC': self.expr_ATAC[idx,:], 'RNA':self.expr_RNA[idx,:], 'index':idx}
        
        # if self.transform:
        #     sample = self.transform(sample)
        
        return sample


def graphdata(path, k, diff = "diffmap"):
    data = pd.read_csv(path, index_col = 0).to_numpy()
    data_pca = pca(data, n = 30)

    # normalize
    if diff == "diffmap":
        diff_dist = diff_map_dist(data_pca, n_eign = 10, alpha = 100, diffusion_time = 5)
    elif diff == "dpt":
        diff_dist = dpt_dist(data_pca)

    conn = kneighbors_graph(diff_dist, n_neighbors = k, include_self = False).toarray()
    # conn = conn * conn.T
    conn_diff = conn * diff_dist

    sim_matrix = conn / diff_dist
    sim_matrix[np.isnan(sim_matrix)] = 0
    sim_matrix = torch.FloatTensor(sim_matrix / np.sum(sim_matrix, axis=1)[:,None])
    
    X = torch.FloatTensor(data)
    adj_diff = torch.FloatTensor(conn_diff)
    edge_index_diff = torch.LongTensor(np.array(np.nonzero(adj_diff)))


    return {"X": X, "adj": adj_diff, "edge_index": edge_index_diff, "similarity": sim_matrix}

def testgraphdata(path, k, diff = "diffmap"):

    data = np.tile(np.arange(100), (100, 1)) + np.tile(np.arange(100)[:,None], 100)
    diff_dist = dpt_dist(data)
    conn = kneighbors_graph(diff_dist, n_neighbors = k, include_self = False).toarray()
    # conn = conn * conn.T
    conn_diff = conn * diff_dist

    sim_matrix = conn / diff_dist
    sim_matrix[np.isnan(sim_matrix)] = 0
    sim_matrix = torch.FloatTensor(sim_matrix / np.sum(sim_matrix, axis=1)[:,None])
    
    X = torch.FloatTensor(data)
    adj_diff = torch.FloatTensor(conn_diff)
    edge_index_diff = torch.LongTensor(np.array(np.nonzero(adj_diff)))


    return {"X": X, "adj": adj_diff, "edge_index": edge_index_diff, "similarity": sim_matrix}

class symsimLinear(Dataset):

    def __init__(self, root = "./data/symsim/", dim_reduction = False):
        
        X_unspliced = pd.read_csv(root + "unspliced_counts.txt", sep="\t",header=None).to_numpy().T
        X_spliced = pd.read_csv(root + "spliced_counts.txt", sep = "\t",header=None).to_numpy().T
        true_velo = pd.read_csv(root + "true_velo.txt", sep="\t",header=None).to_numpy().T
        true_time = pd.read_csv(root + "cell_labels.txt", sep = "\t",lineterminator="\n").to_numpy()
        X_obs = pd.DataFrame(data=true_time, index = ['cell_' + str(x) for x in range(X_unspliced.shape[0])], columns = ['back_bone','sim_time'])


        adata_ori = anndata.AnnData(X = csr_matrix(X_spliced),
                        obs = X_obs,
                        layers = dict(
                            unspliced = csr_matrix(X_unspliced),
                            spliced = csr_matrix(X_spliced),
                            true_velo = true_velo
                        ))
    
        adata = adata_ori.copy()
        data = process_adata(adata)
        self.expr_ATAC = data
        self.expr_RNA = data
        
    def __len__(self):
        # number of cells
        return len(self.expr_ATAC)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # index denote the index of the cell
        sample = {'ATAC': self.expr_ATAC[idx,:], 'RNA':self.expr_RNA[idx,:], 'index':idx}
        
        return sample

def process_adata(adata, noise=0.0):

    scv.pp.filter_and_normalize(adata, flavor = 'cell_ranger', min_shared_counts=0, n_top_genes=300, log=True)
    if adata.n_vars > 299:
        adata = adata[:,:299]
    elif adata.n_vars < 299:
        raise ValueError("Feature number", adata.n_vars)

    X_spliced = adata.X.toarray()

    # X_spliced is original, X_pca is after pca
    x = X_spliced.copy()
    x = StandardScaler().fit_transform(x)
    x = torch.FloatTensor(x.copy())

    return x