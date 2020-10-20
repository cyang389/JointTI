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

from sklearn import manifold, datasets



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

class hhRNADataset(Dataset):

    def __init__(self, atac_seq_file = "./data/human_hematopoiesis/scATAC_tfscore.csv", rna_seq_file = "./data/human_hematopoiesis/scRNA.csv",
     atac_celltype_file = "./data/human_hematopoiesis/scATAC_celltype.txt", rna_celltype_file = "./data/human_hematopoiesis/scRNA_celltype.txt", dim_reduction = False):
        self.expr_RNA = pd.read_csv(rna_seq_file, index_col=0).to_numpy()[::5,:]
        self.cell_type_RNA = []
        with open(rna_celltype_file, "r") as fp:
            for i in fp:
                self.cell_type_RNA.append(i.strip("\n"))
        self.cell_type_RNA = np.array(self.cell_type_RNA)[::5]
        
        if dim_reduction:
            self.expr_RNA = StandardScaler().fit_transform(self.expr_RNA)
            self.expr_RNA = PCA(n_components=100).fit_transform(self.expr_RNA)

        # self.transform = transform
        self.expr_RNA = torch.FloatTensor(self.expr_RNA)
        
    def __len__(self):
        # number of cells
        return len(self.expr_RNA)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # index denote the index of the cell
        sample = {'RNA':self.expr_RNA[idx,:], 'index':idx, 'cell_type': self.cell_type_RNA[idx]}
        
        return sample

class hhATACDataset(Dataset):

    def __init__(self, atac_seq_file = "./data/human_hematopoiesis/scATAC_tfscore.csv", rna_seq_file = "./data/human_hematopoiesis/scRNA.csv",
     atac_celltype_file = "./data/human_hematopoiesis/scATAC_celltype.txt", rna_celltype_file = "./data/human_hematopoiesis/rna_celltype.txt", dim_reduction = False):
        self.expr_ATAC = pd.read_csv(atac_seq_file, index_col=0).to_numpy()
        self.cell_type_ATAC = []
        with open(atac_celltype_file, "r") as fp:
            for i in fp:
                self.cell_type_ATAC.append(i.strip("\n"))
        self.cell_type_ATAC = np.array(self.cell_type_ATAC)
        
        if dim_reduction:
            self.expr_ATAC = latent_semantic_indexing(self.expr_ATAC, k=100)

        self.expr_ATAC = torch.FloatTensor(self.expr_ATAC)
        
    def __len__(self):
        # number of cells
        return len(self.expr_ATAC)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # index denote the index of the cell
        sample = {'ATAC': self.expr_ATAC[idx,:], 'index':idx, 'cell_type': self.cell_type_ATAC[idx]}
        
        return sample

    
class test_s_curve(Dataset):
    
    def __init__(self):
        n_points = 3000
        X, color = datasets.make_s_curve(n_points, random_state=0)
        self.expr_RNA = torch.FloatTensor(X)
        self.time = color
        
        
    def __len__(self):
        # number of cells
        return len(self.expr_RNA)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # index denote the index of the cell
        sample = {'RNA':self.expr_RNA[idx,:], 'index':idx, 'time': self.time[idx]}
        
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


def graphdata(path, k, diff = "dpt"):
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


class test_paul(Dataset):
    
    def __init__(self, file_path = "./data/Paul/Paul_processed_expr.csv"):
        self.expr_RNA = torch.FloatTensor(pd.read_csv(file_path, index_col=0).values)
        
        
    def __len__(self):
        # number of cells
        return len(self.expr_RNA)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # index denote the index of the cell
        sample = {'RNA':self.expr_RNA[idx,:], 'index':idx}
        
        return sample
