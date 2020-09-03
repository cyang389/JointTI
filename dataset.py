import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.linalg import svd
import pandas as pd 
from sklearn.decomposition import PCA
from torch_geometric.data import InMemoryDataset, Data
from sklearn.neighbors import kneighbors_graph
from utils import *

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
        
        # if self.transform:
        #     sample = self.transform(sample)
        
        return sample

# # transform
# class standardize(object):

#     def __call__(self, sample):
#         sample_ATAC = StandardScaler().fit_transform(sample['ATAC'][None,:])
#         sample_RNA = StandardScaler().fit_transform(sample['RNA'][None,:])
#         return {'ATAC': torch.from_numpy(sample_ATAC.squeeze()), 'RNA':torch.from_numpy(sample_RNA.squeeze())}

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

class scGraphDataset(InMemoryDataset):

    def __init__(self, root='data/', transform=None, pre_transform=None, atac_seq_file = "./data/expr_atac_processed.csv", rna_seq_file = "./data/expr_rna_processed.csv"):
        super(scGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

        self.expr_ATAC = pd.read_csv(atac_seq_file, index_col=0).to_numpy()
        self.expr_RNA = pd.read_csv(rna_seq_file, index_col=0).to_numpy()
        
        if pre_transform:
            self.expr_RNA = StandardScaler().fit_transform(self.expr_RNA)
            self.expr_RNA = PCA(n_components=100).fit_transform(self.expr_RNA)
            self.expr_ATAC = latent_semantic_indexing(self.expr_ATAC, k=100)

        # self.transform = transform
        self.expr_ATAC = torch.FloatTensor(self.expr_ATAC)
        self.expr_RNA = torch.FloatTensor(self.expr_RNA)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['scGraphDataset.dataset']

    def download(self):
        pass

    def process(self):
        data_list = []
        atac = self.expr_ATAC
        rna = self.expr_RNA

        atac_pca = pca(atac, n=30)
        rna_pca = pca(np.log1p(rna), n=30)

        diff_sim_atac = diff_map_dist(atac_pca, n_eign = 10, alpha = 100, diffusion_time = 5)
        diff_sim_rna = diff_map_dist(rna_pca, n_eign = 10, alpha = 15, diffusion_time = 5)
        DPT_atac = dpt_dist(atac)
        DPT_rna = dpt_dist(rna)

        # normalize
        diff_sim_atac = normalize(diff_sim_atac)
        diff_sim_rna = normalize(diff_sim_rna)
        DPT_atac = normalize(DPT_atac)
        DPT_rna = normalize(DPT_rna)

        conn_atac = kneighbors_graph(diff_sim_atac, 20, include_self=True).toarray()
        conn_rna = kneighbors_graph(diff_sim_rna, 20, include_self=True).toarray()
        conn_atac = conn_atac * conn_atac.T
        conn_rna = conn_rna * conn_rna.T
        conn_atac_diff = conn_atac * diff_sim_atac
        conn_rna_diff = conn_rna * diff_sim_rna

        

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

def graphdata(path, k):

    data_list = []
    data = pd.read_csv(path, index_col=0).to_numpy()

    data_pca = pca(data, n=30)

    diff_sim = diff_map_dist(data_pca, n_eign = 10, alpha = 100, diffusion_time = 5)
    DPT = dpt_dist(data)

    # normalize
    diff_sim = normalize(diff_sim)
    DPT = normalize(DPT)

    conn = kneighbors_graph(diff_sim, n_neighbors=k, include_self=True).toarray()
    conn = conn * conn.T
    conn_diff = conn * diff_sim

    conn = kneighbors_graph(DPT, n_neighbors=k, include_self=True).toarray()
    conn = conn * conn.T
    conn_DPT = conn * DPT

    X = torch.FloatTensor(data)
    adj_diff = torch.FloatTensor(conn_diff)
    adj_DPT = torch.FloatTensor(conn_DPT)
    edge_index_diff = torch.LongTensor(np.array(np.nonzero(adj_diff)))
    edge_index_DPT = torch.LongTensor(np.array(np.nonzero(adj_DPT)))

    return {"X": X, "adj_diff": adj_diff, "adj_DPT": adj_DPT,
        "edge_index_diff": edge_index_diff, "edge_index_DPT": edge_index_DPT}
        