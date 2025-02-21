import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd 
from scipy.sparse import csr_matrix

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.neighbors import kneighbors_graph

from utils import *
import anndata
import scanpy as sc


class symsim_batches(Dataset):
    def __init__(self, rand_num = 1, batch_num = 1):
        """\
        Symsim dataset

        Parameters
        ------------
        rand_num
            dataset number, from 1 to 5
        batch_num
            batch number, from 1 to 2
        """
        path = "./data/Symsim/"

        # n_obs by n_features
        count = pd.read_csv(path + "rand" + str(rand_num) + "/counts.txt", sep = "\t", header = None).values.T
        cell_labels = pd.read_csv(path + "rand" + str(rand_num) + "/cell_labels.txt", sep = "\t")
        
        adata = anndata.AnnData(X = count)
        sc.pp.normalize_per_cell(adata)
        sc.pp.log1p(adata)

        # get the index from batch-selection
        batch_idx = cell_labels.loc[cell_labels["batch"] == batch_num].index.values
        
        # get processed count matrix 
        self.expr = torch.FloatTensor(adata.X[batch_idx,:])
        self.cell_labels = cell_labels.iloc[batch_idx,:]

        # get batch number 
        self.batch_num = batch_num
        
    
    def __len__(self):
        return self.expr.shape[0]
    
    def __getitem__(self, idx):
        # data original data, index the index of cell, label, corresponding labels, batch, corresponding batch number
        sample = {"count": self.expr[idx,:], "index": idx, "batch": self.batch_num}
        return sample

class hhRNA1(Dataset):
    def __init__(self, standardize = False, rna_seq_file = "./data/human_hematopoiesis/count_rna_1.csv", rna_celltype_file = "./data/human_hematopoiesis/celltypes_rna_1.txt"):
        """\
        Symsim dataset

        Parameters
        ------------
        rand_num
            dataset number, from 1 to 5
        batch_num
            batch number, from 1 to 2
        """

        count = pd.read_csv(rna_seq_file, index_col=0).to_numpy()
        cell_labels = []
        with open(rna_celltype_file, "r") as fp:
            for i in fp:
                cell_labels.append(i.strip("\n"))
        cell_labels = np.array(cell_labels)

        self.raw = torch.FloatTensor(count)

        if standardize:
            count = StandardScaler().fit_transform(count)
        
        # get processed count matrix 
        self.expr = torch.FloatTensor(count)
        self.cell_labels = cell_labels

        # get batch number 
        self.batch_num = 1
        
    def __len__(self):
        return self.expr.shape[0]
    
    def __getitem__(self, idx):
        # data original data, index the index of cell, label, corresponding labels, batch, corresponding batch number
        sample = {"count": self.expr[idx,:], "index": idx, "batch": self.batch_num, "raw": self.raw[idx,:]}
        return sample

class hhATAC1(Dataset):
    def __init__(self, standardize = False, atac_seq_file = "./data/human_hematopoiesis/count_atac_1.csv", atac_celltype_file = "./data/human_hematopoiesis/celltypes_atac_1.txt"):
        """\
        Symsim dataset

        Parameters
        ------------
        rand_num
            dataset number, from 1 to 5
        batch_num
            batch number, from 1 to 2
        """

        count = pd.read_csv(atac_seq_file, index_col=0).to_numpy()
        cell_labels = []
        with open(atac_celltype_file, "r") as fp:
            for i in fp:
                cell_labels.append(i.strip("\n"))
        cell_labels = np.array(cell_labels)

        self.raw = torch.FloatTensor(count)

        if standardize:
            count = StandardScaler().fit_transform(count)
        
        # get processed count matrix 
        self.expr = torch.FloatTensor(count)
        self.cell_labels = cell_labels

        # get batch number 
        self.batch_num = 2
        
    def __len__(self):
        return self.expr.shape[0]
    
    def __getitem__(self, idx):
        # data original data, index the index of cell, label, corresponding labels, batch, corresponding batch number
        sample = {"count": self.expr[idx,:], "index": idx, "batch": self.batch_num, "raw": self.raw[idx,:]}
        return sample

class endoRNA(Dataset):
    def __init__(self, standardize = False, rna_seq_file = "./data/E10.5_CD44+_E+HE+IAC/scRNA.csv", rna_celltype_file = "./data/E10.5_CD44+_E+HE+IAC/rna_celltype.txt"):
        """\
        endo dataset

        Parameters
        ------------
        rand_num
            dataset number, from 1 to 5
        batch_num
            batch number, from 1 to 2
        """

        count = pd.read_csv(rna_seq_file, index_col=0).to_numpy()
        cell_labels = []
        with open(rna_celltype_file, "r") as fp:
            for i in fp:
                cell_labels.append(i.strip("\n"))

        cell_labels = np.array(cell_labels)

        self.raw = torch.FloatTensor(count)

        if standardize:
            count = StandardScaler().fit_transform(count)
        
        # get processed count matrix 
        self.expr = torch.FloatTensor(count)
        self.cell_labels = cell_labels

        # get batch number 
        self.batch_num = 1
        
    def __len__(self):
        return self.expr.shape[0]
    
    def __getitem__(self, idx):
        # data original data, index the index of cell, label, corresponding labels, batch, corresponding batch number
        sample = {"count": self.expr[idx,:], "index": idx, "batch": self.batch_num, "raw": self.raw[idx,:]}
        return sample

class endoATAC(Dataset):
    def __init__(self, standardize = False, atac_seq_file = "./data/E10.5_CD44+_E+HE+IAC/scATAC.csv", atac_celltype_file = "./data/E10.5_CD44+_E+HE+IAC/atac_celltype.txt"):
        """\
        Symsim dataset

        Parameters
        ------------
        rand_num
            dataset number, from 1 to 5
        batch_num
            batch number, from 1 to 2
        """

        count = pd.read_csv(atac_seq_file, index_col=0).to_numpy()
        cell_labels = []
        with open(atac_celltype_file, "r") as fp:
            for i in fp:
                cell_labels.append(i.strip("\n"))
        cell_labels = np.array(cell_labels)

        self.raw = torch.FloatTensor(count)

        if standardize:
            count = StandardScaler().fit_transform(count)
        
        # get processed count matrix 
        self.expr = torch.FloatTensor(count)
        self.cell_labels = cell_labels

        # get batch number 
        self.batch_num = 2
        
    def __len__(self):
        return self.expr.shape[0]
    
    def __getitem__(self, idx):
        # data original data, index the index of cell, label, corresponding labels, batch, corresponding batch number
        sample = {"count": self.expr[idx,:], "index": idx, "batch": self.batch_num, "raw": self.raw[idx,:]}
        return sample

class endoRNA_noIAC(Dataset):
    def __init__(self, standardize = False, rna_seq_file = "./data/E10.5_CD44+_E+HE+IAC/scRNA_noIAC.csv", rna_celltype_file = "./data/E10.5_CD44+_E+HE+IAC/rna_celltype_noIAC.txt"):
        """\
        endo dataset

        Parameters
        ------------
        rand_num
            dataset number, from 1 to 5
        batch_num
            batch number, from 1 to 2
        """

        count = pd.read_csv(rna_seq_file, index_col=0).to_numpy()
        cell_labels = []
        with open(rna_celltype_file, "r") as fp:
            for i in fp:
                cell_labels.append(i.strip("\n"))

        cell_labels = np.array(cell_labels)

        self.raw = torch.FloatTensor(count)

        if standardize:
            count = StandardScaler().fit_transform(count)
        
        # get processed count matrix 
        self.expr = torch.FloatTensor(count)
        self.cell_labels = cell_labels

        # get batch number 
        self.batch_num = 1
        
    def __len__(self):
        return self.expr.shape[0]
    
    def __getitem__(self, idx):
        # data original data, index the index of cell, label, corresponding labels, batch, corresponding batch number
        sample = {"count": self.expr[idx,:], "index": idx, "batch": self.batch_num, "raw": self.raw[idx,:]}
        return sample

class endoATAC_noIAC(Dataset):
    def __init__(self, standardize = False, atac_seq_file = "./data/E10.5_CD44+_E+HE+IAC/scATAC_noIAC.csv", atac_celltype_file = "./data/E10.5_CD44+_E+HE+IAC/atac_celltype_noIAC.txt"):
        """\
        Symsim dataset

        Parameters
        ------------
        rand_num
            dataset number, from 1 to 5
        batch_num
            batch number, from 1 to 2
        """

        count = pd.read_csv(atac_seq_file, index_col=0).to_numpy()
        cell_labels = []
        with open(atac_celltype_file, "r") as fp:
            for i in fp:
                cell_labels.append(i.strip("\n"))
        cell_labels = np.array(cell_labels)

        self.raw = torch.FloatTensor(count)

        if standardize:
            count = StandardScaler().fit_transform(count)
        
        # get processed count matrix 
        self.expr = torch.FloatTensor(count)
        self.cell_labels = cell_labels

        # get batch number 
        self.batch_num = 2
        
    def __len__(self):
        return self.expr.shape[0]
    
    def __getitem__(self, idx):
        # data original data, index the index of cell, label, corresponding labels, batch, corresponding batch number
        sample = {"count": self.expr[idx,:], "index": idx, "batch": self.batch_num, "raw": self.raw[idx,:]}
        return sample



class lsiATAC(Dataset):
    def __init__(self, standardize = False, atac_seq_file = "./data/E10.5_CD44+_E+HE+IAC/atac_lsi.csv", atac_celltype_file = "./data/E10.5_CD44+_E+HE+IAC/atac_celltype.txt"):
        """\
        Symsim dataset

        Parameters
        ------------
        rand_num
            dataset number, from 1 to 5
        batch_num
            batch number, from 1 to 2
        """

        count = pd.read_csv(atac_seq_file, index_col=0).to_numpy()
        cell_labels = []
        with open(atac_celltype_file, "r") as fp:
            for i in fp:
                cell_labels.append(i.strip("\n"))
        cell_labels = np.array(cell_labels)

        self.raw = torch.FloatTensor(count)

        if standardize:
            count = StandardScaler().fit_transform(count)
        
        # get processed count matrix 
        self.expr = torch.FloatTensor(count)
        self.cell_labels = cell_labels

        # get batch number 
        self.batch_num = 2
        
    def __len__(self):
        return self.expr.shape[0]
    
    def __getitem__(self, idx):
        # data original data, index the index of cell, label, corresponding labels, batch, corresponding batch number
        sample = {"count": self.expr[idx,:], "index": idx, "batch": self.batch_num, "raw": self.raw[idx,:]}
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
        sample = {'count':self.expr_RNA[idx,:], 'index':idx, 'time': self.time[idx]}
        
        return sample

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
        sample = {'count':self.expr_RNA[idx,:], 'index':idx}
        
        return sample





"""
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

    
class scDataset_500(Dataset):

    def __init__(self, atac_seq_file = "./data/expr_rna_500.csv", rna_seq_file = "./data/expr_rna_500.csv", dim_reduction = False):
        self.expr_ATAC = pd.read_csv(atac_seq_file, index_col=0).to_numpy()
        self.expr_RNA = pd.read_csv(rna_seq_file, index_col=0).to_numpy()
        
        if dim_reduction:
            self.expr_RNA = StandardScaler().fit_transform(self.expr_RNA)
            self.expr_RNA = PCA(n_components=100).fit_transform(self.expr_RNA)
            self.expr_ATAC = lsi_ATAC(self.expr_ATAC, k=100)

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
            self.expr_ATAC = lsi_ATAC(self.expr_ATAC, k=100)

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

"""
