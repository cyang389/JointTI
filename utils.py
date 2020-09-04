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