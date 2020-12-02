import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.linalg import svd
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
import diffusion_dist as diff
from model.loss import *
import matplotlib.pyplot as plt
from sklearn import manifold

'''
def pairwise_distance(x):
    """\
    Description:
    -----------
        Pytorch implementation of pairwise distance, similar to squareform(pdist(x))
        
    Parameters:
    -----------
        x: sample by feature matrix
    Returns:
    -----------
        dist: sample by sample pairwise distance
    """
    x_norm = (x**2).sum(1).view(-1, 1)
    y_norm = x_norm.view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(x, 0, 1))
    dist = torch.sqrt(dist + 1e-2)
    return dist 
'''

def lsi_ATAC(X, k = 100, use_first = False):
    """\
    Description:
    ------------
        Compute LSI with TF-IDF transform, i.e. SVD on document matrix, can do tsne on the reduced dimension

    Parameters:
    ------------
        X: cell by feature(region) count matrix
        k: number of latent dimensions
        use_first: since we know that the first LSI dimension is related to sequencing depth, we just ignore the first dimension since, and only pass the 2nd dimension and onwards for t-SNE
    
    Returns:
    -----------
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
    Description:
    ------------    
        Compute tsne

    Parameters:
    ------------
        X: cell by feature(region) count matrix
    
    Returns:
    -----------
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


def train_unpaired(model_rna, model_atac, disc, data_loader_rna, data_loader_atac, diff_sim_rna, 
                   diff_sim_atac, optimizer_rna, optimizer_atac, optimizer_D, n_epochs = 50, 
                   n_iter = 15, lamb_r_rna = 1, lamb_r_atac = 1):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for epoch in range(n_epochs):
        for data in zip(data_loader_rna, data_loader_atac):
            # Update RNA Encoder
            data_rna, data_atac = data
            batch_cols_rna = data_rna['index'].to(device)
            batch_sim_rna = diff_sim_rna[batch_cols_rna,:][:,batch_cols_rna]
            batch_expr_rna = data_rna['count'].to(device)

            batch_expr_r_rna = model_rna(batch_expr_rna)
            z_rna = model_rna[:1](batch_expr_rna)
#             traj_loss(recon_x, x, z, diff_sim, lamb_recon = 1, lamb_dist = 1, recon_mode = "original")
            train_loss_rna, loss_recon_rna, loss_dist_rna = traj_loss(recon_x = batch_expr_r_rna, x = batch_expr_rna, z = z_rna, diff_sim = batch_sim_rna, lamb_recon = lamb_r_rna, lamb_dist = 1, recon_mode = "relative")

            train_loss_rna.backward()
            optimizer_rna.step()
            optimizer_rna.zero_grad()

            # Update ATAC Encoder
            batch_cols_atac = data_atac['index'].to(device)
            batch_sim_atac = diff_sim_atac[batch_cols_atac,:][:,batch_cols_atac]
            batch_expr_atac = data_atac['count'].to(device)

            batch_expr_r_atac = model_atac(batch_expr_atac)
            z_atac = model_atac[:1](batch_expr_atac)

            train_loss_atac, loss_recon_atac, loss_dist_atac = traj_loss(recon_x = batch_expr_r_atac, x = batch_expr_atac, z = z_atac, diff_sim = batch_sim_atac, lamb_recon = lamb_r_atac, lamb_dist = 1, recon_mode = "relative")

            train_loss_atac.backward()
            optimizer_atac.step()
            optimizer_atac.zero_grad()

            # need to go through all the calculation again since the encoder has been updated, ERROR shows up in pytorch 1.5 and above.
            # see: https://github.com/pytorch/pytorch/issues/39141 
            z_rna = model_rna[:1](batch_expr_rna)
            z_atac = model_atac[:1](batch_expr_atac)

            # Update Discriminator
            D_loss_avg = 0
            n_rna = batch_cols_rna.shape[0]
            n_atac = batch_cols_atac.shape[0]
            # note that detach here is necessary, use directly will cause error in encoder update later
            input_disc = torch.cat((z_rna.detach(), z_atac.detach()), dim = 0)
            target = torch.cat((torch.full((n_rna, ), 0, dtype = torch.float), torch.full((n_atac, ), 1, dtype = torch.float))).to(device)
            

            for i in range(n_iter):
                output = disc(input_disc).squeeze()
                D_loss = F.binary_cross_entropy(output, target)
                D_loss_avg += D_loss.item()
                D_loss.backward()
                optimizer_D.step()
                optimizer_D.zero_grad()
            D_loss_avg /= n_iter

            # Update Encoder
            E_loss = -1 * F.binary_cross_entropy(disc(torch.cat((z_rna, z_atac), dim = 0)).squeeze(), target)
            E_loss.backward()
            optimizer_rna.step()
            optimizer_atac.step()
            optimizer_rna.zero_grad()
            optimizer_atac.zero_grad()

        if epoch % 10 == 0:
            log_rna = "RNA loss: {:.5f}, RNA recon loss: {:.5f}, RNA dist loss: {:.5f}".format(train_loss_rna.item(), loss_recon_rna.item(), loss_dist_rna.item())
            log_atac = "ATAC loss: {:.5f}, ATAC recon loss: {:.5f}, ATAC dist loss: {:.5f}".format(train_loss_atac.item(), loss_recon_atac.item(), loss_dist_atac.item())
            log_D = "Discriminator loss: {:.5f}".format(D_loss_avg)
            print("epoch: ", epoch, log_rna, log_atac, log_D)




def train_paired(model_rna, model_atac, disc, data_loader_rna, data_loader_atac, diff_sim_rna, 
                   diff_sim_atac, optimizer_rna, optimizer_atac, optimizer_D, n_epochs = 50, 
                   n_iter = 15, lamb_r_rna = 1, lamb_r_atac = 1):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for epoch in range(n_epochs):
        for data in zip(data_loader_rna, data_loader_atac):
            # Update RNA Encoder
            data_rna, data_atac = data
            batch_cols_rna = data_rna['index'].to(device)
            batch_sim_rna = diff_sim_rna[batch_cols_rna,:][:,batch_cols_rna]
            batch_expr_rna = data_rna['count'].to(device)

            batch_expr_r_rna = model_rna(batch_expr_rna)
            z_rna = model_rna[:1](batch_expr_rna)
            train_loss_rna, loss_recon_rna, loss_dist_rna = traj_loss(recon_x = batch_expr_r_rna, x = batch_expr_rna, z = z_rna, diff_sim = batch_sim_rna, lamb_recon = lamb_r_rna, lamb_dist = 1, recon_mode = "relative")

            train_loss_rna.backward()
            optimizer_rna.step()
            optimizer_rna.zero_grad()

            # Update ATAC Encoder
            batch_cols_atac = data_atac['index'].to(device)
            batch_sim_atac = diff_sim_atac[batch_cols_atac,:][:,batch_cols_atac]
            batch_expr_atac = data_atac['count'].to(device)

            batch_expr_r_atac = model_atac(batch_expr_atac)
            z_atac = model_atac[:1](batch_expr_atac)

            train_loss_atac, loss_recon_atac, loss_dist_atac = traj_loss(recon_x = batch_expr_r_atac, x = batch_expr_atac, z = z_atac, diff_sim = batch_sim_atac, lamb_recon = lamb_r_atac, lamb_dist = 1, recon_mode = "relative")

            train_loss_atac.backward()
            optimizer_atac.step()
            optimizer_atac.zero_grad()

            # need to go through all the calculation again since the encoder has been updated, ERROR shows up in pytorch 1.5 and above.
            # see: https://github.com/pytorch/pytorch/issues/39141 
            z_rna = model_rna[:1](batch_expr_rna)
            z_atac = model_atac[:1](batch_expr_atac)

            # Update Discriminator
            D_loss_avg = 0
            n_rna = batch_cols_rna.shape[0]
            n_atac = batch_cols_atac.shape[0]
            # note that detach here is necessary, use directly will cause error in encoder update later
            input_disc = torch.cat((z_rna.detach(), z_atac.detach()), dim = 0)
            target = torch.cat((torch.full((n_rna, ), 0, dtype = torch.float), torch.full((n_atac, ), 1, dtype = torch.float))).to(device)
            

            for i in range(n_iter):
                output = disc(input_disc).squeeze()
                D_loss = F.binary_cross_entropy(output, target)
                D_loss_avg += D_loss.item()
                D_loss.backward()
                optimizer_D.step()
                optimizer_D.zero_grad()
            D_loss_avg /= n_iter

            # Update Encoder
            E_loss = -1 * F.binary_cross_entropy(disc(torch.cat((z_rna, z_atac), dim = 0)).squeeze(), target)
            E_loss.backward()
            optimizer_rna.step()
            optimizer_atac.step()
            optimizer_rna.zero_grad()
            optimizer_atac.zero_grad()

        if epoch % 10 == 0:
            log_rna = "RNA loss: {:.5f}, RNA recon loss: {:.5f}, RNA dist loss: {:.5f}".format(train_loss_rna.item(), loss_recon_rna.item(), loss_dist_rna.item())
            log_atac = "ATAC loss: {:.5f}, ATAC recon loss: {:.5f}, ATAC dist loss: {:.5f}".format(train_loss_atac.item(), loss_recon_atac.item(), loss_dist_atac.item())
            log_D = "Discriminator loss: {:.5f}".format(D_loss_avg)
            print("epoch: ", epoch, log_rna, log_atac, log_D)



def plot_latent(z1, z2, anno1 = None, anno2 = None, mode = "joint", save = None, figsize = (20,10)):
    fig = plt.figure(figsize = figsize)
    if mode == "modality":
        colormap = plt.cm.get_cmap("Paired")
        ax = fig.add_subplot()
        ax.scatter(z1[:,0], z1[:,1], color = colormap(1), label = "RNA", alpha = 1)
        ax.scatter(z2[:,0], z2[:,1], color = colormap(2), label = "ATAC", alpha = 1)
        ax.legend()
    elif mode == "joint":
        ax = fig.add_subplot()
        cluster_types = np.unique(anno1)
        colormap = plt.cm.get_cmap("tab20", cluster_types.shape[0])

        for i, cluster_type in enumerate(cluster_types):
            index = np.where(anno1 == cluster_type)[0]
            ax.scatter(z1[index,0], z1[index,1], color = colormap(i), label = cluster_type, alpha = 1)
        
        cluster_types = np.unique(anno2)
        colormap = plt.cm.get_cmap("tab20", cluster_types.shape[0])
        for i, cluster_type in enumerate(cluster_types):
            index = np.where(anno2 == cluster_type)[0]
            ax.scatter(z2[index,0], z2[index,1], color = colormap(i), label = cluster_type, alpha = 1)
        ax.legend()

    elif mode == "separate":
        axs = fig.subplots(1,2)
        cluster_types = np.unique(anno1)
        colormap = plt.cm.get_cmap("tab20", cluster_types.shape[0])

        for i, cluster_type in enumerate(cluster_types):
            index = np.where(anno1 == cluster_type)[0]
            axs[0].scatter(z1[index,0], z1[index,1], color = colormap(i), label = cluster_type, alpha = 1)
        axs[0].legend()

        cluster_types = np.unique(anno2)
        colormap = plt.cm.get_cmap("tab20",  cluster_types.shape[0])

        for i, cluster_type in enumerate(cluster_types):
            index = np.where(anno2 == cluster_type)[0]
            axs[1].scatter(z2[index,0], z2[index,1], color = colormap(i), label = cluster_type, alpha = 1)

        axs[1].legend()
    if save:
        fig.savefig(save)

"""
def plot_backbone(model1, model2, loader1, loader2, celltype1, celltype2, device, file_path=None):

    model1.eval()
    model2.eval()
    fig = plt.figure(figsize = (20,10))
    ax = fig.add_subplot()
    ax.set_title('Backbone')

    for data in loader1:
        ae_coordinates = model1[:1](data['count'].to(device)).cpu().detach().numpy()
    cluster_types = np.unique(celltype1)
    colormap = plt.cm.get_cmap("tab20", cluster_types.shape[0])

    for i, cluster_type in enumerate(cluster_types):
        index = np.where(celltype1 == cluster_type)[0]
        ax.scatter(ae_coordinates[index,0], ae_coordinates[index,1], color = colormap(i), alpha = 1)

    for data in loader2:
        ae_coordinates = model2[:1](data['count'].to(device)).cpu().detach().numpy()
    cluster_types = np.unique(celltype2)
    colormap = plt.cm.get_cmap("tab20", cluster_types.shape[0])

    for i, cluster_type in enumerate(cluster_types):
        index = np.where(celltype2 == cluster_type)[0]
        ax.scatter(ae_coordinates[index,0], ae_coordinates[index,1], color = colormap(i), alpha = 1)

    ax.legend(cluster_types)

    if file_path:
        fig.savefig(file_path)

def plot_separate(model1, model2, loader1, loader2, celltype1, celltype2, device, file_path=None):

    model1.eval()
    model2.eval()
    fig = plt.figure(figsize = (20,7))
    axs = fig.subplots(1,2)
    axs[0].set_title('RNA')
    axs[1].set_title('ATAC')

    for data in loader1:
        ae_coordinates = model1[:1](data['count'].to(device)).cpu().detach().numpy()
    cluster_types = np.unique(celltype1)
    colormap = plt.cm.get_cmap("tab20", cluster_types.shape[0])

    for i, cluster_type in enumerate(cluster_types):
        index = np.where(celltype1 == cluster_type)[0]
        axs[0].scatter(ae_coordinates[index,0], ae_coordinates[index,1], color = colormap(i), alpha = 1)
    axs[0].legend(cluster_types)

    for data in loader2:
        ae_coordinates = model2[:1](data['count'].to(device)).cpu().detach().numpy()
    cluster_types = np.unique(celltype2)
    colormap = plt.cm.get_cmap("tab20",  cluster_types.shape[0])

    for i, cluster_type in enumerate(cluster_types):
        index = np.where(celltype2 == cluster_type)[0]
        axs[1].scatter(ae_coordinates[index,0], ae_coordinates[index,1], color = colormap(i), alpha = 1)

    axs[1].legend(cluster_types)
    
    if file_path:
        fig.savefig(file_path)


def plot_merge(model1, model2, loader1, loader2, celltype1, celltype2, device, file_path=None):

    model1.eval()
    model2.eval()
    fig = plt.figure(figsize = (20,10))
    ax = fig.add_subplot()
    ax.set_title('Merge')
    colormap = plt.cm.get_cmap("Paired")

    for data in loader1:
        ae_coordinates = model1[:1](data['count'].to(device)).cpu().detach().numpy()

    ax.scatter(ae_coordinates[:,0], ae_coordinates[:,1], color = colormap(1), label = "batch1", alpha = 1)

    for data in loader2:
        ae_coordinates = model2[:1](data['count'].to(device)).cpu().detach().numpy()
    ax.scatter(ae_coordinates[:,0], ae_coordinates[:,1], color = colormap(2), label = "batch2", alpha = 1)

    ax.legend()
    
    if file_path:
        fig.savefig(file_path)

"""