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
from itertools import cycle

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

def pretrain_embedding(encoder1, encoder2, fusion, decoder1, decoder2, diff_sim1, diff_sim2, data_loader1, data_loader2, 
recon_opt1, recon_opt2, dist_opt1, dist_opt2, n_epochs = 50, lamb_r1 = 1, lamb_r2 = 1, dist_mode = "inner product"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for epoch in range(n_epochs):
        for data in zip(data_loader1, data_loader2):
            data1, data2 = data
            b_idx1 = data1["index"].to(device)
            b_diff_sim1 = diff_sim1[b_idx1,:][:,b_idx1]
            b_data1 = data1["count"].to(device)
            

            # get the latent for discriminator
            z1_1 = encoder1(b_data1)
            # note that the input dimension of decoder should be the same as z1_rna 
            b_r_data1 = decoder1(z1_1)

            loss_r1 = lamb_r1 * recon_loss(recon_x = b_r_data1, x = b_data1, recon_mode = "relative")
            loss_r1.backward()
            recon_opt1.step()
            recon_opt1.zero_grad()

            # Update ATAC Encoder
            b_idx2 = data2['index'].to(device)
            b_diff_sim2 = diff_sim2[b_idx2,:][:,b_idx2]
            b_data2 = data2['count'].to(device)

            # get the latent for discriminator
            z1_2 = encoder2(b_data2)
            # note that the input dimension of decoder should be the same as z1_atac
            b_r_data2 = decoder2(z1_2)

            loss_r2 = lamb_r2 * recon_loss(recon_x = b_r_data2, x = b_data2, recon_mode = "relative")
            loss_r2.backward()
            recon_opt2.step()
            recon_opt2.zero_grad()

            # Update distance
            
            # get the latent for discriminator
            z1_1 = encoder1(b_data1)
            # get the latent for the visualization
            z2_1 = fusion(z1_1)

            loss_d1 = dist_loss(z = z2_1, diff_sim = b_diff_sim1, dist_mode = dist_mode)
            loss_d1.backward()
            dist_opt1.step()
            dist_opt1.zero_grad()

            # get the latent for discriminator
            z1_2 = encoder2(b_data2)
            # get the latent for the visualization
            z2_2 = fusion(z1_2)
            
            loss_d2 = dist_loss(z = z2_2, diff_sim = b_diff_sim2, dist_mode = dist_mode)
            loss_d2.backward()
            dist_opt2.step()
            dist_opt2.zero_grad()

        if epoch % 10 == 0:
            log_rna = "RNA recon loss: {:.5f}, RNA dist loss: {:.5f}".format(loss_r1.item(), loss_d1.item())
            log_atac = "ATAC recon loss: {:.5f}, ATAC dist loss: {:.5f}".format(loss_r2.item(), loss_d2.item())
            print("epoch: ", epoch, log_rna, log_atac)





def pre_train_ae(encoder, decoder, fusion, data_loader, diff_sim, recon_opt, dist_opt, n_epochs = 50, lambda_r = 1, dist_mode = "inner_product"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for epoch in range(n_epochs):
        for data in data_loader:
            b_cell_idx = data["index"].to(device)
            b_diff_sim = diff_sim[b_cell_idx,:][:,b_cell_idx]
            b_expr = data["count"].to(device)

            # reconstruction loss
            b_z1 = encoder(b_expr)
            b_expr_r = decoder(b_z1)
            loss_recon = lambda_r * recon_loss(recon_x = b_expr_r, x = b_expr, recon_mode = "relative")
            loss_recon.backward()
            recon_opt.step()
            recon_opt.zero_grad()

            b_z1 = encoder(b_expr)
            b_z2 = fusion(b_z1)
            loss_dist = 1 * dist_loss(z = b_z2, diff_sim = b_diff_sim, dist_mode = dist_mode)
            loss_dist.backward()
            dist_opt.step()
            dist_opt.zero_grad()

        if epoch % 10 == 0:
            log = "recon loss: {:.5f}, dist loss: {:.5f}".format(loss_recon.item(), loss_dist.item())
            print("epoch: ", epoch, log)


def pre_train_disc(encoder1, encoder2, disc, data_loader1, data_loader2, disc_opt, n_epochs = 10, use_anchor = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for epoch in range(n_epochs):
        for data in zip(data_loader1, data_loader2):
            data1, data2 = data

            b_data1 = data1["count"].to(device)
            b_data2 = data2["count"].to(device)

            
            # before fusion
            z1 = encoder1(b_data1)   
            z2 = encoder2(b_data2)
            # Update Discriminator
            
            n1 = z1.shape[0]
            n2 = z2.shape[0]
            # note that detach here is necessary, use directly will cause error in encoder update later
            input_disc = torch.cat((z1.detach(), z2.detach()), dim = 0)
            target = torch.cat((torch.full((n1, ), 0, dtype = torch.float), torch.full((n2, ), 1, dtype = torch.float))).to(device)
            
            output = disc(input_disc).squeeze()
            D_loss = F.binary_cross_entropy(output, target)
            D_loss.backward()
            disc_opt.step()
            disc_opt.zero_grad()
            

            if use_anchor:
                b_anchor1 = data1["is_anchor"].to(device)
                b_anchor2 = data2["is_anchor"].to(device)
                z1 = encoder1(b_data1)   
                z2 = encoder2(b_data2)

                z1_anchor = z1.detach()[b_anchor1,:]
                z2_anchor = z2.detach()[b_anchor2,:]

                input_disc = torch.cat((z1_anchor, z2_anchor), dim = 0)
                target = torch.cat((torch.zeros(z1_anchor.shape[0], dtype = torch.float), torch.ones(z2_anchor.shape[0], dtype = torch.float)), dim = 0).to(device)
                output = disc(input_disc).squeeze()
                D_loss = F.binary_cross_entropy(output, target)
                



        if epoch % 10 == 0:
            log = "Discriminator loss: {:.5f}".format(D_loss)
            print("epoch: ", epoch, log)


def train_unpaired(encoder1, encoder2, decoder1, decoder2, fusion, disc, data_loader1, data_loader2, diff_sim1, 
                   diff_sim2, recon_opt1, recon_opt2, dist_opt1, dist_opt2, disc_opt, n_epochs = 50, 
                   n_iter = 51, n_iter2 = 1, lamb_r1 = 1, lamb_r2 = 1, lamb_disc = 1, dist_mode = "inner_product", use_anchor = False):
    """\
    Description:
    -----------
        Training the adversarial network
        
    Parameters:
    -----------
        model_rna: 
            auto-encoder for RNA
        model_atac: 
            auto-encoder for ATAC
        disc:
            discriminator
        data_loader_rna:
            data loader for RNA
        data_loader_atac:
            data loader for ATAC
        diff_sim_rna:
            diffusion distance for RNA
        diff_sim_atac:
            diffusion distance for ATAC
        optimizer_rna:
            optimizer for the first auto-encoder
        optimizer_atac:
            optimizer for the second auto-encoder
        optimizer_D:
            optimizer for discriminator
        P_rna:
            transition matrix of RNA diffusion process
        P_atac:
            transition matrix of ATAC diffusion process
        n_epochs:
            number of training epochs
        n_iter:
            number of iteration for the inner loop of adversarial net
        n_iter2:
            number of iteration for the inner loop of the encoder
        lamb_r_rna:
            lambda of reconstruction for RNA
        lamb_r_atac:
            lambda of reconstruction for ATAC
        lamb_disc:
            lambda of discriminator 
        dist_mode:
            distance loss, can be of `inner_product`, `mse` and `kl`
        use_anchor:
            give anchor cluster or not

    Returns:
    -----------
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for epoch in range(n_epochs):
        iteration = zip(data_loader1, data_loader2)
        for data in iteration:
            # Update RNA Encoder
            data1, data2 = data
            b_idx1 = data1["index"].to(device)
            b_diff_sim1 = diff_sim1[b_idx1,:][:,b_idx1]
            b_data1 = data1["count"].to(device)
            

            # get the latent for discriminator
            z1_1 = encoder1(b_data1)
            # note that the input dimension of decoder should be the same as z1_rna 
            b_r_data1 = decoder1(z1_1)

            loss_r1 = lamb_r1 * recon_loss(recon_x = b_r_data1, x = b_data1, recon_mode = "relative")
            loss_r1.backward()
            recon_opt1.step()
            recon_opt1.zero_grad()

            # Update ATAC Encoder
            b_idx2 = data2['index'].to(device)
            b_diff_sim2 = diff_sim2[b_idx2,:][:,b_idx2]
            b_data2 = data2['count'].to(device)

            # get the latent for discriminator
            z1_2 = encoder2(b_data2)
            # note that the input dimension of decoder should be the same as z1_atac
            b_r_data2 = decoder2(z1_2)

            loss_r2 = lamb_r2 * recon_loss(recon_x = b_r_data2, x = b_data2, recon_mode = "relative")
            loss_r2.backward()
            recon_opt2.step()
            recon_opt2.zero_grad()            


            # train_loss_atac, loss_recon_atac, loss_dist_atac = traj_loss(recon_x = batch_expr_r_atac, x = batch_expr_atac, z = z2_atac,
            # diff_sim = batch_sim_atac, lamb_recon = lamb_r_atac, lamb_dist = 1, recon_mode = "relative", dist_mode = dist_mode)

            # train_loss_atac.backward()
            # optimizer_atac.step()
            # optimizer_atac.zero_grad()

            # UPDATE discriminator
            # need to go through all the calculation again since the encoder has been updated, ERROR shows up in pytorch 1.5 and above.
            # see: https://github.com/pytorch/pytorch/issues/39141 
            z1_1 = encoder1(b_data1)
            z1_2 = encoder2(b_data2)

            # Update Discriminator
            D_loss_avg = 0
            n1 = b_idx1.shape[0]
            n2 = b_idx2.shape[0]
            # note that detach here is necessary, use directly will cause error in encoder update later
            input_disc = torch.cat((z1_1.detach(), z1_2.detach()), dim = 0)
            target = torch.cat((torch.full((n1, ), 0, dtype = torch.float), torch.full((n2, ), 1, dtype = torch.float))).to(device)
            

            for i in range(n_iter):
                output = disc(input_disc).squeeze()
                D_loss = lamb_disc * F.binary_cross_entropy(output, target)
                D_loss_avg += D_loss.item()
                D_loss.backward()
                disc_opt.step()
                disc_opt.zero_grad()
            D_loss_avg /= n_iter

            # Update Encoder
            for i in range(n_iter2):
                z1_1 = encoder1(b_data1)
                z1_2 = encoder2(b_data2)
                E_loss = -lamb_disc * F.binary_cross_entropy(disc(torch.cat((z1_1, z1_2), dim = 0)).squeeze(), target)
                E_loss.backward()
                recon_opt1.step()
                recon_opt2.step()
                recon_opt1.zero_grad()
                recon_opt2.zero_grad()
            

            # Update distance
            
            # get the latent for discriminator
            z1_1 = encoder1(b_data1)
            # get the latent for the visualization
            z2_1 = fusion(z1_1)

            loss_d1 = dist_loss(z = z2_1, diff_sim = b_diff_sim1, dist_mode = dist_mode)
            loss_d1.backward()
            dist_opt1.step()
            dist_opt1.zero_grad()

            # get the latent for discriminator
            z1_2 = encoder2(b_data2)
            # get the latent for the visualization
            z2_2 = fusion(z1_2)
            
            loss_d2 = dist_loss(z = z2_2, diff_sim = b_diff_sim2, dist_mode = dist_mode)
            loss_d2.backward()
            dist_opt2.step()
            dist_opt2.zero_grad()
            
            
            if use_anchor:
                # update disc for anchor
                b_anchor1 = data1["is_anchor"].to(device)
                b_anchor2 = data2["is_anchor"].to(device)
                z1_1 = encoder1(b_data1)
                z1_2 = encoder2(b_data2)

                z1_anchor = z1_1.detach()[b_anchor1,:]
                z2_anchor = z1_2.detach()[b_anchor2,:]


                input_disc = torch.cat((z1_anchor, z2_anchor), dim = 0)
                target = torch.cat((torch.zeros(z1_anchor.shape[0], dtype = torch.float), torch.ones(z2_anchor.shape[0], dtype = torch.float)), dim = 0).to(device)
                D_loss_anchor = 0

                for i in range(n_iter):
                    output = disc(input_disc).squeeze()
                    D_loss = lamb_disc * F.binary_cross_entropy(output, target)
                    D_loss_anchor += D_loss.item()
                    D_loss.backward()
                    disc_opt.step()
                    disc_opt.zero_grad()
                D_loss_avg = (D_loss_avg + D_loss_anchor/n_iter)/2
                
                # update encoder for anchor
                for i in range(n_iter2):
                    z1_1 = encoder1(b_data1)
                    z1_2 = encoder2(b_data2)

                    E_loss = -lamb_disc * F.binary_cross_entropy(disc(torch.cat((z1_1[b_anchor1,:], z1_2[b_anchor2,:]), dim = 0)).squeeze(), target)
                    E_loss.backward()
                    recon_opt1.step()
                    recon_opt2.step()
                    recon_opt1.zero_grad()
                    recon_opt2.zero_grad()
             

        if epoch % 10 == 0:
            log_rna = "RNA recon loss: {:.5f}, RNA dist loss: {:.5f}".format(loss_r1.item(), loss_d1.item())
            log_atac = "ATAC recon loss: {:.5f}, ATAC dist loss: {:.5f}".format(loss_r2.item(), loss_d2.item())
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

