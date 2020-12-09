import torch
import numpy as np
import torch.nn.functional as F


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


def _gaussian_rbf(dist):
    """\
    Description:
    ------------
        Given a pairwise-distance matrix, calculate the gaussian rbf kernel matrix(affinity matrix)
    Paramters:
    ------------
    dist:
        Pairwise-distance matrix, of the shape (n_batch, n_batch)
    Returns:
    -----------
    K:
        Kernel matrix of the shape (n_batch, n_batch)
    """
    # Multi-scale RBF kernel. (average of some bandwidths of gaussian kernels)
    # This must be properly set for the scale of embedding space
    sigmas = [1e-5, 1e-4, 1e-3, 1e-2]
    # beta of the shape (4,1)
    beta = 1. / (2. * torch.unsqueeze(0.05 * torch.tensor(sigmas), 1))
    # (4,1) * (1, cell*cell), s with the shape (4,cell * cell)
    s = torch.matmul(beta, torch.reshape(dist, (1, -1)))
    # torch.sum sum over dimension 0 (all betas) and make the dimension (1, cell * cell), 
    # and reshape it to be (cell, cell) and divide by the beta length
    # average over all beta actually
    K = torch.reshape(torch.sum(torch.exp(-s), 0), dist.shape) / len(sigmas)
    return K


def mmd_loss(z1, z2):
    """\
    Description:
    ------------
        Maximum Mean Discrepancy regularization, using gaussian rbf kernel function 
        Reference [https://stats.stackexchange.com/questions/276497/maximum-mean-discrepancy-distance-distribution]
    Parameters:
    ------------
    z1:
        Learned latent space, of the size (batch_size1, 2)
    z2:
        Learned latent space, of the size (batch_size2, 2)
    Returns:
    ------------
    loss:
        mmd loss
    """
    embed = torch.cat([z1, z2], dim = 0)
    e = embed / torch.mean(embed)
    K = pairwise_distance(e)
    K = K / torch.max(K)
    K = _gaussian_rbf(K)
    loss = 0

    # z1  kernel matrix within z1 cells 
    K1 = K[:z1.shape[0], :z1.shape[0]]
    # z2 kernel matrix within z2 cells
    K2 = K[z1.shape[0]:, z1.shape[0]:]
    # z1 and z2, kernel matrix in between
    K12 = K[:z1.shape[0], z1.shape[0]:]
    
    # estimate the expectation value empirically
    var_within_z1 = torch.sum(K1) / (z1.shape[0]**2)
    var_within_z2 = torch.sum(K2) / (z2.shape[0]**2)
    loss = loss + var_within_z1 + var_within_z2

    var_between_batches = torch.sum(K12) / (z1.shape[0] * z2.shape[0])
    loss -= 2 * var_between_batches

    return loss

def paired_loss(z1, z2):
    """\
    Description:
    ------------
        distribution loss for paired dataset, note that z1 and z2 should be paired 
    Parameters:
    ------------
    z1:
        Learned latent space, of the size (batch_size1, 2)
    z2:
        Learned latent space, of the size (batch_size1, 2)
    Returns:
    ------------
    loss:
        loss
    """
    return F.mse_loss(z2, z1)





def kernel(dist, knn = 5, decay = 5):
    """
    Kernelize the distance matrix

    Parameters
    -------
    dist: Pair-wise distance matrix

    Returns
    -------
    K : kernel matrix, shape=[n_samples, n_samples]
        symmetric matrix with ones down the diagonal
        with no non-negative entries.

    Raises
    ------
    ValueError: if `precomputed` is not an acceptable value
    """

    # np.partition, first sort the value of each row(small to large), pick the knn+1th element, and put it in the knn+1th place, 
    # and the elements smaller than the value are put in the front, rest are put in the later (without changing order)

    # here find the knnth neighbor
    knn_dist = np.partition(dist.detach().numpy(), knn + 1, axis=1)[:, :knn + 1]
    # find the largest one
    bandwidth = np.max(knn_dist, axis=1)
    bandwidth = torch.FloatTensor(bandwidth)
    # divide by the knnth neighbor
    dist = (dist.T / bandwidth).T
    K = torch.exp(-1 * dist ** decay)

    # handle nan
    # K[K!=K] = 1.0
    # K[K < thresh] = 0

    K = (K + K.T) / 2
    return K


def traj_loss(recon_x, x, z, diff_sim, Pt = None, lamb_recon = 1, lamb_dist = 1, recon_mode = "original", dist_mode = "inner_product", use_potential = True):
    """\
    Description:
    ------------
        Loss for latent space learning that preserve the trajectory structure. Include reconstruction(relative) loss and distance preservation loss.
    
    Parameters:
    ------------
    recon_x:
        Reconstructed feature matrix, of the size (batch_size, n_features)
    x:
        Original input, of the size (batch_size, n_features)
    z:
        Learned latent space, of the size (batch_size, 2)
    diff_sim:
        Diffusion distance calculated on original dataset, ground truth, of the size (batch_size, batch_size)
    Pt:
        Diffusion probability in the original space
    lamb_recon:
        Regularization coefficient for reconstruction loss
    lamb_dis
        Regularization coefficient for distance preservation loss
    recon_mode:
        Reconstruction mode, of two mode, "original" calculuates the original mse loss, "relative" calculates the mse loss of normalized data.  
    
    Returns:
    ------------
    loss:
        Total loss
    loss_recon:
        Reconstruction loss
    loss_dist:
        Distance preservation loss
    """

    if recon_mode == "original":
        loss_recon = lamb_recon * F.mse_loss(recon_x, x)
    elif recon_mode == "relative":
        mean_recon = torch.mean(recon_x, dim = 0)
        var_recon = torch.var(recon_x, dim = 0)
        mean_x = torch.mean(x, dim = 0)
        var_x = torch.var(x, dim = 0)
        # relative loss
        loss_recon = lamb_recon * F.mse_loss(torch.div(torch.add(x, -1.0 * mean_x), (torch.sqrt(var_x + 1e-12)+1e-12)), torch.div(torch.add(x, -1.0 * mean_recon), (torch.sqrt(var_recon + 1e-12)+1e-12)))
    else:
        raise ValueError("recon_mode can only be original or relative")

    # cosine similarity loss
    latent_sim = pairwise_distance(z)

    if dist_mode == "inner_product":
        # normalize latent similarity matrix
        latent_sim = latent_sim / torch.norm(latent_sim, p='fro')
        diff_sim = diff_sim / torch.norm(diff_sim, p = 'fro')
        # inner product loss, maximize, so add negative before, in addition, make sure those two values are normalized, with norm 1
        loss_dist = - lamb_dist * torch.sum(diff_sim * latent_sim) 

    elif dist_mode == "mse":
        # MSE loss
        # normalize latent similarity matrix
        latent_sim = latent_sim / torch.norm(latent_sim, p='fro')
        diff_sim = diff_sim / torch.norm(diff_sim, p = 'fro')
        loss_dist = lamb_dist * torch.norm(diff_sim - latent_sim, p = 'fro')

    elif dist_mode == "kl":
        if Pt is None:
            raise ValueError("`Pt` cannot be `None`")
        
        if use_potential:
            K = latent_sim
        else:
            K = kernel(latent_sim)
        
        # normalize into transition probability
        K = K / torch.sum(K, dim = 1)[:,None] + 1e-7


        # n_sample by n_dimension, diffusion distribution from original space
        Pt = Pt / torch.sum(Pt, dim = 1)[:,None] + 1e-7


        # average JSD
        M = 0.5 * (Pt + K)
        kl = 0.5 * torch.sum(K * torch.log(K / M)) + 0.5 *  torch.sum(Pt * torch.log(Pt / M))
        assert torch.isnan(kl) == False

        loss_dist = lamb_dist * kl
    else:
        raise ValueError("`dist_model` should only be `mse` or `inner_product`")

    loss = loss_recon + loss_dist
    return loss, loss_recon, loss_dist

'''
def traj_loss(recon_x, x, z, diff_sim, U_t = None, lamb_recon = 1, lamb_dist = 1, recon_mode = "original", dist_mode = "inner_product"):
    """\
    Description:
    ------------
        Loss for latent space learning that preserve the trajectory structure. Include reconstruction(relative) loss and distance preservation loss.
    
    Parameters:
    ------------
    recon_x:
        Reconstructed feature matrix, of the size (batch_size, n_features)
    x:
        Original input, of the size (batch_size, n_features)
    z:
        Learned latent space, of the size (batch_size, 2)
    diff_sim:
        Diffusion distance calculated on original dataset, ground truth, of the size (batch_size, batch_size)
    lamb_recon:
        Regularization coefficient for reconstruction loss
    lamb_dis
        Regularization coefficient for distance preservation loss
    recon_mode:
        Reconstruction mode, of two mode, "original" calculuates the original mse loss, "relative" calculates the mse loss of normalized data.  
    
    Returns:
    ------------
    loss:
        Total loss
    loss_recon:
        Reconstruction loss
    loss_dist:
        Distance preservation loss
    """

    if recon_mode == "original":
        loss_recon = lamb_recon * F.mse_loss(recon_x, x)
    elif recon_mode == "relative":
        mean_recon = torch.mean(recon_x, dim = 0)
        var_recon = torch.var(recon_x, dim = 0)
        mean_x = torch.mean(x, dim = 0)
        var_x = torch.var(x, dim = 0)
        # relative loss
        loss_recon = lamb_recon * F.mse_loss(torch.div(torch.add(x, -1.0 * mean_x), (torch.sqrt(var_x + 1e-12)+1e-12)), torch.div(torch.add(x, -1.0 * mean_recon), (torch.sqrt(var_recon + 1e-12)+1e-12)))
    else:
        raise ValueError("recon_mode can only be original or relative")

    # cosine similarity loss
    latent_sim = pairwise_distance(z)

    if dist_mode == "inner_product":
        # normalize latent similarity matrix
        latent_sim = latent_sim / torch.norm(latent_sim, p='fro')
        diff_sim = diff_sim / torch.norm(diff_sim, p = 'fro')
        # inner product loss, maximize, so add negative before, in addition, make sure those two values are normalized, with norm 1
        loss_dist = - lamb_dist * torch.sum(diff_sim * latent_sim) 

    elif dist_mode == "mse":
        # MSE loss
        # normalize latent similarity matrix
        latent_sim = latent_sim / torch.norm(latent_sim, p='fro')
        diff_sim = diff_sim / torch.norm(diff_sim, p = 'fro')
        loss_dist = lamb_dist * torch.norm(diff_sim - latent_sim, p = 'fro')
    elif dist_mode == "kl":
        if U_t is None:
            raise ValueError("`U_t` cannot be `None`")
        K = kernel(latent_sim)
        # normalize into transition probability
        P = K / torch.sum(K, dim = 1)[:,None]
        # n_sample by n_dimension, diffusion distribution from original space
        U_t = U_t / torch.sum(U_t, dim = 1)[:,None]
        # average JSD
        M = 0.5 * (U_t + P)
        kl = 0.5 * torch.sum(P * torch.log(P / M)) + 0.5 *  torch.sum(M * torch.log(M / P))
        loss_dist = lamb_dist * kl
    else:
        raise ValueError("`dist_model` should only be `mse` or `inner_product`")

    loss = loss_recon + loss_dist
    return loss, loss_recon, loss_dist
'''

