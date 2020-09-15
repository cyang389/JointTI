import torch
import torch.nn.functional as F


def pairwise_distance(x):
    x_norm = (x**2).sum(1).view(-1, 1)
    y_norm = x_norm.view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(x, 0, 1))
    
    # # nan for approx 0 distance
    # dist = torch.sqrt(dist)
    return dist 


def gvae_loss(latent1, latent2, adj1, adj2, recon_adj1, recon_adj2, logvar_latent1, logvar_latent2, lamb_align, lamb_kl, dist_loss_type = "cosine"):
    
    # make squared
    adj1 = adj1 ** 2
    adj2 = adj2 ** 2

    loss_align = lamb_align * torch.norm(latent1 - latent2, p = 'fro')

    # cosine similarity loss, don't forget to normalize the matrix before calculate inner product
    if dist_loss_type == "cosine":
        # normalize latent similarity matrix
        recon_adj1 = recon_adj1 / torch.norm(recon_adj1, p='fro')
        recon_adj2 = recon_adj2 / torch.norm(recon_adj2, p = 'fro')

        adj1 = adj1 / torch.norm(adj1, p = "fro")
        adj2 = adj2 / torch.norm(adj2, p = "fro")

        # cosine similarity, loss will be -1 when two matrices are exactly the same with only scale difference        
        similarity_loss1 = - torch.sum(adj1 * recon_adj1)
        similarity_loss2 = - torch.sum(adj2 * recon_adj2)

    # pearson correlationship
    elif dist_loss_type == "pearson":
        
        Vs1 = recon_adj1 - torch.mean(recon_adj1)
        Vs2 = recon_adj2 - torch.mean(recon_adj2)

        Vd1 = adj1 - torch.mean(adj1)
        Vd2 = adj2 - torch.mean(adj2)

        # maximize correlationship, loss will be -1 when two matrices are exactly the same
        similarity_loss1 = - torch.sum(Vs1 * Vd1) / (torch.sqrt(torch.sum(Vs1 ** 2)) * torch.sqrt(torch.sum(Vd1 ** 2)))
        similarity_loss2 = - torch.sum(Vs2 * Vd2) / (torch.sqrt(torch.sum(Vs2 ** 2)) * torch.sqrt(torch.sum(Vd2 ** 2)))
    
    # mse loss
    elif dist_loss_type == "mse":
        orig_1 = (adj1 / torch.norm(adj1, p = 'fro')).reshape(1, -1) * 100
        reco_1 = (recon_adj1 / torch.norm(recon_adj1, p = "fro")).reshape(1, -1) * 100
        assert orig_1.shape[1] == 2641 ** 2
        assert reco_1.shape[1] == 2641 ** 2
        orig_2 = (adj2 / torch.norm(adj2, p = 'fro')).reshape(1, -1) * 100
        reco_2 = (recon_adj2 / torch.norm(recon_adj2, p = "fro")).reshape(1, -1) * 100
        assert orig_2.shape[1] == 2641 ** 2
        assert reco_2.shape[1] == 2641 ** 2

        similarity_loss1 = F.mse_loss(orig_1, reco_1)
        similarity_loss2 = F.mse_loss(orig_2, reco_2)

    # KL-loss
    kl_latent1 = lamb_kl * 0.5 * torch.sum(- logvar_latent1 + torch.exp(logvar_latent1) + latent1 * latent1 - 1)
    kl_latent2 = lamb_kl * 0.5 * torch.sum(- logvar_latent2 + torch.exp(logvar_latent2) + latent2 * latent2 - 1)
    loss = loss_align + similarity_loss1 + similarity_loss2 + kl_latent1 + kl_latent2
    
    return loss, loss_align, similarity_loss1,  similarity_loss2, kl_latent1, kl_latent2



def aligned_gae_loss(latent1, latent2, adj1, adj2, recon_adj1, recon_adj2, lamb_align = 0.01, dist_loss_type = "cosine"):

    loss_align = lamb_align * torch.norm(latent1 - latent2, p = 'fro')
    
    
    adj1 = adj1 ** 2
    adj2 = adj2 ** 2
    adj1 = (adj1 / torch.norm(adj1, p = 'fro')) 
    adj2 = (adj2 / torch.norm(adj2, p = 'fro')) 
    
    recon_adj1 = (recon_adj1 / torch.norm(recon_adj1, p = "fro"))
    recon_adj2 = (recon_adj2 / torch.norm(recon_adj2, p = "fro")) 

    # mse approx, mse loss change sqrt with mean from norm loss
#     similarity_loss1 = torch.norm(recon_adj1 - adj1, p = "fro") ** 2 / (adj1.shape[0] ** 2)
#     similarity_loss2 = torch.norm(recon_adj2 - adj2, p = "fro") ** 2 / (adj2.shape[0] ** 2)
#     similarity_loss1_2 = F.mse_loss(adj1.reshape(1,-1), recon_1.reshape(1,-1), reduce="mean")
#     similarity_loss2_2 = F.mse_loss(adj2.reshape(1,-1), recon_2.reshape(1,-1), reduce="mean")

    if dist_loss_type == "cosine":
        # cosine similarity, loss will be -1 when two matrices are exactly the same with only scale difference        
        similarity_loss1 = - torch.sum(adj1 * recon_adj1)
        similarity_loss2 = - torch.sum(adj2 * recon_adj2)

    elif dist_loss_type == "pearson":
        Vs1 = recon_adj1 - torch.mean(recon_adj1)
        Vs2 = recon_adj2 - torch.mean(recon_adj2)

        Vd1 = adj1 - torch.mean(adj1)
        Vd2 = adj2 - torch.mean(adj2)

        # maximize correlationship, loss will be -1 when two matrices are exactly the same
        similarity_loss1 = - torch.sum(Vs1 * Vd1) / (torch.sqrt(torch.sum(Vs1 ** 2)) * torch.sqrt(torch.sum(Vd1 ** 2)))
        similarity_loss2 = - torch.sum(Vs2 * Vd2) / (torch.sqrt(torch.sum(Vs2 ** 2)) * torch.sqrt(torch.sum(Vd2 ** 2)))
    
    elif dist_loss_type == "mse": 
        similarity_loss1 = torch.norm(recon_adj1 - adj1, p = "fro")
        similarity_loss2 = torch.norm(recon_adj2 - adj2, p = "fro")
        
    else:
        similarity_loss1 = 0
        similarity_loss2 = 0
    
    loss = loss_align + similarity_loss1 + similarity_loss2 
    
    return loss, loss_align, similarity_loss1,  similarity_loss2



def ae_loss(recon_x1, recon_x2, x1, x2, z, dist_x1, dist_x2, lamb, lamb_var, dist_loss_type = "cosine"):
    
    
    # loss_recon_rna = F.mse_loss(recon_rna, rna)
    # loss_recon_atac = F.mse_loss(recon_atac, atac)
    loss_x1 = torch.norm(recon_x1 - x1, p = "fro")
    loss_x2 = torch.norm(recon_x2 - x2, p = "fro")

    # loss_variance = - lamb_var * (torch.sum((z[:,0] - torch.mean(z[:,0])) ** 2) + torch.sum((z[:,1] - torch.mean(z[:,1])) ** 2))

    # cosine similarity loss, don't forget to normalize the matrix before calculate inner product
    if dist_loss_type == "cosine":
        # diff_atac and diff_rna are only constant, but better to be normalized
        dist_recon = pairwise_distance(z)
        # normalize latent similarity matrix
        dist_recon = dist_recon / torch.norm(dist_recon, p='fro')

        # inner product loss, maximize, so add negative before, in addition, make sure those two values are normalized, with norm 1
        loss_dist_x1 = - lamb * torch.sum(dist_x1 * dist_recon)
        loss_dist_x2 = - lamb * torch.sum(dist_x2 * dist_recon)

    # pearson correlationship
    elif dist_loss_type == "pearson":
        dist_recon = pairwise_distance(z)
        Vs = dist_recon - torch.mean(dist_recon)

        Vd_x1 = dist_x1 - torch.mean(dist_x1)
        Vd_x2 = dist_x2 - torch.mean(dist_x2)

        # maximize correlationship
        loss_dist_x1 = - lamb * torch.sum(Vs * Vd_x1) / (torch.sqrt(torch.sum(Vs ** 2)) * torch.sqrt(torch.sum(Vd_x1 ** 2)))
        loss_dist_x2 = - lamb * torch.sum(Vs * Vd_x2) / (torch.sqrt(torch.sum(Vs ** 2)) * torch.sqrt(torch.sum(Vd_x2 ** 2)))
    
    # mse loss
    elif dist_loss_type == "mse":
#         loss_dist_x1 = lamb * F.mse_loss(dist_x1.reshape(-1), pairwise_distance(z).reshape(-1))
#         loss_dist_x2 = lamb * F.mse_loss(dist_x2.reshape(-1), pairwise_distance(z).reshape(-1))
        dist_recon = pairwise_distance(z)
        loss_dist_x1 = lamb * torch.norm(dist_x1/torch.norm(dist_x1, p = "fro") - dist_recon/torch.norm(dist_recon, p = "fro"), p = "fro")
        loss_dist_x2 = lamb * torch.norm(dist_x2/torch.norm(dist_x2, p = "fro") - dist_recon/torch.norm(dist_recon, p = "fro"), p = "fro")

    loss = loss_x1 + loss_x2 + loss_dist_x1 + loss_dist_x2
    return loss, loss_x1, loss_x2,  loss_dist_x1,  loss_dist_x2

