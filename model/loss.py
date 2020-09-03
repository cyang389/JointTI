import torch
import torch.nn.functional as F


def gvae_loss(latent1, latent2, adj1, adj2, recon_adj1, recon_adj2, logvar_latent1, logvar_latent2, lamb_align, lamb_kl, dist_loss_type = "cosine"):

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


def gae_loss(latent1, latent2, adj1, adj2, recon_adj1, recon_adj2, lamb_align, dist_loss_type = "cosine"):

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
        orig_1 = (adj1 / torch.norm(adj1, p = 'fro')).reshape(1, -1) * 1e4
        reco_1 = (recon_adj1 / torch.norm(recon_adj1, p = "fro")).reshape(1, -1) * 1e4
        assert orig_1.shape[1] == 2641 ** 2
        assert reco_1.shape[1] == 2641 ** 2

        orig_2 = (adj2 / torch.norm(adj2, p = 'fro')).reshape(1, -1) * 1e4
        reco_2 = (recon_adj2 / torch.norm(recon_adj2, p = "fro")).reshape(1, -1) * 1e4
        assert orig_2.shape[1] == 2641 ** 2
        assert reco_2.shape[1] == 2641 ** 2

        similarity_loss1 = F.mse_loss(orig_1, reco_1)
        similarity_loss2 = F.mse_loss(orig_2, reco_2)

    loss = loss_align + similarity_loss1 + similarity_loss2 
    
    return loss, loss_align, similarity_loss1,  similarity_loss2