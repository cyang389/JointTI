import torch
import torch.nn as nn
import torch.nn.functional as F

class Fusion(nn.Module):
    def __init__(self, in_channels, layer1_channels = 16, layer2_channels = 8, latent_channels = 2):
        super(Fusion, self).__init__()

        self.lin1 = nn.Linear(in_channels, layer1_channels)
        self.lin2 = nn.Linear(layer1_channels, layer2_channels)
        self.lin3 = nn.Linear(layer2_channels, latent_channels)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return self.lin3(x)

class Fusion_small(nn.Module):
    def __init__(self, in_channels, layer1_channels = 8, latent_channels = 2):
        super(Fusion_small, self).__init__()

        self.lin1 = nn.Linear(in_channels, layer1_channels)
        self.lin2 = nn.Linear(layer1_channels, latent_channels)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        return self.lin2(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, layer1_channels = 128, layer2_channels = 64, latent_channels = 2):
        super(Encoder, self).__init__()

        self.lin1 = nn.Linear(in_channels, layer1_channels)
        self.lin2 = nn.Linear(layer1_channels, layer2_channels)
        self.lin_mu = nn.Linear(layer2_channels, latent_channels)
        self.lin_logvar = nn.Linear(layer2_channels, latent_channels)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return self.lin_mu(x), self.lin_logvar(x)

class Decoder(nn.Module):
    def __init__(self, out_channels, layer1_channels = 128, layer2_channels = 64, latent_channels = 2):
        super(Decoder, self).__init__()

        self.lin1 = nn.Linear(latent_channels, layer2_channels)
        self.lin2 = nn.Linear(layer2_channels, layer1_channels)
        self.lin3 = nn.Linear(layer1_channels, out_channels)

    def forward(self, z):
        z = F.relu(self.lin1(z))
        z = F.relu(self.lin2(z))
        return self.lin3(z)

class aligned_vae(nn.Module):
    def __init__(self, in_channels_atac, in_channels_rna, latent_channels_atac, latent_channels_rna, latent_channels_z, layer1_channels = 128, layer2_channels = 64):
        super(aligned_vae, self).__init__()
        self.atac_encoder = Encoder(in_channels = in_channels_atac, layer1_channels = layer1_channels, layer2_channels = layer2_channels, latent_channels = latent_channels_atac)
        self.rna_encoder = Encoder(in_channels = in_channels_rna, layer1_channels = layer1_channels, layer2_channels = layer2_channels, latent_channels = latent_channels_rna)
        
        # self.fusion_mu = Fusion(in_channels = latent_channels_atac + latent_channels_rna, layer1_channels = latent_channels_z * 4, layer2_channels = latent_channels_z * 2, latent_channels = latent_channels_z)
        # self.fusion_logvar = Fusion(in_channels = latent_channels_atac + latent_channels_rna, layer1_channels = latent_channels_z * 4, layer2_channels = latent_channels_z * 2, latent_channels = latent_channels_z)
        
        self.fusion_mu = Fusion_small(in_channels = latent_channels_atac + latent_channels_rna, layer1_channels = latent_channels_z * 2, latent_channels = latent_channels_z)
        self.fusion_logvar = Fusion_small(in_channels = latent_channels_atac + latent_channels_rna, layer1_channels = latent_channels_z * 2, latent_channels = latent_channels_z)

        self.atac_decoder = Decoder(latent_channels = latent_channels_z, layer1_channels = layer1_channels, layer2_channels = layer2_channels, out_channels = in_channels_atac)
        self.rna_decoder = Decoder(latent_channels = latent_channels_z, layer1_channels = layer1_channels, layer2_channels = layer2_channels, out_channels = in_channels_rna)

    def forward(self, atac, rna):
        # encode
        mu1, logvar1 = self.atac_encoder(atac)
        mu2, logvar2 = self.rna_encoder(rna)
        # print(mu1.shape)
        # print(mu2.shape)
        # print(torch.cat((mu1,mu2), dim=1).shape)
        muz = self.fusion_mu(torch.cat((mu1, mu2), dim=1))
        logvarz = self.fusion_logvar(torch.cat((logvar1, logvar2), dim=1))
        z = self.reparameterize(muz, logvarz)
        # decode
        return self.atac_decoder(z), self.rna_decoder(z), z, logvarz, muz

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar * 0.5)
            eps = torch.empty_like(std).normal_()
            z = mu + (std * eps)
            return z
        else:
            return mu

class vae(nn.Module):
    def __init__(self, in_channels, latent_channels, layer1_channels = 128, layer2_channels = 64):
        super(vae, self).__init__()
        self.encoder = Encoder(in_channels = in_channels, layer1_channels = layer1_channels, layer2_channels = layer2_channels, latent_channels = latent_channels)

        self.decoder = Decoder(latent_channels = latent_channels, layer1_channels = layer1_channels, layer2_channels = layer2_channels, out_channels = in_channels)
        
    def forward(self, rna):
        # encode
        mu, logvar = self.encoder(rna)
        # sampling
        z = self.reparameterize(mu, logvar)
        # decode
        return self.decoder(z), z, mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar * 0.5)
            eps = torch.empty_like(std).normal_()
            z = mu + (std * eps)
            return z
        else:
            return mu


class ae_from_vae(nn.Module):
    def __init__(self, in_channels, latent_channels, layer1_channels = 128, layer2_channels = 64):
        super(ae_from_vae, self).__init__()
        self.encoder = Encoder(in_channels = in_channels, layer1_channels = layer1_channels, layer2_channels = layer2_channels, latent_channels = latent_channels)

        self.decoder = Decoder(latent_channels = latent_channels, layer1_channels = layer1_channels, layer2_channels = layer2_channels, out_channels = in_channels)
        
    def forward(self, rna):
        # encode
        mu, logvar = self.encoder(rna)
        # skip sampling
        recon = self.decoder(mu)
        return recon, mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar * 0.5)
            eps = torch.empty_like(std).normal_()
            z = mu + (std * eps)
            return z
        else:
            return mu
        

class AE(nn.Module):
  def __init__(self, **kwarg):
    super().__init__()
    n_features = kwarg["n_features"]
    n_embed = kwarg["n_embed"]
    self.encoder_hidden1 = nn.Linear(in_features=n_features, out_features=128)
    self.relu1 = nn.ReLU()
    self.encoder_hidden2 = nn.Linear(in_features=128, out_features= 32)
    self.relu2 = nn.ReLU()
    self.encoder_output = nn.Linear(in_features=32, out_features=n_embed)
    self.relu3 = nn.ReLU()
    self.decoder_hidden1 = nn.Linear(in_features=n_embed, out_features=32)
    self.relu4 = nn.ReLU()
    self.decoder_hidden2 = nn.Linear(in_features=32, out_features=128)
    self.relu5 = nn.ReLU()
    self.decoder_output = nn.Linear(in_features=128, out_features=n_features)
  
  def forward(self, x):
    # embedded, bottle neck of the output
    x_embed = self.relu3(self.encoder_output(self.relu2(self.encoder_hidden2(self.relu1(self.encoder_hidden1(x))))))
    # final output
    x = self.decoder_output(self.relu5(self.decoder_hidden2(self.relu4(self.decoder_hidden1(x_embed)))))
    return x_embed, x
