import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_channels, latent_channels):
        super(Encoder, self).__init__()

        self.lin1 = nn.Linear(in_channels, 128)
        self.lin2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, latent_channels)
        self.fc_logvar = nn.Linear(64, latent_channels)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return self.fc_mu(x), self.fc_logvar(x)

class Decoder(nn.Module):
    def __init__(self, latent_channels, out_channels):
        super(Decoder, self).__init__()

        self.lin1 = nn.Linear(latent_channels, 64)
        self.lin2 = nn.Linear(64, 128)
        self.lin3 = nn.Linear(128, out_channels)

    def forward(self, z):
        z = F.relu(self.lin1(z))
        z = F.relu(self.lin2(z))
        return self.lin3(z)

class Fusion(nn.Module):
    def __init__(self, in_channels, latent_channels):
        super(Encoder, self).__init__()

        self.lin1 = nn.Linear(in_channels, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, latent_channels)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return self.lin3(x)

class VAE(nn.Module):
    def __init__(self, in_channels_atac, in_channels_rna, latent_channels_atac, latent_channels_rna, latent_channels_z):
        super(VAE, self).__init__()

        self.encoder_atac = Encoder(in_channels_atac, latent_channels_atac)
        self.encoder_rna = Encoder(in_channels_rna, latent_channels_rna)
        
        self.fusion = Fusion(latent_channels_atac + latent_channels_rna, latent_channels_z)

        self.decoder_atac = Decoder(latent_channels_z, in_channels_atac)
        self.decoder_rna = Decoder(latent_channels_z, in_channels_rna)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, atac, rna):
        mu1, logvar1 = self.encoder_atac(atac)
        mu2, logvar2 = self.encoder_rna(rna)
        z1 = self.reparameterize(mu1, logvar1)
        z2 = self.reparameterize(mu2, logvar2)
        z = self.fusion(torch.cat((z1, z2), dim=1))
        return z, self.decoder_atac(z), self.decoder_rna(z)

        

