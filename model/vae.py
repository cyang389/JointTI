import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_channels, layer1_channels = 128, layer2_channels = 64, latent_channels = 2):
        super(Encoder, self).__init__()

        self.lin1 = nn.Linear(in_channels, layer1_channels)
        self.lin2 = nn.Linear(layer1_channels, layer2_channels)
        self.fc_mu = nn.Linear(layer2_channels, latent_channels)
        self.fc_logvar = nn.Linear(layer2_channels, latent_channels)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return self.fc_mu(x), self.fc_logvar(x)

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

class VAE(nn.Module):
    def __init__(self, in_channels_atac, in_channels_rna, latent_channels_atac, latent_channels_rna, latent_channels_z):
        super(VAE, self).__init__()

        self.encoder_atac = Encoder(in_channels = in_channels_atac, latent_channels = latent_channels_atac)
        self.encoder_rna = Encoder(in_channels = in_channels_rna, latent_channels = latent_channels_rna)
        
        self.fusion_mu = Fusion(in_channels = latent_channels_atac + latent_channels_rna, latent_channels = latent_channels_z)
        self.fusion_logvar = Fusion(in_channels = latent_channels_atac + latent_channels_rna, latent_channels = latent_channels_z)

        self.decoder_atac = Decoder(latent_channels = latent_channels_z, out_channels = in_channels_atac)
        self.decoder_rna = Decoder(latent_channels = latent_channels_z, out_channels = in_channels_rna)

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
        muz = self.fusion_mu(torch.cat((mu1, mu2), dim=1))
        logvarz = self.fusion_logvar(torch.cat((logvar1, logvar2), dim=1))
        z = self.reparameterize(muz, logvarz)
        return self.decoder_atac(z), self.decoder_rna(z), z

        

