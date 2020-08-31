import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_channels, latent_channels):
        super(Encoder, self).__init__()

        self.lin1 = nn.Linear(in_channels, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, latent_channels)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return self.lin3(x)

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

class AutoEncoder(nn.Module):
    def __init__(self, in_channels_atac, in_channels_rna, latent_channels_atac, latent_channels_rna, latent_channels_z):
        super(AutoEncoder, self).__init__()
        self.atac_encoder = Encoder(in_channels_atac, latent_channels_atac)
        self.rna_encoder = Encoder(in_channels_rna, latent_channels_rna)

        self.fusion = Encoder(latent_channels_atac+latent_channels_rna, latent_channels_z)

        self.atac_decoder = Decoder(latent_channels_z, in_channels_atac)
        self.rna_decoder = Decoder(latent_channels_z, in_channels_rna)

    def forward(self, atac, rna):
        # encode
        latent_atac = self.atac_encoder(atac)
        latent_rna = self.rna_encoder(rna)
        z = self.fusion(torch.cat((latent_atac, latent_rna), dim=1))

        # decode
        return self.atac_decoder(z), self.rna_decoder(z)