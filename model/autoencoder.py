import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels, layer1_channels = 128, layer2_channels = 64, latent_channels = 2):
        super(Encoder, self).__init__()

        self.lin1 = nn.Linear(in_channels, layer1_channels)
        self.lin2 = nn.Linear(layer1_channels, layer2_channels)
        self.lin3 = nn.Linear(layer2_channels, latent_channels)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return self.lin3(x)

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

class oldAutoEncoder(nn.Module):
    def __init__(self, in_channels_atac, in_channels_rna, latent_channels_atac, latent_channels_rna, latent_channels_z, layer1_channels = 128, layer2_channels = 64):
        super(oldAutoEncoder, self).__init__()
        self.atac_encoder = Encoder(in_channels = in_channels_atac, layer1_channels = layer1_channels, layer2_channels = layer2_channels, latent_channels = latent_channels_atac)
        self.rna_encoder = Encoder(in_channels = in_channels_rna, layer1_channels = layer1_channels, layer2_channels = layer2_channels, latent_channels = latent_channels_rna)
        
        self.fusion = Encoder(in_channels = latent_channels_atac + latent_channels_rna, layer1_channels = latent_channels_z * 4, layer2_channels = latent_channels_z * 2, latent_channels = latent_channels_z)

        self.atac_decoder = Decoder(latent_channels = latent_channels_z, layer1_channels = layer1_channels, layer2_channels = layer2_channels, out_channels = in_channels_atac)
        self.rna_decoder = Decoder(latent_channels = latent_channels_z, layer1_channels = layer1_channels, layer2_channels = layer2_channels, out_channels = in_channels_rna)

    def forward(self, atac, rna):
        # encode
        latent_atac = self.atac_encoder(atac)
        latent_rna = self.rna_encoder(rna)
        z = self.fusion(torch.cat((latent_atac, latent_rna), dim=1))

        # decode
        return self.atac_decoder(z), self.rna_decoder(z), z

class AutoEncoder(nn.Module):
    def __init__(self, in_channels_atac, in_channels_rna, latent_channels_atac, latent_channels_rna, latent_channels_z, layer1_channels = 128, layer2_channels = 64):
        super(AutoEncoder, self).__init__()
        self.atac_encoder = Encoder(in_channels = in_channels_atac, layer1_channels = layer1_channels, layer2_channels = layer2_channels, latent_channels = latent_channels_atac)
        self.rna_encoder = Encoder(in_channels = in_channels_rna, layer1_channels = layer1_channels, layer2_channels = layer2_channels, latent_channels = latent_channels_rna)
        
        self.fusion = nn.Linear(latent_channels_atac + latent_channels_rna, latent_channels_z)

        self.atac_decoder = Decoder(latent_channels = latent_channels_atac, layer1_channels = layer1_channels, layer2_channels = layer2_channels, out_channels = in_channels_atac)
        self.rna_decoder = Decoder(latent_channels = latent_channels_rna, layer1_channels = layer1_channels, layer2_channels = layer2_channels, out_channels = in_channels_rna)

    def forward(self, atac, rna):
        # encode
        latent_atac = self.atac_encoder(atac)
        latent_rna = self.rna_encoder(rna)
        z = self.fusion(torch.cat((latent_atac, latent_rna), dim=1))

        # decode
        return self.atac_decoder(latent_atac), self.rna_decoder(latent_rna), z, latent_atac, latent_rna