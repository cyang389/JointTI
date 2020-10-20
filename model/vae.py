import torch
import torch.nn as nn
import torch.nn.functional as F

class Fusion_small(nn.Module):
    def __init__(self, in_channels, latent_channels = 2):
        super(Fusion_small, self).__init__()

        self.lin1 = nn.Linear(in_channels, latent_channels)

    def forward(self, x):
        x = self.lin1(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels = [256, 128, 64], latent_channels = 2):
        super(Encoder, self).__init__()

        modules = []
        for h_channel in hidden_channels:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_channel),
                    nn.BatchNorm1d(h_channel),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_channel

        self.encoder = nn.Sequential(*modules)
        self.lin_mu = nn.Linear(hidden_channels[-1], latent_channels)
        self.lin_logvar = nn.Linear(hidden_channels[-1], latent_channels)

    def forward(self, x):
        x = self.encoder(x)
        return self.lin_mu(x), self.lin_logvar(x)

class Decoder(nn.Module):
    def __init__(self, out_channels, hidden_channels = [64, 128, 256], latent_channels = 2):
        super(Decoder, self).__init__()

        modules = []
        in_channels = latent_channels
        for h_channel in hidden_channels:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_channel),
                    nn.BatchNorm1d(h_channel),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_channel
        
        self.decoder = nn.Sequential(*modules)
        self.final = nn.Linear(in_channels, out_channels)

    def forward(self, z):
        z = self.decoder(z)
        return self.final(z)

class vae(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels):
        super(vae, self).__init__()

        self.encoder = Encoder(in_channels = in_channels, hidden_channels = hidden_channels, latent_channels = latent_channels)
        self.decoder = Decoder(out_channels = in_channels, hidden_channels = hidden_channels[::-1], latent_channels = latent_channels)
        
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

class aligned_vae(nn.Module):
    def __init__(self, in_channels_atac, in_channels_rna, latent_channels_atac, latent_channels_rna, latent_channels_z, layer1_channels = 128, layer2_channels = 64):
        super(aligned_vae, self).__init__()
        self.atac_encoder = Encoder(in_channels = in_channels_atac, layer1_channels = layer1_channels, layer2_channels = layer2_channels, latent_channels = latent_channels_atac)
        self.rna_encoder = Encoder(in_channels = in_channels_rna, layer1_channels = layer1_channels, layer2_channels = layer2_channels, latent_channels = latent_channels_rna)
       
        self.fusion_mu = Fusion_small(in_channels = latent_channels_atac + latent_channels_rna, latent_channels = latent_channels_z)
        self.fusion_logvar = Fusion_small(in_channels = latent_channels_atac + latent_channels_rna, latent_channels = latent_channels_z)

        self.atac_decoder = Decoder(latent_channels = latent_channels_z, layer1_channels = layer1_channels, layer2_channels = layer2_channels, out_channels = in_channels_atac)
        self.rna_decoder = Decoder(latent_channels = latent_channels_z, layer1_channels = layer1_channels, layer2_channels = layer2_channels, out_channels = in_channels_rna)

    def forward(self, atac, rna):
        # encode
        mu1, logvar1 = self.atac_encoder(atac)
        mu2, logvar2 = self.rna_encoder(rna)
        muz = self.fusion_mu(torch.cat((mu1, mu2), dim=1))
        logvarz = self.fusion_logvar(torch.cat((logvar1, logvar2), dim=1))
        z_atac = self.reparameterize(mu1, logvar1)
        z_rna = self.reparameterize(mu2, logvar2)
        z = self.reparameterize(muz, logvarz)
        # decode
        return self.atac_decoder(z_atac), self.rna_decoder(z_rna), z, logvarz, muz, z_atac, logvar1, mu1, z_rna, logvar2, mu2

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar * 0.5)
            eps = torch.empty_like(std).normal_()
            z = mu + (std * eps)
            return z
        else:
            return mu

class old_aligned_vae(nn.Module):
    def __init__(self, in_channels_atac, in_channels_rna, latent_channels_atac, latent_channels_rna, latent_channels_z, layer1_channels = 128, layer2_channels = 64):
        super(old_aligned_vae, self).__init__()
        self.atac_encoder = Encoder(in_channels = in_channels_atac, layer1_channels = layer1_channels, layer2_channels = layer2_channels, latent_channels = latent_channels_atac)
        self.rna_encoder = Encoder(in_channels = in_channels_rna, layer1_channels = layer1_channels, layer2_channels = layer2_channels, latent_channels = latent_channels_rna)
       
        self.fusion_mu = Fusion_small(in_channels = latent_channels_atac + latent_channels_rna, layer1_channels = latent_channels_z * 2, latent_channels = latent_channels_z)
        self.fusion_logvar = Fusion_small(in_channels = latent_channels_atac + latent_channels_rna, layer1_channels = latent_channels_z * 2, latent_channels = latent_channels_z)

        self.atac_decoder = Decoder(latent_channels = latent_channels_z, layer1_channels = layer1_channels, layer2_channels = layer2_channels, out_channels = in_channels_atac)
        self.rna_decoder = Decoder(latent_channels = latent_channels_z, layer1_channels = layer1_channels, layer2_channels = layer2_channels, out_channels = in_channels_rna)

    def forward(self, atac, rna):
        # encode
        mu1, logvar1 = self.atac_encoder(atac)
        mu2, logvar2 = self.rna_encoder(rna)
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