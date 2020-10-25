import torch
import torch.nn as nn
import torch.nn.functional as F


CONFIG = { 
    'in_features': 500,
    'layers': [512, 256, 128, 2], # number of nodes in each layer of encoder and decoder.
    'minibatch_size': 256,
    'use_batchnorm': True, # use batch normalization layer.
    'use_tanh': False,
#     'max_iterations': 1000, # max iteration steps
#     'log_interval': 100, # interval of steps to display loss information.
#     'use_gpu': False, 
#     'train_dir': './tmp', # dir to save the state of the model
#     'data_dir': './data', # input files dir
#     'out_dir': './results', # dir to output result files
#     'seed':13 # seed of random number generators
}

class Encoder(nn.Module):
    def __init__(self, cfg = CONFIG):
        super(Encoder, self).__init__()
        
        self.cfg = cfg
        
        self.hidden_layer1 = nn.Linear(in_features = cfg['in_features'], out_features = cfg['layers'][0])
        self.lrelu_1 = nn.LeakyReLU(negative_slope = 0.2)

        self.hidden_layer2 = nn.Linear(in_features = cfg['layers'][0], out_features = cfg['layers'][1])
        self.lrelu_2 = nn.LeakyReLU(negative_slope = 0.2)

        self.hidden_layer3 = nn.Linear(in_features = cfg['layers'][1], out_features = cfg['layers'][2])
        self.lrelu_3 = nn.LeakyReLU(negative_slope = 0.2)

        self.hidden_layer4 = nn.Linear(in_features = cfg['layers'][2], out_features = cfg['layers'][3])
        
        if self.cfg['use_batchnorm']:
            self.batch_norm1 = nn.BatchNorm1d(num_features = self.cfg['layers'][0])
            self.batch_norm2 = nn.BatchNorm1d(num_features = self.cfg['layers'][1])
            self.batch_norm3 = nn.BatchNorm1d(num_features = self.cfg['layers'][2])


    def forward(self, x):
        if self.cfg['use_batchnorm']:
            x = self.lrelu_1(self.batch_norm1(self.hidden_layer1(x)))
            x = self.lrelu_2(self.batch_norm2(self.hidden_layer2(x)))
            x = self.lrelu_3(self.batch_norm3(self.hidden_layer3(x)))
            embed = self.hidden_layer4(x)
        
        else:
            x = self.lrelu_1(self.hidden_layer1(x))
            x = self.lrelu_2(self.hidden_layer2(x))
            x = self.lrelu_3(self.hidden_layer3(x))
            embed = self.hidden_layer4(x)
        
        if self.cfg['use_tanh']:
            embed = F.tanh(embed)
            
        return embed

class Decoder(nn.Module):
    def __init__(self, cfg = CONFIG):
        super(Decoder, self).__init__() 
        
        self.cfg = cfg
        
        self.hidden_layer1 = nn.Linear(in_features = cfg['layers'][3], out_features = cfg['layers'][2])
        self.lrelu_1 = nn.LeakyReLU(negative_slope = 0.2)

        self.hidden_layer2 = nn.Linear(in_features = cfg['layers'][2], out_features = cfg['layers'][1])
        self.lrelu_2 = nn.LeakyReLU(negative_slope = 0.2)

        self.hidden_layer3 = nn.Linear(in_features = cfg['layers'][1], out_features = cfg['layers'][0])
        self.lrelu_3 = nn.LeakyReLU(negative_slope = 0.2)

        self.hidden_layer4 = nn.Linear(in_features = cfg['layers'][0], out_features = cfg['in_features'])
        
        if self.cfg['use_batchnorm']:
            self.batch_norm1 = nn.BatchNorm1d(num_features = cfg['layers'][2])
            self.batch_norm2 = nn.BatchNorm1d(num_features = cfg['layers'][1])
            self.batch_norm3 = nn.BatchNorm1d(num_features = cfg['layers'][0])



    def forward(self, embed):
        if self.cfg['use_batchnorm']:
            x = self.lrelu_1(self.batch_norm1(self.hidden_layer1(embed)))
            x = self.lrelu_2(self.batch_norm2(self.hidden_layer2(x)))
            x = self.lrelu_3(self.batch_norm3(self.hidden_layer3(x)))
            recon = self.hidden_layer4(x)
        
        else:
            x = self.lrelu_1(self.hidden_layer1(embed))
            x = self.lrelu_2(self.hidden_layer2(x))
            x = self.lrelu_3(self.hidden_layer3(x))
            recon = self.hidden_layer4(x)
            
        return recon

class Fusion(nn.Module):
    def __init__(self, in_channels, embed_channels = 2):
        super(Fusion, self).__init__()
        self.linear = nn.Linear(in_channels, embed_channels)
#         self.hidden = nn.Linear(in_channels, hidden_channels)
#         self.output = nn.Linear(hidden_channels, embed_channels)
#         self.lrelu = nn.LeakyRelu(negative_slope = 0.2)
   
    def forward(self, x):
#         embed = self.output(self.lrelu(self.hidden(x)))
        return self.linear(x)



#################################################################################

#                           TEST, with only one dataset                         #

#################################################################################



#################################################################################

#                   unpaired, using adversarial or MMD loss                     #

#################################################################################

# a simple mlp
class discriminator(nn.Module):
    def __init__(self, infeatures = 2, hidden1 = 64, hidden2 = 16, hidden3 = 2):
        super(discriminator, self).__init__()
        self.lin1 = nn.Linear(infeatures, hidden1)
        self.lin2 = nn.Linear(hidden1, hidden2)
        self.lin3 = nn.Linear(hidden2, hidden3)
    
    def forward(self, latent_rep):
        x = F.relu(self.lin1(latent_rep))
        x = F.relu(self.lin2(x))
        # calculate along dimension 1, (0 is batches)
        x = F.softmax(self.lin3(x), dim = 1)
        return x
        

# autoencoder for unpaired dataset
class AE_unpaired(nn.Module):
    def __init__(self, cfg_rna, cfg_atac):
        super(AutoEncoder, self).__init__()
        self.atac_encoder = Encoder(cfg_atac)
        self.rna_encoder = Encoder(cfg_rna)
        
        self.atac_decoder = Decoder(cfg_atac)
        self.rna_decoder = Decoder(cfg_rna)

    def forward(self, atac, rna):
        # encode
        latent_atac = self.atac_encoder(atac)
        latent_rna = self.rna_encoder(rna)

        # decode
        return self.rna_decoder(latent_rna), self.atac_decoder(latent_atac), latent_rna, latent_atac




#################################################################################

#                             Paired, using Fusion network                      #

#################################################################################


class oldAutoEncoder(nn.Module):
    def __init__(self, cfg_rna, cfg_atac):
        super(oldAutoEncoder, self).__init__()
        
        self.atac_encoder = Encoder(cfg_atac)
        self.rna_encoder = Encoder(cfg_atac)
        
        self.fusion = Fusion(in_channels = cfg_rna['layers'][-1] + cfg_atac['layers'][-1], embed_channels = 2)

        self.atac_decoder = Decoder(cfg_atac)
        self.rna_decoder = Decoder(cfg_rna)

    def forward(self, atac, rna):
        # encode
        latent_atac = self.atac_encoder(atac)
        latent_rna = self.rna_encoder(rna)
        z = self.fusion(torch.cat((latent_atac, latent_rna), dim=1))

        # decode
        return self.atac_decoder(z), self.rna_decoder(z), z

    
class AutoEncoder(nn.Module):
    def __init__(self, cfg_rna, cfg_atac):
        super(AutoEncoder, self).__init__()
        self.atac_encoder = Encoder(cfg_atac)
        self.rna_encoder = Encoder(cfg_rna)
        
        self.fusion = Fusion(in_channels = cfg_rna['layers'][-1] + cfg_atac['layers'][-1], embed_channels = 2)

        self.atac_decoder = Decoder(cfg_atac)
        self.rna_decoder = Decoder(cfg_rna)

    def forward(self, atac, rna):
        # encode
        latent_atac = self.atac_encoder(atac)
        latent_rna = self.rna_encoder(rna)
        z = self.fusion(torch.cat((latent_atac, latent_rna), dim=1))

        # decode
        return self.atac_decoder(latent_atac), self.rna_decoder(latent_rna), z, latent_atac, latent_rna