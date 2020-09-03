import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GraphConvolutionSage(nn.Module):
    """
    GraphSAGE
    """

    def __init__(self, in_features, out_features, dropout=0.):
        super(GraphConvolutionSage, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout

        self.weight_neigh = Parameter(torch.FloatTensor(2 * out_features, 2 * out_features))
        self.weight_self = Parameter(torch.FloatTensor(in_features, 2 * out_features))
        self.weight_support = Parameter(torch.FloatTensor(in_features, 2 * out_features))
        self.weight_linear = Parameter(torch.FloatTensor(2 * out_features, out_features))


        # with dimension (1, out_features), with broadcast -> (N, Dout)
        self.bias_support = Parameter(torch.FloatTensor(1, 2 * out_features))
        self.bias_linear = Parameter(torch.FloatTensor(1, out_features))

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_neigh)
        torch.nn.init.xavier_uniform_(self.weight_self)
        torch.nn.init.xavier_uniform_(self.weight_support)
        torch.nn.init.xavier_uniform_(self.weight_linear)

        # initialization requires two dimension
        torch.nn.init.xavier_uniform_(self.bias_support)
        torch.nn.init.xavier_uniform_(self.bias_linear)
        

    def forward(self, input, adj):
        # first dropout some inputs
        input = F.dropout(input, self.dropout, self.training)

        # Message: two ways
        support = F.sigmoid(torch.mm(input, self.weight_support) + self.bias_support)

        # Aggregation:
        # addition here, could try element-wise max, make diagonal position 0
        output = torch.mm(adj, support)

        # Update: 
        # output of dimension N * Dout, 
        # tried tanh and relu, not very good result, add one linear layer
        output = F.tanh(torch.mm(output, self.weight_neigh) + torch.mm(input, self.weight_self))
        output = torch.mm(output, self.weight_linear) + self.bias_linear

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class pairwiseDistDecoder(nn.Module):
    """Decoder for using pair-wise distance for prediction."""

    def __init__(self, dropout):
        super(pairwiseDistDecoder, self).__init__()
        self.dropout = dropout

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        x_norm = (z ** 2).sum(1).view(-1, 1)
        y_norm = x_norm.view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(z, torch.transpose(z, 0, 1))
        return dist 

class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = torch.mm(z, z.t())
        return adj


class gnn_vae(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, hidden_dim3, dropout = 0., use_mlp = True, decoder = "distance"):
        super(gnn_vae, self).__init__()

        self.gc1 = GraphConvolutionSage(input_feat_dim, hidden_dim1, dropout)
        # the later two layers with activation linear
        self.gc2 = GraphConvolutionSage(hidden_dim1, hidden_dim2, dropout)
        self.gc3 = GraphConvolutionSage(hidden_dim2, hidden_dim3, dropout)
        self.gc4 = GraphConvolutionSage(hidden_dim2, hidden_dim3, dropout)

        self.fc1 = nn.Linear(in_features = hidden_dim2, out_features = hidden_dim3, bias = True)
        self.fc2 = nn.Linear(in_features = hidden_dim2, out_features = hidden_dim3, bias = True)
        
        
        self.dc_pairwise = pairwiseDistDecoder(dropout)
        self.dc_inner_prod = InnerProductDecoder(dropout)

        self.decoder_ver = decoder
        self.use_mlp = use_mlp


    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()
        self.gc3.reset_parameters()
        self.gc4.reset_parameters()

        self.fc1.reset_parameters()
        self.fc2.reset_parameters()


    def encode(self, x, adj):
        # N * hidden_dim1
        hidden1 = self.gc2(self.gc1(x, adj), adj)
        # mean and variance of the dimension N * hidden_dim2
        if self.use_mlp:
            mu, logvar = self.fc1(hidden1), self.fc2(hidden1)
        else:
            mu, logvar = self.gc3(hidden1, adj), self.gc4(hidden1, adj)
        return mu, logvar
        

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        
        # return inner product decoder
        if self.decoder_ver == "distance":
            adj_recon = self.dc_pairwise(z)
        elif self.decoder_ver == "inner-product":
            adj_recon = self.dc_inner_prod(z)
        else:
            raise ValueError("incorrect decoder")
        
        return adj_recon, mu, logvar
   


class gnn_ae(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, hidden_dim3, dropout = 0., use_mlp = True, decoder = "distance"):
        super(gnn_ae, self).__init__()

        self.gc1 = GraphConvolutionSage(input_feat_dim, hidden_dim1, dropout)
        # the later two layers with activation linear
        self.gc2 = GraphConvolutionSage(hidden_dim1, hidden_dim2, dropout)

        # final layer can be either graph conv or linear
        self.gc3 = GraphConvolutionSage(hidden_dim2, hidden_dim3, dropout)

        self.fc1 = nn.Linear(in_features = hidden_dim2, out_features = hidden_dim3, bias = True)
        
        self.dc_pairwise = pairwiseDistDecoder(dropout)
        self.dc_inner_prod = InnerProductDecoder(dropout)

        self.use_mlp = use_mlp
        self.decoder_ver = decoder


    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()
        self.gc3.reset_parameters()
        self.fc1.reset_parameters()


    def encode(self, x, adj):
        # N * hidden_dim1
        hidden1 = self.gc2(self.gc1(x, adj), adj)
        # mean and variance of the dimension N * hidden_dim2
        if self.use_mlp == True:
            z = self.fc1(hidden1)
        else:
            z = self.gc3(hidden1, adj)
        return z

    def forward(self, x, adj):
        z = self.encode(x, adj)
        
        if self.decoder_ver == "distance":
            adj_recon = self.dc_pairwise(z)
        elif self.decoder_ver == "inner-product":
            adj_recon = self.dc_inner_prod(z)       
        

        return adj_recon, z


class aligned_gvae(nn.Module):
    def __init__(self, feature1_dim, feature2_dim, hidden_dim1, hidden_dim2, hidden_dim3, dropout = 0., use_mlp = True, decoder = "distance"):
        super(aligned_gvae, self).__init__()

        self.gvae1 = gnn_vae(feature1_dim, hidden_dim1, hidden_dim2, hidden_dim3, dropout = dropout, use_mlp = use_mlp, decoder = decoder)
        self.gvae2 = gnn_vae(feature2_dim, hidden_dim1, hidden_dim2, hidden_dim3, dropout=dropout, use_mlp = use_mlp, decoder = decoder)
    
    def reset_parameters(self):
        self.gvae1.reset_parameters()
        self.gvae2.reset_parameters()

    def forward(self, x1, x2, adj1, adj2):
        adj_recon1, mu_x1, logvar_x1 = self.gvae1(x1, adj1)
        adj_recon2, mu_x2, logvar_x2 = self.gvae2(x2, adj2)

        return adj_recon1, adj_recon2, mu_x1, mu_x2, logvar_x1, logvar_x2



class aligned_gae(nn.Module):
    def __init__(self, feature1_dim, feature2_dim, hidden_dim1, hidden_dim2, hidden_dim3, dropout = 0., use_mlp = True, decoder = "distance"):
        super(aligned_gae, self).__init__()

        self.gae1 = gnn_ae(feature1_dim, hidden_dim1, hidden_dim2, hidden_dim3, dropout = dropout, use_mlp = use_mlp, decoder = decoder)
        self.gae2 = gnn_ae(feature2_dim, hidden_dim1, hidden_dim2, hidden_dim3, dropout=dropout, use_mlp = use_mlp, decoder = decoder)
    
    def reset_parameters(self):
        self.gae1.reset_parameters()
        self.gae2.reset_parameters()

    def forward(self, x1, x2, adj1, adj2):
        adj_recon1, z1 = self.gae1(x1, adj1)
        adj_recon2, z2 = self.gae2(x2, adj2)

        return adj_recon1, adj_recon2, z1, z2

