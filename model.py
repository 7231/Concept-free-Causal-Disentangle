import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GraphConvolution, GraphConvolution_A, causal_layer, causal_layer_reperameter
import utils as ut 
import numpy as np
import torch
import random
import numpy

class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, n_nodes, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelVAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)
        self.net = nn.Sequential(
			nn.Linear(n_nodes, 64),
			nn.ELU(),
			nn.Linear(64, 32),
			nn.ELU(),
            nn.Linear(32, 2 * 16),
		)

        self.causal_layer = causal_layer(hidden_dim2, hidden_dim2)
        self.causal_layer_reperameter = causal_layer_reperameter(hidden_dim2, hidden_dim2)
        self.C = self.causal_layer.C
        self.label_net = nn.Sequential(
			nn.Linear(hidden_dim2, 32),
			nn.ELU(),
			nn.Linear(32, 64),
			nn.ELU(),
			nn.Linear(64, input_feat_dim),
		)
        self.hidden_dim2 = hidden_dim2


    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def encode_A(self,adj):
        adj = adj.to_dense()
        xy = adj.view(-1, adj.size()[0])#xy: 64*36864
        h = self.net(xy)#h: 64*32
        m, v = ut.gaussian_parameters(h, dim=1)
        return m, v #m: 64*16 v: 64*16
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z1 = self.reparameterize(mu, logvar)
        #z1 = mu
        z = z1
        z = self.causal_layer(z1)
        z = self.reparameterize(z, logvar)
        return self.label_net(z), self.dc(z), mu, logvar, z1, z


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj


class GCNModelVAE_A(nn.Module):
    def __init__(self, input_feat_dim, n_nodes, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelVAE_A, self).__init__()
        self.gc1 = GraphConvolution_A(input_feat_dim, hidden_dim1, n_nodes, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)
        self.net = nn.Sequential(
			nn.Linear(n_nodes, 900),
			nn.ELU(),
			nn.Linear(900, 300),
			nn.ELU(),
            nn.Linear(300, 2 * 16*16),
		)
    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def encode_A(self,adj):
        adj = adj.to_dense()
        xy = adj.view(-1, adj.size()[0])#xy: 64*36864
        h = self.net(xy)#h: 64*32
        m, v = ut.gaussian_parameters(h, dim=1)
        return m, v #m: 64*16 v: 64*16
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
        


    def forward(self, adj):
        mu, logvar = self.encode_A(adj)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar

class GCNModelVAE_C(nn.Module):
    def __init__(self, input_feat_dim, n_nodes, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelVAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)
        self.net = nn.Sequential(
			nn.Linear(n_nodes, 900),
			nn.ELU(),
			nn.Linear(900, 300),
			nn.ELU(),
            nn.Linear(300, 2 * 16*16),

		)

        self.causal_layer = causal_layer(hidden_dim2/4, hidden_dim2/4)
        self.C = self.causal_layer.C
        self.label_net = nn.Sequential(
			nn.Linear(hidden_dim2, 32),
			nn.ELU(),
			nn.Linear(32, 64),
			nn.ELU(),
			nn.Linear(64, input_feat_dim),
		)
        self.hidden_dim2 = hidden_dim2


    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def encode_A(self,adj):
        adj = adj.to_dense()
        xy = adj.view(-1, adj.size()[0])#xy: 64*36864
        h = self.net(xy)#h: 64*32
        m, v = ut.gaussian_parameters(h, dim=1)
        return m, v #m: 64*16 v: 64*16
        
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
        z = z.view(-1, int(z.size()[1]/4),int(z.size()[1]/4))
        z = self.causal_layer(z)
        z = z.view(-1, self.hidden_dim2)
        return self.label_net(z), self.dc(z), mu, logvar
