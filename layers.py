import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import numpy

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphConvolution_A(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, n_node, dropout=0., act=F.relu):
        super(GraphConvolution_A, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.input = Parameter(torch.FloatTensor(n_node, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.input)

    def forward(self, adj):
        input = F.dropout(self.input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class causal_layer(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(causal_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.C = Parameter(torch.FloatTensor(int(in_features), int(out_features)))
        self.reset_parameters()

    def causal_net(self, input, input_dim, output_dim):
        net = nn.Sequential(
			nn.Linear(input_dim, 8),
			nn.ELU(),
			nn.Linear(8, output_dim),
		)
        out = net(input)
        return out

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.C)

    def forward(self, input):
        input = F.dropout(input, self.dropout, self.training)
        output = torch.matmul(input, self.C)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class causal_layer_reperameter(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(causal_layer_reperameter, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.C = Parameter(torch.FloatTensor(int(in_features), int(out_features)))
        self.reset_parameters()

    def causal_net(self, input, input_dim, output_dim):
        net = nn.Sequential(
			nn.Linear(input_dim, 8),
			nn.ELU(),
			nn.Linear(8, output_dim),
		)
        out = net(input)
        return out

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.C)

    def forward_mu(self, input):
        input = F.dropout(input, self.dropout, self.training)
        output = torch.matmul(input, self.C)
        output = self.act(output) 
        return output

    def forward_var(self, input):
        input = torch.diag_embed(input)#100*16*16
        input = F.dropout(input, self.dropout, self.training)
        output = torch.matmul(input, self.C)
        output = torch.matmul(self.C.T, output)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
