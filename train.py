from __future__ import division
from __future__ import print_function

import argparse
from statistics import mode
import time

import numpy as np
import scipy.sparse as sp
import torch
from torch import optim

from model import GCNModelVAE, GCNModelVAE_A
from optimizer import loss_function
from utils import load_data, mask_test_edges, preprocess_graph, get_roc_score, _h_A
import random
import torch.nn.functional as F

import warnings

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='texas', help='type of dataset.')
parser.add_argument('--alpha', type=int, default=10e-10, help='Parameter one')
parser.add_argument('--beta', type=int, default=0.05e-10, help='Parameter two')
parser.add_argument('--gamma', type=int, default=1000e-10, help='Parameter three')


args = parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def gae_for(args, feature_in = None):
    print("Using {} dataset".format(args.dataset_str))
    adj, features = load_data(args.dataset_str)


    if feature_in != None:
        features = feature_in

    n_nodes, feat_dim = features.shape


    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj  # ndarray
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)  # ?
    adj_orig.eliminate_zeros()
    
    adj_train, train_edges, val_edges , val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train

    # Some preprocessing: 
    adj_norm = preprocess_graph(adj)  # ndarray
    adj_label = adj_train + sp.eye(adj_train.shape[0])  # (1,1)=1
    adj_label = torch.FloatTensor(adj_label.toarray())

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    pos_weight = torch.FloatTensor([pos_weight])
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)


    model = GCNModelVAE(feat_dim, n_nodes, args.hidden1, args.hidden2, args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    mse = torch.nn.MSELoss()

    hidden_emb = None


    for epoch in range(args.epochs):
        t = time.time()    
        model.train()
        optimizer.zero_grad()

        re_label, recovered, mu, logvar, z, z2 = model(features, adj_norm)  # recovered: reconstructed A, adj_norm: norm, tensor

        dag_param = torch.eye(model.C.size()[0])-model.C.inverse() #causal matrix
        h_a = _h_A(dag_param, dag_param.size()[0]) #h_a=tr((I+A*A/16)^16)-16 A=dag_param

        label_loss = mse(re_label,features)

        loss_rec, KLD = loss_function(preds=recovered, labels=adj_label,
                             mu=mu, logvar=logvar, n_nodes=n_nodes,
                             norm=norm, pos_weight=pos_weight)
        

        loss_rec = loss_rec*args.alpha
        h_a = h_a*args.beta
        label_loss = label_loss*args.gamma
        loss = loss_rec + h_a + label_loss

        loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        # valid
        hidden_emb = mu.data.numpy()
        roc_curr, ap_curr = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)
        print("Epoch:", '%04d' % (epoch + 1), 
                "train_loss=", "{:.5f}".format(cur_loss),
              "val_ap=", "{:.5f}".format(ap_curr),
              #'causal_loss : ' + str(causal_loss.item()),
              "time=", "{:.5f}".format(time.time() - t)
              )


    print("Optimization Finished!")
    roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))


def new_func(adj_train):
    sp.eye(adj_train.shape[0])


if __name__ == '__main__':
    setup_seed(100)
    gae_for(args)