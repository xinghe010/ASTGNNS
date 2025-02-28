import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch_scatter import scatter_mean, scatter_sum
from torch_geometric.nn import global_mean_pool, GCNConv, SAGEConv, global_max_pool, SAGPooling


class MLPBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 activation=F.relu,
                 bias=True,
                 batch=False,
                 drop=False):
        super(MLPBlock, self).__init__()
        self.activation = activation
        self.bias = bias
        self.batch = batch
        self.drop = drop
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)
        self.rnn = nn.LSTM(in_channels, out_channels, bias=bias)
        if batch:
            self.BN = nn.BatchNorm1d(out_channels)
        if self.drop:
            self.drop = nn.Dropout(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        if self.activation == F.relu:
            nn.init.kaiming_normal_(self.lin.weight, nonlinearity="relu")
        elif self.activation == F.leaky_relu:
            nn.init.kaiming_normal_(self.lin.weight)
        else:
            nn.init.xavier_normal_(self.lin.weight)
        if self.bias:
            nn.init.zeros_(self.lin.bias)

    def forward(self, x):
        x = self.lin(x)
        if self.batch and x.size()[0] > 1:
            x = self.BN(x)
        if self.drop:
            x = self.drop(x)
        x = self.activation(x)
        return x


class Initialization(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Initialization, self).__init__()
        self.embedding = nn.Embedding(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.embedding.weight)

    def forward(self, x):
        indices = torch.argmax(x, dim=1)
        output = self.embedding(indices)
        return output




class DAGEmbedding(nn.Module):
    def __init__(self, node_out_channels, layers):
        super(DAGEmbedding, self).__init__()
        self.K = layers
        self.F_T = [
            MLPBlock(3 * node_out_channels,
                     node_out_channels,
                     batch=True) for i in range(layers)
        ]
        self.F_M = [
            MLPBlock(3 * node_out_channels,
                     node_out_channels,
                     batch=True) for i in range(layers)
        ]
        self.F_B = [
            MLPBlock(3 * node_out_channels,
                     node_out_channels,
                     batch=True) for i in range(layers)
        ]
        self.F_1 = [
            MLPBlock(3 * node_out_channels,
                     1,
                     batch=True) for i in range(layers)
        ]
        self.F_trans_TW = [
            MLPBlock(node_out_channels,
                     node_out_channels,
                     batch=True) for i in range(layers)
        ]

    def forward(self, x, term_walk_index):
        N = x.size()[0]
        for i in range(self.K):
            term_walk_feat = torch.cat([x[term_walk_index[0]],
                                        x[term_walk_index[1]],
                                        x[term_walk_index[2]]], dim=1)
            score = self.F_1[i](term_walk_feat)
            att = F.softmax(score, dim=0)
            trans_T = self.F_T[i](term_walk_feat)
            m_T = scatter_mean(att * trans_T,
                               index=term_walk_index[0],
                               dim=0, dim_size=N)
            trans_M = self.F_M[i](term_walk_feat)
            m_M = scatter_mean(att * trans_M,
                               index=term_walk_index[1],
                               dim=0, dim_size=N)

            trans_B = self.F_B[i](term_walk_feat)
            m_B = scatter_mean(att * trans_B,
                               index=term_walk_index[2],
                               dim=0, dim_size=N)

            m_TW = m_T + m_M + m_B
            m_TW = self.F_trans_TW[i](m_TW)
            x = x + m_TW
        return x


class DAGPooling(nn.Module):
    def __init__(self, node_out_channels):
        super(DAGPooling, self).__init__()
        self.lin1 = MLPBlock(node_out_channels, 1, bias=False)
        self.SAGPool = SAGPooling(node_out_channels, ratio=0.9, GNN=GCNConv)

    def forward(self, x, batch_index, child_index, parent_index):

        x1, edge_index1, _, batch1, perm, score = self.SAGPool(x, child_index, None, batch_index)
        x2, edge_index2, _, batch2, perm, score = self.SAGPool(x, parent_index, None, batch_index)
        #x3, edge_index3, _, batch3, perm, score = self.SAGPool(x1, edge_index1, None, batch1)
        #x4, edge_index4, _, batch4, perm, score = self.SAGPool(x2, edge_index2, None, batch1)
        output1 = torch.cat([global_mean_pool(x1, batch1), global_max_pool(x1, batch1)], dim=1)
        output2 = torch.cat([global_mean_pool(x2, batch2), global_max_pool(x2, batch2)], dim=1)
        #output3 = global_mean_pool(x2, batch2)

        #node_score = self.lin1(x, treelet_index_reverse)
        #indices = torch.argsort(node_score, dim=0, descending=True)
        #k = int(math.ceil(0.85 * x.size()[0]))
        #indices = indices[:k]
        #for i in range(x.size()[0]):
        #    if i in indices:
        #        x[i] = x[i]
        #    else:
        #        x[i] = 0
        #output = global_mean_pool(x, batch_index)
        output = output1 + output2
        return output


class Classifier(nn.Module):
    def __init__(self, node_out_channels):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            MLPBlock(4 * node_out_channels, node_out_channels,
                     batch=True),
            MLPBlock(node_out_channels, 2, activation=lambda x: x,
                     batch=True))

    def forward(self, conj_batch, prem_batch):
        x_concat = torch.cat([conj_batch, prem_batch], dim=1)
        pred_y = self.classifier(x_concat)
        return pred_y


class PremiseSelectionModel(nn.Module):
    def __init__(self, node_in_channels, node_out_channels, layers):
        super(PremiseSelectionModel, self).__init__()
        self.initial = Initialization(node_in_channels, node_out_channels)
        self.dag_emb = DAGEmbedding(node_out_channels, layers)
        self.pooling = DAGPooling(node_out_channels)
        self.classifier = Classifier(node_out_channels)
        self.criterion = nn.CrossEntropyLoss()
        self.corrects = None

    def forward(self, batch):
        h_s = self.initial(batch.x_s)
        h_t = self.initial(batch.x_t)
        h_s = self.dag_emb(h_s, batch.term_walk_index_s)
        h_t = self.dag_emb(h_t, batch.term_walk_index_t)
        h_g_s = self.pooling(h_s, batch.x_s_batch, batch.child_index_s, batch.parent_index_s)
        h_g_t = self.pooling(h_t, batch.x_t_batch, batch.child_index_t, batch.parent_index_t)
        pred_y = self.classifier(h_g_s, h_g_t)
        pred_label = torch.max(pred_y, dim=1)[1]
        self.corrects = (pred_label == batch.y).sum().cpu().item()
        loss = self.criterion(pred_y, batch.y)

        return loss, batch.y, pred_label
