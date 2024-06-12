import torch.nn as nn
import torch, os
from torch_geometric.nn import GCNConv
import numpy as np
from models.utils import FC


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, bias=False):
        super(GraphConvolution, self).__init__()
        if dropout:
            self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        nn.init.xavier_uniform_(self.weight)  # xavier初始化，就是论文里的glorot初始化
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, inputs, adj):
        # inputs: (N, n_channels), adj: sparse_matrix (N, N)
        if self.dropout:
            support = torch.mm(self.dropout(inputs), self.weight)
        else:
            support = torch.mm(inputs, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCN(nn.Module):
    def __init__(self, n_features, hidden_dim, dropout, n_classes, supports):
        super(GCN, self).__init__()
        self.gc1 = GCNConv(n_features, hidden_dim, dropout, bias=True)
        self.gc2 = GCNConv(hidden_dim, n_classes, bias=False)
        self.relu = nn.ReLU()
        # self.res_c_layer1 = nn.Linear(n_features, hidden_dim)
        # self.res_c_layer2 = nn.Linear(n_features, hidden_dim)
        # self.res_c_layer3 = nn.Linear(n_features, hidden_dim)
        # self.res_c_layer4 = nn.Linear(n_features, hidden_dim)
        self.register_buffer('supports', supports)
    
    def forward(self, inputs):
        x = inputs
        # res_c1 = self.res_c_layer1(inputs)
        # res_c3 = self.res_c_layer3(inputs)
        # res_c4 = self.res_c_layer4(inputs)
        # res_c2 = self.res_c_layer2(inputs)
        x = self.relu(self.gc1(x, self.supports))+x
        x = self.gc2(x, self.supports)+x
        return x

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()
class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)

class mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha


    def forward(self,x,adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for _ in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho = self.mlp(ho)
        return ho

class graph_constructor(nn.Module):
    def __init__(self, n_features, hidden_dim, dropout, n_classes, nnodes, k, dim, alpha=3, static_feat=None):
        super(graph_constructor, self).__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat
        # self.idx_linear = FC(32*12*227*32, units=[self.hidden_dim, self.hidden_dim], activations=[torch.nn.ReLU(), None],bn=True, use_bias=True, drop=None,bn_decay=0.99)

        self.gcn = mixprop(self.n_features, self.hidden_dim, 2, dropout, 0.05)
        self.gcn1 = mixprop(self.n_features, self.hidden_dim, 2, dropout, 0.05)
        self.gcn2 = mixprop(self.n_features, self.hidden_dim, 2, dropout, 0.05)

    def forward(self, x, is_transpose=False,is_tc_moudle=False):

        # idx = torch.tensor(np.random.permutation(range(self.nnodes))).to(x.device)
        idx = torch.tensor(range(self.nnodes)).to(x.device)
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0)) - torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = nn.functional.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(x.device)
        mask.fill_(float('0'))
        s1,t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        if is_transpose:
            adj = adj.transpose(1,0)

        if is_tc_moudle:
            x = self.gcn(x,adj)
        else:
            x = torch.reshape(x, shape=[-1, 12, self.nnodes, self.n_features])
            x = x.permute(0, 3, 2, 1)
            x = self.gcn(x,adj)
            x = x.permute(0, 3, 2, 1)
            x = torch.reshape(x, shape=[-1, self.nnodes, self.n_features])

        return x