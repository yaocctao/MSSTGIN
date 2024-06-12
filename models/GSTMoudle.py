import torch.nn as nn
from models.utils import FC
import torch
from models.GCN import graph_constructor
from models.fusion import AFF


class GSTMoudle(nn.Module):
    def __init__(self, conf,D, bn, bn_decay, supports, adj):
        super(GSTMoudle, self).__init__()
        self.conf = conf
        self.emb_size = conf["emb_size"]
        self.num_heads = conf["num_heads"]
        self.input_len = conf["input_length"]
        self.spatioTemporal = SpatioTemporal(conf, bn, bn_decay,supports,adj)

    def forward(self, speed, speed_all,STE, bn_decay):
        encoder_outs = self.spatioTemporal(speed, STE[:, :self.input_len], speed_all, bn_decay=bn_decay)
        return encoder_outs

class STEmbedding(nn.Module):
    def __init__(self, conf,D, bn, bn_decay):
        super(STEmbedding, self).__init__()
        self.conf = conf
        self.emb_size = conf["emb_size"]
        self.features = conf["features"]
        self.input_len = conf["input_length"]
        self.site_num = conf["site_num"]
        self.SE = FC(self.emb_size, units=[D, D], activations=[torch.nn.ReLU(), None],bn=bn, bn_decay=bn_decay)
        self.TE = FC(self.emb_size*2, units=[D, D], activations=[torch.nn.ReLU(), None],bn=bn, bn_decay=bn_decay)

    
    def forward(self, SE, TE, bn_decay):
        # spatial embedding
        SE = self.SE(SE,bn_decay)
        # temporal embedding
        TE = torch.cat((TE), axis=-1)
        TE = self.TE(TE,bn_decay)
        return torch.add(SE, TE)

class TemporalAttention(nn.Module):
    def __init__(self, conf,D, bn, bn_decay,mask=True):
        super(TemporalAttention, self).__init__()
        self.conf = conf
        self.emb_size = conf["emb_size"]
        self.num_heads = conf["num_heads"]
        self.is_mask = mask
        self.query = FC(self.emb_size, units=D, activations=torch.nn.ReLU(),bn=bn, bn_decay=bn_decay)
        self.key = FC(self.emb_size, units=D, activations=torch.nn.ReLU(),bn=bn, bn_decay=bn_decay)
        self.value = FC(self.emb_size, units=D, activations=torch.nn.ReLU(),bn=bn, bn_decay=bn_decay)
        self.fc = FC(D, units=[D, D], activations=[torch.nn.ReLU(), None],bn=bn, bn_decay=bn_decay)
        if self.is_mask:
            self.batch_size = conf["batch_size"]
            num_step = conf["input_length"]
            self.N = conf["site_num"]
            mask = torch.ones(num_step, num_step)
            mask = torch.triu(mask)
            mask = mask.unsqueeze(0).unsqueeze(0)
            # mask = mask.repeat(self.num_heads * batch_size, N, 1, 1)
            mask = mask.type(torch.bool)
            self.register_buffer("mask",mask)
        

    def forward(self,X, STE, K, d,bn_decay):

        query = self.query(X,bn_decay)
        key = self.key(X,bn_decay)
        value = self.value(X,bn_decay)

        # [K * batch_size, num_step, N, d]
        query = torch.cat(torch.chunk(query, K, dim=-1), dim=0)
        key = torch.cat(torch.chunk(key, K, dim=-1), dim=0)
        value = torch.cat(torch.chunk(value, K, dim=-1), dim=0)

        # query: [K * batch_size, N, num_step, d]
        # key:   [K * batch_size, N, d, num_step]
        # value: [K * batch_size, N, num_step, d]
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)

        # [K * batch_size, N, num_step, num_step]
        attention = torch.matmul(query, key)
        attention /= (d ** 0.5)

        # mask attention score
        if self.is_mask:
            attention = torch.where(
                self.mask.repeat(self.num_heads * self.batch_size, self.N, 1, 1), attention, torch.tensor(-2 ** 15 + 1))
            
        # softmax
        attention = torch.nn.functional.softmax(attention, dim=-1)
        # [batch_size, num_step, N, D]
        X = torch.matmul(attention,value)
        X = X.permute(0, 2, 1, 3)
        X = torch.cat(torch.chunk(X, K, dim=0), dim=-1)
        X = self.fc(X,bn_decay)
        return X

class spatialAttention(nn.Module):
    def __init__(self, conf,D, bn, bn_decay,mask=True,adj=None):
        super(spatialAttention, self).__init__()
        self.conf = conf
        self.emb_size = conf["emb_size"]
        self.num_heads = conf["num_heads"]
        self.is_mask = mask
        self.adj = adj
        self.query = FC(self.emb_size, units=D, activations=torch.nn.ReLU(),bn=bn, bn_decay=bn_decay)
        self.key = FC(self.emb_size, units=D, activations=torch.nn.ReLU(),bn=bn, bn_decay=bn_decay)
        self.value = FC(self.emb_size, units=D, activations=torch.nn.ReLU(),bn=bn, bn_decay=bn_decay)
        self.fc = FC(D, units=[D, D], activations=[torch.nn.ReLU(), None],bn=bn, bn_decay=bn_decay)
        if self.is_mask:
            mask = self.adj.type(torch.bool)
            self.register_buffer("mask",mask)
            
    def forward(self,X, STE, K, d ,bn_decay):
        query = self.query(X,bn_decay)
        key = self.key(X,bn_decay)
        value = self.value(X,bn_decay)

        # [K * batch_size, num_step, N, d]
        query = torch.cat(torch.chunk(query, K, dim=-1), dim=0)
        key = torch.cat(torch.chunk(key, K, dim=-1), dim=0)
        value = torch.cat(torch.chunk(value, K, dim=-1), dim=0)

        # [K * batch_size, num_step, N, N]
        attention = torch.matmul(query, key.transpose(-2, -1))
        attention /= (d ** 0.5)
            
        # mask attention score
        # if self.is_mask:
        #     attention = torch.where(
        #         self.mask, attention, torch.tensor(-2 ** 15 + 1))
            
        # softmax
        attention = torch.nn.functional.softmax(attention, dim=-1)
        # [batch_size, num_step, N, D]
        X = torch.matmul(attention, value)

        X = torch.cat(torch.chunk(X, K, dim=0), dim=-1)
        X = self.fc(X,bn_decay)
        return X

def fusionGate(x,y):
        '''
        :param x: [-1, len, site, dim]
        :param y: [-1, len, site, dim]
        :return: [-1, len, site, dim]
        '''
        z = torch.sigmoid(torch.mul(x, y))
        h = torch.add(torch.mul(z, x), torch.mul(1 - z, y))
        return h

class SpatioTemporal(nn.Module):
    def __init__(self, conf, bn, bn_decay,supports,adj):
        super(SpatioTemporal, self).__init__()
        self.conf = conf
        self.input_length = conf["input_length"]
        self.num_heads = conf["num_heads"]
        self.emb_size = conf["emb_size"]
        self.site_num = conf["site_num"]
        self.dropout = conf["dropout"]
        self.batch_size = conf["batch_size"]
        self.hidden_size = conf["hidden_size"]
        self.hidden_layer = conf["hidden_layer"]
        self.num_blocks = conf["num_blocks"]
        self.site_num = conf['site_num']
        self.supports = supports
        self.adj = adj
        self.D = self.num_heads * self.emb_size // self.num_heads
        self.temporalAttentions = nn.ModuleList()
        self.spatialAttentions = nn.ModuleList()
        self.temporalAttentions.extend([TemporalAttention(conf,self.D, bn, bn_decay) for _ in range(self.num_blocks)])
        self.spatialAttentions.extend([spatialAttention(conf,self.D, bn, bn_decay,True,self.adj) for _ in range(self.num_blocks)])
        # self.lstm = nn.LSTM(input_size=self.emb_size, hidden_size=self.emb_size, num_layers=self.hidden_layer, batch_first=True, dropout=1-self.dropout)
        self.gcn1 = graph_constructor(self.emb_size * 1, self.emb_size, self.dropout, self.emb_size * 1,self.site_num, 20, 40)

        # self.gcn2 = graph_constructor(self.emb_size * 1, self.emb_size, self.dropout, self.emb_size * 1,self.site_num, 20, 40)
        # self.gcn = GCN(self.emb_size * 1, self.emb_size, self.dropout, self.emb_size * 1, self.supports)
        # self.HT_XL_fusion = AFF(self.emb_size)
        # self.HS_XS_fusion = AFF(self.emb_size)
        # self.HS_HT_fusion = AFF(self.emb_size)


    def forward(self, speed=None, STE=None, speed_all=None,mask=True, bn_decay=None):
        X = speed
        # X_All = speed_all
        HT = X

        for i in range(len(self.temporalAttentions)):
            # HT = self.temporalAttentions[i](HT + X_All[:,:self.input_length] + STE, STE, self.num_heads, self.emb_size // self.num_heads, bn_decay)
            HT = self.temporalAttentions[i](HT + STE, STE, self.num_heads, self.emb_size // self.num_heads, bn_decay)
            # HT = self.temporalAttentions[i](HT, STE, self.num_heads, self.emb_size // self.num_heads, bn_decay)

        # XL = (X + X_All[:, :self.input_length]).permute(0, 2, 1, 3)
        # XL = (X).permute(0, 2, 1, 3)
        # XL = torch.reshape(XL, shape=[-1, self.input_length, self.emb_size])
        # XL, (hn, cn) = self.lstm(XL)

        # XL = torch.reshape(XL, shape=[self.batch_size, self.site_num, self.input_length, self.emb_size])
        # XL = XL.permute(0, 2, 1, 3)
        # HT = fusionGate(HT, XL)
        # HT = self.HT_XL_fusion(HT, XL)

        HS = X
        for i in range(len(self.spatialAttentions)):
            # HS = self.spatialAttentions[i](HS + X_All[:,self.input_length:], STE, self.num_heads, self.emb_size // self.num_heads,bn_decay)
            HS = self.spatialAttentions[i](HS, STE, self.num_heads, self.emb_size // self.num_heads,bn_decay)
        # XS = torch.reshape(X + X_All[:,self.input_length:] + STE, shape=[-1, self.site_num, self.emb_size * 1])
        # XS = torch.reshape(X + X_All[:,self.input_length:], shape=[-1, self.site_num, self.emb_size * 1])
        XS = torch.reshape(X + STE, shape=[-1, self.site_num, self.emb_size * 1])
        # XS = torch.reshape(X, shape=[-1, self.site_num, self.emb_size * 1])
        XS = self.gcn1(XS)
        # XS = self.gcn(XS)
        XS = torch.reshape(XS, shape=[-1, self.input_length, self.site_num, self.emb_size])
        HS = fusionGate(HS, XS)
        # HS = self.HS_XS_fusion(HS, XS)

        # H = HS
        H = fusionGate(HS, HT)
        # H = self.HS_HT_fusion(HS, HT)
        
        X = torch.add(X, H)

        return X
