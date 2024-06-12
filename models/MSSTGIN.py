import torch.nn as nn
import torch
import torch.nn.functional as F
from models.utils import FC
from models.embedding import Embedding
from models.utils import CustomBatchNorm
from models.SemanticEnhancement import SpeedTimeEnhanceEncoder
from models.LSTMoudle import LSTMoudle
from models.GSTMoudle import STEmbedding, GSTMoudle, fusionGate
import pandas as pd
import numpy as np
from models.fusion import AFF



class MSSTGIN(nn.Module):
    def __init__(self, conf, mean, std):
        super(MSSTGIN, self).__init__()
        self.conf = conf
        self.mean = mean
        self.std = std
        self.emb_size = conf["emb_size"]
        self.site_num = conf["site_num"]
        self.features = conf["features"]
        self.input_len = conf["input_length"]
        self.file_adj = conf["file_adj"]
        self.num_heads = conf["num_heads"]
        self.residual_channels = 32
        self.conv_channels = 32
        self.new_dilation = 1
        self.seq_length = conf["output_length"]
        self.kernel_size = 7
        self.layers = 3
        self.receptive_field = self.layers*(self.kernel_size-1) + 1

        self.fc = FC(self.features, units=[self.emb_size, self.emb_size], activations=[torch.nn.ReLU(), None],bn=True, use_bias=True, drop=None,bn_decay=0.99)
        # self.xfc = FC(self.features, units=[self.emb_size, self.emb_size], activations=[torch.nn.ReLU(), None],bn=True, use_bias=True, drop=None,bn_decay=0.99)
        self.embdding = Embedding(conf)
        self.speedTimeEnhanceEncoder = SpeedTimeEnhanceEncoder(conf)
        # self.pre_fusion = AFF(self.seq_length)
        # self.fusion = AFF(self.emb_size)
        # self.filter_convs = nn.ModuleList()
        # self.gate_convs = nn.ModuleList()
        # self.filter_convs.append(dilated_inception(self.residual_channels, self.conv_channels, dilation_factor=self.new_dilation))
        # self.gate_convs.append(dilated_inception(self.residual_channels, self.conv_channels, dilation_factor=self.new_dilation))
        self.STEmbedding = STEmbedding(conf, self.emb_size, bn=True, bn_decay=0.99)
        self.adj = self.adjecent()
        self.supports = self.get_supports()
        self.lstMoudle = LSTMoudle(conf,supports = self.supports)
        self.gstMoudle = GSTMoudle(conf, self.emb_size, bn=True, bn_decay=0.99, supports = self.supports,adj = self.adj)
        # self.bridgeTrans = BridgeTransformer(conf, self.emb_size, bn=True, bn_decay=0.99)
        self.pre = FC(self.emb_size, units=[self.emb_size, 1], activations=[None, None],bn=True, use_bias=True, drop=0.1,bn_decay=0.99)



    def forward(self, X, DoW, D, H, M, XALL, bn_decay):
        # if self.features <= 1:
        #     X = X.unsqueeze(-1)
        #     X_All = X_All.unsqueeze(-1)

        #embedding D,M,position
        timestamp, position = self.embdding(DoW, M)

        XALL = self.fc(XALL,bn_decay)
        # X_fc = self.fc(X,bn_decay)

        speed = X.permute(0,2,3,1)
        speed = speed.reshape(-1, self.features, self.input_len)
        speed = self.speedTimeEnhanceEncoder(speed)


        STE = self.STEmbedding(position, timestamp,bn_decay)
        encoder_outs = self.gstMoudle(speed, XALL, STE , bn_decay)

        # X = self.bridgeTrans(encoder_outs, encoder_outs + STE[:, :self.input_len], STE[:, self.input_len:] + X_All[:,self.input_len:], self.num_heads, self.emb_size // self.num_heads,bn_decay)
        # X = self.bridgeTrans(encoder_outs, encoder_outs + STE[:, :self.input_len], speed, self.num_heads, self.emb_size // self.num_heads,bn_decay)
        # X = self.bridgeTrans(encoder_outs, encoder_outs + STE[:,self.input_len:],speed + XALL[:,self.input_len:], self.num_heads, self.emb_size // self.num_heads,bn_decay)
        # X = self.gstMoudle.dynamic_decoding(encoder_outs, STE[:, self.input_len:])
        # tc_pre = self.lstMoudle(torch.cat([encoder_outs,speed],dim=-1))
        # encoder_outs = fusionGate(encoder_outs,X)
        tc_pre = self.lstMoudle(encoder_outs)
        # tc_pre = self.lstMoudle(speed)
        #w/o tc_pre = self.lstMoudle(encoder_outs,speed)

        # pre = self.pre(encoder_outs,bn_decay)

        pre = tc_pre
        # pre = fusionGate(pre,tc_pre)
        # pre = self.pre_fusion(pre,tc_pre)
        pre = pre * (self.std) + self.mean
        pre = pre.squeeze(-1).transpose(1, 2)

        return pre
    
    def normalization(self,data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    def adjecent(self):
        '''
        :return: adj matrix
        '''
        data = pd.read_csv(filepath_or_buffer=self.file_adj)
        adj = np.zeros(shape=[self.site_num, self.site_num], dtype=np.int32)
        for line in data[['from_id', 'to_id']].values:
            adj[int(line[0])][int(line[1])] = 1
        #adj 标准化
        adj = self.normalization(adj)
        # adj = adj + np.eye(self.site_num)
        return torch.from_numpy(adj)
    
    
    def get_supports(self):
        data = pd.read_csv(filepath_or_buffer=self.file_adj)
        edge_index = [[line[0],line[1]]for line in data[['from_id', 'to_id']].values]
        edge_index.extend([[i,i]for i in range(self.site_num)])
        edge_index = torch.tensor(edge_index,dtype=torch.long)
        return edge_index.t().contiguous()

    def encoder():
        pass

    def adjust_bn_momentum(self, momentum):
        for m in self.modules():
            if isinstance(m, CustomBatchNorm):
                m.momentum = momentum


        
