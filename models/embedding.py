import torch.nn as nn
import torch

class Embedding(nn.Module):
    def __init__(self, conf):
        super(Embedding, self).__init__()
        self.conf = conf
        self.total_len = conf["input_length"]+conf["output_length"]
        self.granularity = conf["granularity"]
        self.site_num = conf["site_num"]
        self.emb_size = conf["emb_size"]
        self.position_emb = nn.Embedding(self.site_num, self.emb_size)
        self.week_emb = nn.Embedding(7, self.emb_size)
        self.minute_emb = nn.Embedding(24 * 60 //self.granularity, self.emb_size)
        self.register_buffer('site_num_tensor', torch.IntTensor([[i for i in range(self.site_num)]]))
    
    def forward(self, Dow, M):
        #embedding Dow,M,position
        self.p_emd = self.position_emb(self.site_num_tensor)
        self.p_emd = self.p_emd.reshape(1,self.site_num, self.emb_size)
        self.p_emd = self.p_emd.unsqueeze(0)

        self.w_emd = self.week_emb(Dow.reshape(-1, self.site_num))
        self.w_emd = self.w_emd.reshape(-1,self.total_len,self.site_num, self.emb_size)

        self.m_emd = self.minute_emb(M.reshape(-1, self.site_num))
        self.m_emd = self.m_emd.reshape(-1,self.total_len,self.site_num, self.emb_size)

        timestamp = [self.w_emd, self.m_emd]
        position = self.p_emd

        return timestamp, position