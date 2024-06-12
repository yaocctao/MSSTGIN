import torch.nn as nn
import torch
import torch.nn.functional as F
from models.GCN import graph_constructor
from models.utils import LayerNorm

class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2,3,6,7]
        cout = int(cout/len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin,cout,(1,kern),dilation=(1,dilation_factor)))

    def forward(self,input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][...,-x[-1].size(3):]
        x = torch.cat(x,dim=1)
        return x

    
class LSTMoudle(nn.Module):
    def __init__(self, conf, supports):
        super(LSTMoudle,self).__init__()
        self.supports = supports
        self.residual_channels = 32
        self.conv_channels = 32
        self.new_dilation = 1
        self.seq_length = conf["output_length"]
        self.kernel_size = 7
        self.layers = 3
        self.receptive_field = self.layers*(self.kernel_size-1) + 1
        self.dropout = 0.3
        self.skip_channels = 64
        self.site_num = 227
        self.in_dim = 2
        self.end_channels = 128
        self.out_dim = 12
        self.layer_norm_affline = True
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.gcns = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        # self.temporalAttentions = nn.ModuleList()
        # self.spatialAttentions = nn.ModuleList()
        # self.global_TA_convs = nn.ModuleList()
        self.D = 8 * 64 // 8
        self.start_conv = nn.Conv2d(in_channels=self.in_dim,
                            out_channels=self.residual_channels,
                            kernel_size=(1, 1))
        kernel_size = 7
        self.dilation_exponential = 1
        self.gcn_true = True
        if self.dilation_exponential > 1:
            self.receptive_field = int(1+(kernel_size-1)*(self.dilation_exponential**self.layers-1)/(self.dilation_exponential-1))
        else:
            self.receptive_field = self.layers*(kernel_size-1) + 1

        
        for i in range(1):
            if self.dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(self.dilation_exponential**self.layers-1)/(self.dilation_exponential-1))
            else:
                rf_size_i = i*self.layers*(kernel_size-1)+1

            new_dilation = 1
            for j in range(1,self.layers+1):
                if self.dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(self.dilation_exponential**j-1)/(self.dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                # self.temporalAttentions.extend([TemporalAttention(conf,self.D,self.receptive_field, True, 0.99)])
                # self.spatialAttentions.extend([spatialAttention(conf,32, True, 0.99, False)])
                # self.global_TA_convs.append(nn.Conv2d(in_channels=self.D,out_channels=self.skip_channels,kernel_size=(1, self.receptive_field)))
                self.filter_convs.append(dilated_inception(self.residual_channels, self.conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception(self.residual_channels, self.conv_channels, dilation_factor=new_dilation))
        

                if self.seq_length>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=self.conv_channels,
                                                    out_channels=self.skip_channels,
                                                    kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=self.conv_channels,
                                                    out_channels=self.skip_channels,
                                                    kernel_size=(1, self.receptive_field-rf_size_j+1)))

                if self.gcn_true:
                    self.gconv1.append(graph_constructor(self.conv_channels, self.residual_channels, self.dropout, self.conv_channels,self.site_num, 20, 40))
                    self.gconv2.append(graph_constructor(self.conv_channels, self.residual_channels, self.dropout, self.conv_channels,self.site_num, 20, 40))
                    # self.gcns.append(GCN(32, 32, self.dropout, 32, self.supports))

                if self.seq_length > self.receptive_field:
                    self.norm.append(LayerNorm((self.residual_channels, self.site_num, self.seq_length - rf_size_j + 1),elementwise_affine = self.layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((self.residual_channels, self.site_num, self.receptive_field - rf_size_j + 1),elementwise_affine = self.layer_norm_affline))

                new_dilation *= self.dilation_exponential
                
        self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels,
                                             out_channels=self.end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels,
                                             out_channels=self.seq_length,
                                             kernel_size=(1,1),
                                             bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=self.residual_channels, out_channels=self.skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)

        else:
            # self.skip0 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skip0 = nn.Conv2d(in_channels=self.residual_channels, out_channels=self.skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=self.residual_channels, out_channels=self.skip_channels, kernel_size=(1, 1), bias=True)
        



    

    def forward(self, x):
        input = nn.functional.pad(x.permute(0,3,2,1),(self.receptive_field-self.seq_length,0,0,0))
        # x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        x = input
        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            # global_TA = self.temporalAttentions[0](input.permute(0,3,2,1),None,8,4,0.99).permute(0,3,2,1)
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_true:
                x = self.gconv1[i](x, is_tc_moudle = True) + self.gconv2[i](x, is_transpose = True, is_tc_moudle = True)
                # x = x.permute(0,3,2,1)
                # x = torch.reshape(x, shape=[-1, self.site_num, 32])
                # x = self.gcns[i](x)
                # x = torch.reshape(x,shape=[32, -1, self.site_num, 32])
                # x = x.permute(0,3,2,1)
                
                # global_SA = self.spatialAttentions[0](input.permute(0,3,2,1), None, 8, 4, 0.99).permute(0,3,2,1)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]

            x = self.norm[i](x)

        # print(x.shape)
        skip = self.skipE(x) + skip
        # print(skip.shape)
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        # print(x.shape)
        x = self.end_conv_2(x)
        # print(x.shape)

        return x