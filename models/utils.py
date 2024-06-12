import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.nn import init
import numbers

class FC(nn.Module):
    def __init__(self, in_features,units, activations, bn, use_bias=True, drop=None, bn_decay=0.1):
        super(FC, self).__init__()
        self.fcs = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.bn = bn
        self.dropouts = nn.ModuleList()


        if isinstance(units, int):
            units = [units]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            activations = list(activations)
        assert type(units) == list

        for num_unit, activation in zip(units, activations):
            
            if drop is not None:
                self.dropouts.append(nn.Dropout(p=drop))

            # self.fcs.append(nn.Linear(in_features=in_features, out_features=num_unit, bias=use_bias))
            self.fcs.append(nn.Conv2d(in_channels=in_features, out_channels=num_unit, kernel_size=[1,1],stride=[1,1], bias=use_bias,padding='valid'))
            in_features = num_unit

            if activation is not None:
                if bn:
                    self.batch_norms.append(nn.BatchNorm2d(num_features=num_unit, momentum=bn_decay))
                self.activations.append(activation)
                

    def forward(self, x,bn_decay):
        x = x.permute(0,3,2,1)
        for i,fc in enumerate(self.fcs):
            if len(self.dropouts) !=0 and self.training:
                x = self.dropouts[i](x)

            x = fc(x)

            if i < len(self.activations):
                if self.bn:
                    x = self.batch_norms[i](x)
                x = self.activations[i](x)
        x = x.permute(0,3,2,1)
        return x


class CustomBatchNorm(nn.Module):

    def __init__(self, num_features, eps=1e-3, momentum=0.1):
        super(CustomBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # 初始化可学习参数 gamma 和 beta
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        # 初始化运行时均值和方差
        # self.running_mean = torch.zeros(num_features)
        # self.running_var = torch.ones(num_features)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))


    def forward(self, x):
        x = x.permute(0,3,2,1)
        moment_dims = list(range(len(x.shape)-1))
        if self.training:
            # 计算当前批次的均值和方差
            batch_mean = torch.mean(x, dim=moment_dims)
            batch_var = torch.var(x, dim=moment_dims)

            # 更新运行时均值和方差
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

            # 规范化
            x_norm = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        else:
            # 测试时使用运行时均值和方差进行规范化
            x_norm = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        # 应用可学习参数 gamma 和 beta
        return (self.gamma * x_norm + self.beta).permute(0,3,2,1)
    
    def extra_repr(self) -> str:
        return f'num_features={self.num_features}, eps={self.eps}, momentum={self.momentum}'
    

def metric(pred, label):
    
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = np.not_equal(label, 0.0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)

        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label.astype(np.float32))
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
        cor = np.mean(np.multiply((label - np.mean(label)),
                                  (pred - np.mean(pred)))) / (np.std(pred) * np.std(label))
        sse = np.sum((label - pred) ** 2)
        sst = np.sum((label - np.mean(label)) ** 2)
        r2 = 1 - sse / sst  # r2_score(y_actual, y_predicted, multioutput='raw_values')
    return mae, rmse, mape


class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input):
        return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)