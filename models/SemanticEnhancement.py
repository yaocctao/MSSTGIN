import torch.nn as nn
import torch

class SpeedTimeEnhanceEncoder(nn.Module):
    def __init__(self, conf):
        super(SpeedTimeEnhanceEncoder, self).__init__()
        self.conf = conf
        self.emb_size = conf["emb_size"]
        self.out_channels = self.emb_size
        self.features = conf["features"]
        self.input_len = conf["input_length"]
        self.site_num = conf["site_num"]
        self.cov = nn.Conv1d(in_channels=self.features, out_channels=self.out_channels, kernel_size=1, padding="same")
        nn.init.normal_(self.cov.weight, mean=0.0, std=0.01)
        self.reverse_cov = nn.Conv1d(in_channels=self.features, out_channels=self.out_channels, kernel_size=1, padding="same")
        nn.init.normal_(self.reverse_cov.weight, mean=0.0, std=0.01)
        #w/o use STGIN SpeedTimeEnhanceEncoder
        # self.cov = nn.Conv1d(in_channels=self.features, out_channels=self.out_channels, kernel_size=3, padding="same")
        # nn.init.normal_(self.cov.weight, mean=0.0, std=0.01)
        # self.reverse_cov = nn.Conv1d(in_channels=self.features, out_channels=self.out_channels, kernel_size=3, padding="same")
        # nn.init.normal_(self.reverse_cov.weight, mean=0.0, std=0.01)
        # self.cov2 = nn.Conv1d(in_channels=self.features, out_channels=self.out_channels, kernel_size=1, padding="same")
        # nn.init.normal_(self.cov2.weight, mean=0.0, std=0.01)
        # forward
        # hidden = self.cov(speed)
        # reverse_hidden = self.reverse_cov(torch.flip(speed,dims=[2]))
        # hidden2 = self.cov2(speed)

        # reverse_hidden = torch.flip(reverse_hidden, dims=[2])
        # reverse_hidden = torch.multiply(reverse_hidden,torch.sigmoid(reverse_hidden))
        # hidden = torch.multiply(hidden,torch.sigmoid(hidden))

        # hidden_speed =hidden +reverse_hidden + hidden2
        # hidden_speed = hidden_speed.reshape(-1, self.site_num, self.emb_size, self.input_len)
        # hidden_speed = hidden_speed.permute(0,3,1,2)

    def forward(self, speed):
        hidden = self.cov(speed)
        reverse_hidden = self.reverse_cov(torch.flip(speed,dims=[2]))

        reverse_hidden = torch.flip(reverse_hidden, dims=[2])
        reverse_hidden = torch.multiply(reverse_hidden,torch.sigmoid(reverse_hidden))
        hidden = torch.multiply(hidden,torch.sigmoid(hidden))

        hidden_speed = 0.5*hidden + 0.5*reverse_hidden
        hidden_speed = hidden_speed.reshape(-1, self.site_num, self.emb_size, self.input_len)
        hidden_speed = hidden_speed.permute(0,3,1,2)
        
        # hidden = self.cov(speed)
        # reverse_hidden = self.reverse_cov(torch.flip(speed,dims=[2]))
        # hidden2 = self.cov2(speed)

        # reverse_hidden = torch.flip(reverse_hidden, dims=[2])
        # reverse_hidden = torch.multiply(reverse_hidden,torch.sigmoid(reverse_hidden))
        # hidden = torch.multiply(hidden,torch.sigmoid(hidden))

        # hidden_speed =hidden +reverse_hidden + hidden2
        # hidden_speed = hidden_speed.reshape(-1, self.site_num, self.emb_size, self.input_len)
        # hidden_speed = hidden_speed.permute(0,3,1,2)

        return hidden_speed