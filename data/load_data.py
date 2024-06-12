#load datasets for pytorch
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader,Subset
import numpy as np
import datetime
from torch.utils.data.dataset import random_split


class TrafficDataset(Dataset):
    def __init__(self, data, batch_size = 32, P=12, Q=12, low_index=672, high_index=100, granularity=15, sites = 145, type='train',mean=66.9478,std=47.0847,features = 2):
        self.data = data.values
        #将多个时间维度拼接在一起
        self.datetime = pd.to_datetime(data['date'].astype(str) + ' ' + data['hour'].astype(str) + ":" + data['minute'].astype(str), format='%Y-%m-%d %H:%M')
        self.datetime = self.datetime.to_numpy()
        self.batch_size = batch_size
        self.length = len(data)
        self.P = P
        self.Q = Q
        
        self.granularity = granularity
        self.sites = sites
        self.type = type
        self.mean = mean
        self.std = std
        self.total_week_len = 60 // granularity * 24 * 7
        self.low_index = self.total_week_len
        # self.low_index = 0
        self.features = features
        self.traffic_flow_mean = 63.0928
        self.traffic_flow_std = 83.5519


    def __len__(self):
        return (self.length // self.sites - self.P - self.Q - self.total_week_len + 1) // self.batch_size * self.batch_size
        # return (self.length // self.sites - self.P - self.Q + 1) // self.batch_size * self.batch_size

    def __getitem__(self, idx):       
        label = self.data[(self.low_index + idx) * self.sites: (self.low_index + idx + self.P + self.Q) * self.sites, 5:6]
        label = np.concatenate([label[i * self.sites: (i + 1) * self.sites] for i in range(self.Q+self.P)], axis=1)
        date = self.data[(self.low_index + idx) * self.sites: (self.low_index + idx + self.P + self.Q) * self.sites, 1]
        X = torch.tensor(np.reshape(self.data[(self.low_index + idx) * self.sites: (self.low_index + idx + self.P) * self.sites, 5],[self.P, self.sites,1]).astype(np.float32))
        # counts = torch.tensor(np.reshape(self.data[(self.low_index + idx) * self.sites: (self.low_index + idx + self.P) * self.sites, 6],[self.P, self.sites,1]).astype(np.float32))
        # truck_num = torch.tensor(np.reshape(self.data[(self.low_index + idx) * self.sites: (self.low_index + idx + self.P) * self.sites, 7],[self.P, self.sites,1]).astype(np.float32))
        # counts = (counts - self.traffic_flow_mean) / self.traffic_flow_std
        daytime = self.datetime[(self.low_index + idx) * self.sites: (self.low_index + idx + self.P) * self.sites]
        time_in_day = torch.tensor(np.reshape((daytime-daytime.astype("datetime64[D]"))/ np.timedelta64(1, "D"), [self.P, self.sites, 1]).astype(np.float32))
        X = (X - self.mean) / self.std  # Normalize X
        # X = np.concatenate((X,counts),axis=-1)
        X = np.concatenate((X,time_in_day),axis=-1)
        DoW = torch.tensor(np.reshape([datetime.date(int(char.replace('/', '-').split('-')[0]), int(char.replace('/', '-').split('-')[1]),
                              int(char.replace('/', '-').split('-')[2])).weekday() for char in date],[self.P+self.Q, self.sites]).astype(np.int32))
        D = torch.tensor(np.reshape(self.data[(self.low_index + idx) * self.sites: (self.low_index + idx + self.P + self.Q) * self.sites, 2],[self.P+self.Q, self.sites]).astype(np.int32))
        H = torch.tensor(np.reshape(self.data[(self.low_index + idx) * self.sites: (self.low_index + idx + self.P + self.Q) * self.sites, 3],[self.P+self.Q, self.sites]).astype(np.float32))

        hours_to_minutes = self.data[(self.low_index + idx) * self.sites: (self.low_index + idx + self.P + self.Q) * self.sites, 3] * 60
        minutes_index_of_day = np.add(hours_to_minutes, self.data[(self.low_index + idx) * self.sites: (self.low_index + idx + self.P + self.Q) * self.sites, 4])
        M = torch.tensor(np.reshape(minutes_index_of_day // self.granularity,[self.P+self.Q, self.sites]).astype(np.int32))
        L = torch.tensor(np.reshape(label,[self.sites, self.Q+self.P]).astype(np.float32))
        XAll = torch.tensor(np.reshape(self.data[(self.low_index + idx - self.total_week_len) * self.sites: (self.low_index + idx - self.total_week_len + self.P + self.Q) * self.sites, 5],[self.P+self.Q, self.sites,1]).astype(np.float32))
        Alldaytime = self.datetime[(self.low_index + idx - self.total_week_len) * self.sites: (self.low_index + idx - self.total_week_len + self.P + self.Q) * self.sites]
        Alltime_in_day = torch.tensor(np.reshape((Alldaytime-Alldaytime.astype("datetime64[D]"))/ np.timedelta64(1, "D"), [self.P+self.Q, self.sites,1]).astype(np.float32))
        # All_counts = torch.tensor(np.reshape(self.data[(self.low_index + idx - self.total_week_len) * self.sites: (self.low_index + idx - self.total_week_len + self.P + self.Q) * self.sites, 6],[self.P+self.Q, self.sites,1]).astype(np.float32))
        # All_truck_num = torch.tensor(np.reshape(self.data[(self.low_index + idx - self.total_week_len) * self.sites: (self.low_index + idx - self.total_week_len + self.P + self.Q) * self.sites, 7],[self.P+self.Q, self.sites,1]).astype(np.float32))
        # All_counts = (All_counts - self.traffic_flow_mean) / self.traffic_flow_std
        # XAll = (XAll - self.mean) / self.std
        # XAll = np.concatenate((XAll,All_counts),axis=-1)
        XAll = np.concatenate((XAll,Alltime_in_day),axis=-1)


        return X, DoW, D, H, M, L, XAll

if __name__ == '__main__':
    df = pd.read_csv('data/G15_v3/G15_ETC_sections_speed.csv')
    dataset = TrafficDataset(df,sites=227, P=12, Q=12, granularity=15)
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    train_indices = range(0, 1696)
    val_indices = range(1723,1723+320)
    test_indices = range(2045, 2045+640)
    # train_size = len(dataset) - test_size
    # generator = torch.Generator().manual_seed(42)
    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size],generator)
    train_dataset = Subset(dataset, train_indices)
    traindataloader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=1)
    for X, DoW, D, H, M, L, XAll in traindataloader:
        print(X.shape, DoW.shape, D.shape, H.shape, M.shape, L.shape, XAll.shape)
    def get_mean_std(loader):
        # Var[x] = E[X**2]-E[X]**2
        X_sum,X_squared_sum,num_batches = 0,0,0
        for X, DoW, D, H, M, L, XAll in loader:
            print(X[0,:,0,:])
            X = X[:,:,:,1:2]
            X_sum += torch.mean(X)
            X_squared_sum += torch.mean(X**2)
            num_batches += 1
        mean = X_sum/num_batches
        std = (X_squared_sum/num_batches - mean**2) **0.5
        return mean,std

    mean,std = get_mean_std(traindataloader)
    print(mean,std)
        