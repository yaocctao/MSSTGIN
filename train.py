import json
import os
import pandas as pd
from data.load_data import TrafficDataset
from torch.utils.data import DataLoader,Subset
from torch.optim import Adam
import torch
from models.MSSTGIN import MSSTGIN
from tqdm import tqdm
from models.utils import metric
import numpy as np
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def mae_loss(predicted, observed,null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(observed)
    else:
        mask = (observed!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(predicted-observed)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def evaluate(testloader,model,input_length,bn_decay,test=False):
    model.eval()
    model.is_training = False
    pres_list = []
    labels_list = []

    with torch.no_grad():
        for X, DoW, D, H, M, L, XAll in testloader:
            X, DoW, D, H, M, XAll, L= X.to(device), DoW.to(device), D.to(device), H.to(device), M.to(device), XAll.to(device), L.to(device)
            labels = L[:,:,input_length:]
            predicted = model(X, DoW, D, H, M, XAll, bn_decay)

            pres_list.append(predicted.detach().cpu().numpy())
            labels_list.append(labels.detach().cpu().numpy())

    labels_list = np.concatenate(labels_list, axis=0)
    pres_list = np.concatenate(pres_list, axis=0)

    mae, rmse, mape = metric(pres_list, labels_list)
    print('average:         %.3f\t\t%.3f\t\t%.3f%%' %(mae, rmse, mape * 100))
    if test:
        for i in range(input_length):
            mae, rmse, mape = metric(pres_list[:, :, i], labels_list[:, :, i])
            log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            print(log.format(i + 1, mae, rmse, mape))

    return mae,rmse,mape

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def weights_init(model):  #初始化权重
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            m.bias.data.zero_()

        elif isinstance(m, nn.Conv1d):
            m.weight.data.normal_(0.0, 0.02)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight,1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

def train(conf):
    data = pd.read_csv(conf["file_train_s"])

    dataset = TrafficDataset(data,conf["batch_size"],sites = conf["site_num"], P=conf["input_length"], Q=conf["output_length"])
    mean = dataset.mean
    std = dataset.std
    # train_size = int(0.8 * len(dataset))
    #0.7
    test_size = 832
    train_size = 1696

    train_indices = range(0, 1696)
    val_indices = range(1723,1723+320)
    test_indices = range(2045, 2045+640)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    traindataloader = DataLoader(train_dataset, batch_size=conf["batch_size"], shuffle=True, pin_memory=True, num_workers=4)
    print("traindataloader length:",len(traindataloader))
    valdataloader = DataLoader(val_dataset, batch_size=conf["batch_size"], shuffle=True, pin_memory=True, num_workers=4)
    testdataloader = DataLoader(test_dataset, batch_size=conf["batch_size"], shuffle=True, pin_memory=True, num_workers=4)
    
    # num_train = 23967
    global_step = 0
    learning_rate = conf["learning_rate"]
    input_length = conf["input_length"]
    epoch_num = conf["epoch"]
    weight_decay = conf["weight_decay"]
    
    model = MSSTGIN(conf, mean, std)
    print(model)
    # weights_init(model)
    model.to(device)
    model.train()
    optimizer = Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay)

    min_lr = 1e-5
    decay_rate = conf["decay_rate"]
    decay_epoch = conf["decay_epoch"]
    num_train = train_size
    batch_size = conf["batch_size"]

    def exponential_decay(initial_learning_rate, global_step, decay_steps, decay_rate, staircase=True):
        if staircase:
            learning_rate = initial_learning_rate * (decay_rate ** (global_step // decay_steps))
        else:
            learning_rate = initial_learning_rate * (decay_rate ** (global_step / decay_steps))
        
        return learning_rate
    bn_momentum = exponential_decay(0.5, global_step, decay_epoch * num_train // batch_size, 0.5, staircase=True)
    bn_decay = min(0.99,1-bn_momentum)
    

    if not config["is_training"]:
        model.load_state_dict(torch.load("weights/STGIN_sections/epoch_22.pth"))
        evaluate(testdataloader,model,input_length,bn_decay)
        return
    #float 最大值
    max_mae = 1e10
    #判断save_model文件路径是否存在
    if not os.path.exists(conf["save_path"]):
        os.makedirs(conf["save_path"])
    if not os.path.exists(f'{conf["save_path"]}/result.json'):
        with open(f'{conf["save_path"]}/result.json', 'w') as result_file:
            result = {}
            json.dump(result, result_file)  
    result = {}
    
    for epoch_index in tqdm(range(epoch_num)):
        #计算每epoch平均loss
        loss_sum = 0
        for X, DoW, D, H, M, L, XAll in traindataloader:
            X, DoW, D, H, M, XAll, L= X.to(device), DoW.to(device), D.to(device), H.to(device), M.to(device), XAll.to(device), L.to(device)
            global_step += 1
            labels = L[:,:,input_length:]
            predicted = model(X, DoW, D, H, M, XAll, bn_decay)

            loss = mae_loss(predicted, labels)
            optimizer.zero_grad()
            loss.backward()
            loss_sum += loss.item()
            optimizer.step()
            # scheduler.step()
            bn_momentum = exponential_decay(0.5, global_step, decay_epoch * num_train // batch_size, 0.5, staircase=True)
            bn_decay = min(0.99,1-bn_momentum)

            writer.add_scalar("loss", loss.item(), global_step)
            # print(scheduler.get_last_lr()[0])
            # print(optimizer.state_dict()['param_groups'][0]['lr'])
            # for name, parms in model.named_parameters():
                # writer.add_histogram(name, parms.data.flatten(), global_step)
                # writer.add_histogram(name+"grad", parms.grad.data.flatten(), global_step)
    
        mae,rmse,mape = evaluate(valdataloader,model,input_length,bn_decay)
        loss_sum /= len(traindataloader)
        result[f"epoch_{epoch_index}"] = {"mae":float(mae),"rmse":float(rmse),"mape":float(mape), "loss":float(loss_sum)}
        print("loss:",str(float(loss_sum)))
        #保存模型训练结果到json中
        with open(f'{conf["save_path"]}/result.json', "w") as result_file:
            json.dump(result, result_file)

        if max_mae > mae:
            print("in the %dth epoch, the validate average loss value is : %.3f" % (epoch_index + 1, mae))
            max_mae = mae
            mae,rmse,mape = evaluate(testdataloader,model,input_length,bn_decay,test=True)
            print("in the %dth epoch, the test mae,rmse,mape value is : %.3f,%.3f,%.3f%%" % (epoch_index + 1, mae,rmse,mape * 100))
            torch.save(model.state_dict(), f'{conf["save_path"]}/epoch_{epoch_index}.pth')
            print("save model")
        model.train()
    writer.close()


if __name__ == '__main__':
    #nohup python -u train.py > train_v3_add_time_feature_emb_size32_decay_epoch10_weight_decay0001_alter_gcn_STblockwithoutxallandLSTMandBridgeandLR_addTCmoudleAfterEncoderoutNotFusionEncoderout_IAFF.log 2>&1 &
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    
    with open('config/config.json', 'r') as config_file:
        config = json.load(config_file)
        train_path = config["train_config_path"]
        with open(train_path, 'r') as train_config_file:
            train_config = json.load(train_config_file)
            config.update(train_config)
    logs_path = config["tensorboard_logs_path"]
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    writer = SummaryWriter(logs_path)
    setup_seed(22)

    train(config) 