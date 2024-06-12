import torch,os
import torch.nn as nn
import numpy as np
from models.utils import metric
from models.MSSTGIN import MSSTGIN
from torch.utils.data import DataLoader, random_split, Subset
from data.load_data import TrafficDataset
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from save_predictions import save_predications, save_metrics


def draw_Linear_regression(pres_list, labels_list,days_list,hours_list):
    # Randomly sample 100 numbers
    random_numbers = np.random.choice(pres_list.shape[0], size=100)
    pres_list = pd.DataFrame(pres_list[random_numbers, :, :].flatten(), columns=['predict'])
    labels_list = pd.DataFrame(labels_list[random_numbers, :, :].flatten(), columns=['observed'])
    dataframe = pd.concat([labels_list,pres_list],axis=1)
    dataframe = dataframe[dataframe['observed'] != 0]
    sns.set_theme(style="darkgrid")

    sns.jointplot(x="observed", y="predict", data=dataframe, marginal_ticks=True, height=7)
    plt.xlim(0, 150)
    # g.set_axis_labels("", "error")
    plt.savefig('pre_test_jointplot.png')
    plt.show()
    
    plt.clf()
    plt.figure(figsize=(14, 6))
    # palette = sns.xkcd_palette(["dusty purple", "windows blue"])
    palette = sns.xkcd_palette(["windows blue","orange"])
    # sns.lineplot(x="observed", y="predict", hue='index', palette=palette, data=dataframe.T.reset_index())
    # sns.lineplot(palette=palette, data=dataframe.loc[len(dataframe)-1200:len(dataframe), ['observed','predict']],dashes=False)
    sns.lineplot(palette=palette, data=dataframe,dashes=False)
    plt.savefig('pre_test_linear_regression.png')
    # dataframe = dataframe.loc[len(dataframe)-1200:len(dataframe), ['observed','predict']]
    # dataframe.to_csv("./test.csv",index=False)

def get_dataframe(pres_list, labels_list,days_list,hours_list):
    dataframe = pd.DataFrame(columns=['day','hour','label','predict'])
    #取pres_list中每行的平均值替换每行原有的值
    pres_list = np.mean(pres_list,axis=2)
    labels_list = np.mean(labels_list,axis=2)
    error = pd.DataFrame(np.abs(pres_list - labels_list))
    error = error.rename(columns=lambda x: f"dragon_{x}").stack().reset_index().drop(["level_0"],axis=1).rename(columns={"level_1":"dragon",0:"error"})    

    days_list = pd.DataFrame(days_list[:,:,0].flatten(),columns=["day"])
    hours_list = pd.DataFrame(hours_list[:,:,0].flatten(),columns=['hour'])

    dataframe = pd.concat([days_list,hours_list,error],axis=1)

    sns.set_theme(style="dark")

    # Plot each year's time series in its own facet
    g = sns.relplot(
        data=dataframe,
        x="dragon", y="error", col="day", hue="hour",
        kind="line", palette="crest", linewidth=4, zorder=5,
        col_wrap=3, height=2, aspect=1.5,
    )
    # for day, ax in g.axes_dict.items():

    #     # Add the title as an annotation within the plot
    #     ax.text(.8, .85, day, transform=ax.transAxes, fontweight="bold")

    #     # Plot every year's time series in the background
    #     sns.lineplot(
    #         data=dataframe, x="dragon", y="error", units=["day","hour"],
    #         estimator=None, color=".7", linewidth=1, ax=ax,
    #     )

    # # Reduce the frequency of the x axis ticks
    # ax.set_xticks(ax.get_xticks()[::2])

    # Tweak the supporting aspects of the plot
    g.set_titles("")
    g.set_axis_labels("", "error")
    g.tight_layout()
    plt.show()

    return dataframe

def evaluate(testloader,model,input_length,conf):
    model.eval()
    pres_list = []
    labels_list = []
    hours_list = []
    days_list = []
    
    with torch.no_grad():
        for X, DoW, D, H, M, L, XAll in testloader:
            X, DoW, D, H, M, XAll, L= X.to(device), DoW.to(device), D.to(device), \
            H.to(device), M.to(device), XAll.to(device), L.to(device)
            labels = L[:,:,input_length:]
            predicted = model(X, DoW, D, H, M, XAll,0.5)

            hours_list.append(H[:,input_length:,:].permute(0,2,1).detach().cpu().numpy())
            days_list.append(D[:,input_length:,:].permute(0,2,1).detach().cpu().numpy())     
            pres_list.append(predicted.detach().cpu().numpy())
            labels_list.append(labels.detach().cpu().numpy())

    days_list = np.concatenate(days_list, axis=0)
    hours_list = np.concatenate(hours_list, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)
    pres_list = np.concatenate(pres_list, axis=0)

    length = 1
    maes = []
    rmses = []
    mapes = []
    for i in range(0,12):
        mae,rmse,mape = metric(pres_list[:,:,i], labels_list[:,:,i])
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, mae, rmse, mape))
    
    #取出预测值和真实值大于120速度的值
    _pres_list = pres_list.flatten()
    _labels_list = labels_list.flatten()
    pres_list_larger_120 = _pres_list[_labels_list>120]
    labels_list_larger_120 = _labels_list[_labels_list>120]
    #取出预测值和真实值低于60速度的值
    pres_list_lower_60 = _pres_list[_labels_list<60]
    labels_list_lower_60 = _labels_list[_labels_list<60]   
    
    mae,rmse,mape = metric(pres_list_larger_120, labels_list_larger_120)
    print('average:         %.3f\t\t%.3f\t\t%.3f%%' %(mae, rmse, mape * 100))
    mae,rmse,mape = metric(pres_list_lower_60, labels_list_lower_60)
    print('average:         %.3f\t\t%.3f\t\t%.3f%%' %(mae, rmse, mape * 100))
    
    mae, rmse, mape = metric(pres_list, labels_list)
    print('average:         %.3f\t\t%.3f\t\t%.3f%%' %(mae, rmse, mape * 100))
    maes.append(mae)
    rmses.append(rmse)
    mapes.append(mape)
    for i in range(0,hours_list.shape[0]-length+1,length):
        mae,rmse,mape = metric(pres_list[i:i+length], labels_list[i:i+length])
    draw_Linear_regression(pres_list, labels_list,days_list,hours_list)
    save_metrics(maes,rmses,mapes, conf["model_name"],"./predictions")
    save_predications(labels_list,pres_list,conf["model_name"],"./predictions")
    # res = get_dataframe(pres_list,labels_list,days_list,hours_list)

    return mae


if __name__ == '__main__':
    model_weight = 'epoch_39.pth'
    with open('config/evaluate.json', 'r') as config_file:
        conf = json.load(config_file)
    data = pd.read_csv(conf["file_train_s"])
    dataset = TrafficDataset(data,conf["batch_size"],sites=conf["site_num"])
    # data = np.load('data/G15_numpy/test.npz')
    model_path = os.path.join(conf['save_path'],model_weight)
    mean = dataset.mean
    std = dataset.std
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))

    model = MSSTGIN(conf, mean, std)
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    test_size = 832
    train_size = len(dataset) - test_size
    test_indices = range(train_size, len(dataset))
    # test_indices = range(2045, 2045+640)
    test_dataset = Subset(dataset, test_indices)
    testdataloader = DataLoader(test_dataset, batch_size=conf["batch_size"], shuffle=False)
    evaluate(testdataloader,model,conf["input_length"],conf)