import pandas as pd
import os
import numpy as np


def save_predications(observed,predictions,model_name,save_path):
    #保存numpy array数据
    # observed:真实值
    # predictions:预测值
    # model_name:模型名称
    if not os.path.exists(save_path) :
        os.makedirs(save_path)
    save_path = os.path.join(save_path,model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path , "observed.npy"), observed)
    print("observed: ",observed.shape)
    np.save(os.path.join(save_path , "predictions.npy"), predictions)
    print("predictions: ",predictions.shape)
    print("save predictions successfully")

def save_metrics(maes,rmses,mapes,model_name,save_path):
    #保存metrics数据
    # metrics:metrics数据
    # model_name:模型名称
    if not os.path.exists(save_path) :
        os.makedirs(save_path)
    save_path = os.path.join(save_path,model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    metrics = pd.DataFrame(columns=["mae","rmse","mape"])
    metrics["mae"] = maes
    metrics["rmse"] = rmses
    metrics["mape"] = mapes
    metrics.to_csv(os.path.join(save_path , "metrics.csv"),index=False)
    print("save metrics successfully")

def read_predications(model_name,save_path):
    #读取numpy array数据
    # model_name:模型名称
    # save_path:保存路径
    save_path = os.path.join(save_path,model_name)
    observed = np.load(os.path.join(save_path , "observed.npy"),allow_pickle=True)
    predictions = np.load(os.path.join(save_path , "predictions.npy"),allow_pickle=True)
    return observed,predictions

if __name__ == "__main__":
    observed,predictions = read_predications("FI-RNNs","predictions")
    print(observed.shape)
    print(predictions.shape)