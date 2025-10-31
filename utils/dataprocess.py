import pandas as pd
import numpy as np
from torch.utils.data.dataset import Dataset
import torch
# from PIL import Image
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE


class DatasetFromCSV(Dataset):
    def __init__(self, csv_path, train=True):
        self.data_all = pd.read_csv(csv_path)
        order = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16','V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Class']
        self.data_all = self.data_all[order]
        y = self.data_all['Class']
        self.data_train, self.data_test = train_test_split(self.data_all, test_size=0.2, random_state=888,stratify=y)
        # 统计测试数据集中的欺诈金额和欺诈样本数量
        amount1 = self.data_test[self.data_test['Class']==1].Amount.sum()
        count1 = self.data_test[self.data_test['Class'] == 1].Amount.count()
        print("划分数据集,条件：train=",train)
        print("划分后，测试集中的欺诈样本数量为：",count1)
        print("划分后，测试集中的欺诈样本金额为：", amount1)
        if train:
            self.data = self.data_train
        else:
            self.data = self.data_test
        self.labels = np.asarray(self.data.iloc[:, -1])
        self.targets = torch.tensor(self.labels)

    def __getitem__(self, index):
        single_image_label = self.labels[index]
        single_data = torch.tensor(np.asarray(self.data.iloc[index][0:29]), dtype=torch.float32)
        return (single_data, single_image_label)

    def __len__(self):
        return len(self.data.index)

class DatasetBalance(Dataset):
    def __init__(self, data_frame):
        self.data = data_frame
        # 对数据集进行上采样平衡处理
        y = self.data['Class']
        X = self.data.drop(columns='Class')
        X_resampled, y_resampled = SMOTE(random_state=888).fit_resample(X, y)
        # X_resampled, y_resampled = SMOTE().fit_resample(X, y)
        self.data = pd.concat([X_resampled, y_resampled], axis=1)
        self.labels = np.asarray(self.data.iloc[:, -1])

    def __getitem__(self, index):
        single_image_label = self.labels[index]
        single_data = torch.tensor(np.asarray(self.data.iloc[index][0:29]), dtype=torch.float32)
        return (single_data, single_image_label)

    def __len__(self):
        return len(self.data.index)

class DatasetFromDataframe(Dataset):
    def __init__(self, data_frame):
        self.data = data_frame
        self.labels = np.asarray(self.data.iloc[:, -1])

    def __getitem__(self, index):
        single_image_label = self.labels[index]
        single_data = torch.tensor(np.asarray(self.data.iloc[index][0:29]), dtype=torch.float32)
        return (single_data, single_image_label)

    def __len__(self):
        return len(self.data.index)