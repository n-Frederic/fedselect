#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils.sampling import creditcard_iid, creditcard_noniid
from utils.options import lth_args_parser
from utils.dataprocess import DatasetFromCSV,DatasetBalance
from models.nets import MLP
from models.test import test_model

"""
    这个文件暂时没有用途，只是用来做本地训练使用的
"""
if __name__ == '__main__':
    # parse args
    # args = args_parser() #使用命令行参数方式
    args = lth_args_parser() #使用配置方式
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load dataset and split users
    dataset_train = DatasetFromCSV('./data/creditcard/creditcard.csv', train=True)
    dataset_test = DatasetFromCSV('./data/creditcard/creditcard.csv', train=False)
    # sample users
    if args.iid:
        dict_users = creditcard_iid(dataset_train, args.num_users)
    else:
        dict_users = creditcard_noniid(dataset_train, args.num_users, type=args.split_dataset_type, list_ratio=args.split_dataset_ratio)
    for ui in range(len(dict_users)):
        user_data = dataset_train.data.loc[list(dict_users[ui])]
        print("第{}个客户端: 总样本数量为{}，欺诈样本数量为{}，平均欺诈样本金额为{}".format(ui,len(user_data),len(user_data[user_data.Class==1]),sum(user_data[user_data.Class==1].Amount)/len(user_data[user_data.Class==1])))
    img_size = dataset_train[0][0].shape

    # build model
    len_in = 1
    for x in img_size:
        len_in *= x
    len_in = 29
    net_local = MLP(input_dim=len_in,hidden=120, num_classes=args.num_classes).to(args.device)

    print(net_local)
    for ui in range(len(dict_users)):
        #拆分出本地数据，并进行平衡处理
        train_loader = DataLoader(DatasetBalance(dataset_train.data.loc[list(dict_users[ui])]), batch_size=128, shuffle=True)
        # 加载一个固定的初始模型
        net_local.load_state_dict(torch.load('./fed_model-mlp120.pt'))
        net_local.train()
        optimizer = torch.optim.SGD(net_local.parameters(), lr=args.lr, momentum=args.momentum)

        testset_val_result = pd.DataFrame(
            columns=['loss', 'accuracy', 'accuracy1', 'precision', 'recall', 'recall_ma', 'tnr', 'fpr', 'gmean',
                     'ma_f1', 'mi_f1', 'f1', 'auc', 'ks', 'auprc', 'cm00', 'cm01', 'cm10', 'cm11', 'amount1',
                     'amount1_pred'])
        print('local_dataset_train: client{:3d} start train:'.format(ui))
        for epoch in range(args.com_rounds):
            batch_loss = []
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(args.device), target.to(args.device)
                optimizer.zero_grad()
                output = net_local(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % 50 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss.item()))
                batch_loss.append(loss.item())
            loss_avg = sum(batch_loss) / len(batch_loss)
            print('\nepoch{} Train loss:{}'.format(epoch, loss_avg))

            # testing  在每个迭代轮次评估模型效果
            print("epoch{}训练完成，开始测试模型效果：".format(epoch))
            net_local.eval()
            # 在测试集上测试一下模型效果
            result = test_model(net_local, dataset_test, args)
            testset_val_result.loc[epoch] = [
                result['loss'],
                result['accuracy'],
                result['accuracy1'],
                result['precision'],
                result['recall'],
                result['recall_ma'],
                result['tnr'],
                result['fpr'],
                result['gmean'],
                result['ma_f1'],
                result['mi_f1'],
                result['f1'],
                result['auc'],
                result['ks'],
                result['auprc'],
                result['cm00'],
                result['cm01'],
                result['cm10'],
                result['cm11'],
                result['amount1'],
                result['amount1_pred']]
            # print("本次在测试集上的验证结果：", result)
            print("Test-dataset test cm: {}".format(result['cm']))
            # 持久化模型
            torch.save(net_local.state_dict(), f'./local_models/local-model-mlp120-{args.num_users}-{ui}.pt')
