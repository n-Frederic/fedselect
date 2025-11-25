#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from typing import Dict, Any, List

from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc, precision_recall_curve
from sklearn.metrics import roc_auc_score, roc_curve
"""
    这个文件暂时没有用途
"""

def test_model(net_g, datatest, args):
    rt = {}  # 返回结果
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)

    # 使用 args.device 来统一管理设备
    y_labels = torch.LongTensor([]).to(args.device)
    y_preds = torch.LongTensor([]).to(args.device)
    y_preds_values = torch.FloatTensor([]).to(args.device)  # 概率值是浮点数

    with torch.no_grad():  # 使用 with torch.no_grad() 替代 .data
        for idx, (data, target) in enumerate(data_loader):
            data, target = data.to(args.device), target.to(args.device)

            # 假设 net_g 返回 (特征, log_probs)，如果只返回 log_probs，请用 log_probs = net_g(data)
            log_probs = net_g(data)

            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()

            y_pred = log_probs.argmax(1)  # 直接使用 argmax 获取预测类别

            y_preds_value = F.softmax(log_probs, dim=1)[:, 1]

            # 现在所有张量都在同一个设备上，拼接不会再报错
            y_preds = torch.cat((y_preds, y_pred), 0)
            y_preds_values = torch.cat((y_preds_values, y_preds_value), 0)
            y_labels = torch.cat((y_labels, target), 0)  # 直接拼接 target 即可

            correct += y_pred.eq(target).long().sum().item()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    # if args.verbose:
    #     print('\ntest_model: Testing Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
    #         test_loss, correct, len(data_loader.dataset), accuracy))

    y_labels_np = y_labels.cpu().numpy()
    y_preds_np = y_preds.cpu().numpy()
    y_preds_values_np = y_preds_values.cpu().numpy()

    ma_f1 = f1_score(y_labels_np, y_preds_np, average='macro')
    mi_f1 = f1_score(y_labels_np, y_preds_np, average='micro')
    f1 = f1_score(y_labels_np, y_preds_np, average='binary', pos_label=1)
    cm = confusion_matrix(y_labels_np, y_preds_np, labels=None, sample_weight=None)

    # 确保正样本存在，否则 roc_auc_score 会报错
    if len(set(y_labels_np)) > 1:
        auc_score = roc_auc_score(y_labels_np, y_preds_values_np)
        fpr, tpr, thresholds = roc_curve(y_labels_np, y_preds_values_np, pos_label=1)
        p, r, thrh = precision_recall_curve(y_labels_np, y_preds_values_np, pos_label=1)
        rt['auprc'] = auc(r, p)
        rt['ks'] = max(tpr - fpr)
    else:  # 如果只有一个类别，则无法计算AUC等指标
        auc_score = 0.5
        fpr, tpr, p, r = [0], [0], [0], [0]
        rt['auprc'] = 0.0
        rt['ks'] = 0.0

    accuracy1 = accuracy_score(y_labels_np, y_preds_np)
    precision = precision_score(y_labels_np, y_preds_np, average='binary', pos_label=1, zero_division=0)
    recall = recall_score(y_labels_np, y_preds_np, average='binary', pos_label=1, zero_division=0)
    recall_ma = recall_score(y_labels_np, y_preds_np, average='macro', zero_division=0)

    # 计算检测出的欺诈金额 (假设 datatest.data 存在且为 DataFrame)
    index1 = np.where(y_labels_np == 1)[0]
    index1_pred = np.where((y_labels_np == 1) & (y_preds_np == 1))[0]

    # 处理 Subset 对象
    if isinstance(datatest, Subset):
        # 如果是 Subset，需要通过 indices 映射到原始数据集
        original_indices = [datatest.indices[i] for i in index1]
        original_indices_pred = [datatest.indices[i] for i in index1_pred]
        amount1 = datatest.dataset.data.iloc[original_indices].Amount.sum()
        amount1_pred = datatest.dataset.data.iloc[original_indices_pred].Amount.sum()
    else:
        # 普通数据集
        amount1 = datatest.data.iloc[index1].Amount.sum()
        amount1_pred = datatest.data.iloc[index1_pred].Amount.sum()

    rt['loss'] = test_loss
    rt['accuracy'] = accuracy
    rt['accuracy1'] = accuracy1
    rt['precision'] = precision
    rt['recall'] = recall
    rt['recall_ma'] = recall_ma
    rt['tnr'] = cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0.0
    rt['fpr'] = cm[0][1] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0.0
    rt['gmean'] = (recall * rt['tnr']) ** 0.5
    rt['ma_f1'] = ma_f1
    rt['mi_f1'] = mi_f1
    rt['f1'] = f1
    rt['cm'] = cm
    rt['cm00'] = cm[0][0]
    rt['cm01'] = cm[0][1]
    rt['cm10'] = cm[1][0]
    rt['cm11'] = cm[1][1]
    rt['auc'] = auc_score
    rt['roc'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
    rt['prc'] = {'precision': p, 'recall': r, 'thresholds': thrh}
    rt['amount1'] = amount1
    rt['amount1_pred'] = amount1_pred
    print('test_model: Testing cm: {}\n'.format(rt['cm']))

    return rt


def evaluate_selected_clients(
        model: nn.Module,
        dataset_test: Any,
        dict_users_test: Dict[int, Any],
        selected_clients: List[int],
        args: Any,
) -> Dict[str, Any]:
    """
    评估选定的客户端，并打印每个客户端的详细指标和平均值。
    
    参数:
        model: 全局模型
        dataset_test: 测试数据集
        dict_users_test: 客户端测试数据索引字典
        selected_clients: 选定的客户端ID列表
        args: 参数配置
    
    返回:
        包含所有客户端指标的字典
    """
    print("\n" + "="*60)
    print(f"Selected Clients Evaluation ({len(selected_clients)} clients)")
    print("="*60)
    
    client_metrics = {
        'accuracies': [],
        'recalls': [],
        'f1s': [],
        'aucs': [],
        'precisions': []
    }
    
    # 创建索引映射
    index_map = {v: i for i, v in enumerate(dataset_test.data.index)}
    
    for idx in selected_clients:
        # 创建客户端测试子集
        client_indices = [index_map[i] for i in dict_users_test[idx]]
        client_test_subset = Subset(dataset_test, client_indices)
        
        # 评估客户端
        result = test_model(model, client_test_subset, args)
        
        # 收集指标
        client_metrics['accuracies'].append(result['accuracy1'])
        client_metrics['recalls'].append(result['recall'])
        client_metrics['f1s'].append(result['f1'])
        client_metrics['aucs'].append(result['auc'])
        client_metrics['precisions'].append(result['precision'])
        
        # 打印客户端结果
        cm = result['cm']
        print(f"Client_{idx} test results -> cm: [[{cm[0][0]:4d}, {cm[0][1]:4d}], [{cm[1][0]:4d}, {cm[1][1]:4d}]]")
        print(f"  └─> Accuracy: {result['accuracy1']:.4f}, "
              f"Recall: {result['recall']:.4f}, "
              f"F1: {result['f1']:.4f}, "
              f"AUC: {result['auc']:.4f}")
    
    # 计算并打印平均值
    print(f"\nAverage Metrics:")
    print(f"  Accuracy:  {np.mean(client_metrics['accuracies']):.4f}")
    print(f"  Precision: {np.mean(client_metrics['precisions']):.4f}")
    print(f"  Recall:    {np.mean(client_metrics['recalls']):.4f}")
    print(f"  F1:        {np.mean(client_metrics['f1s']):.4f}")
    print(f"  AUC:       {np.mean(client_metrics['aucs']):.4f}")
    print("="*60 + "\n")
    
    return client_metrics