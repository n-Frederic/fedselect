#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
联邦聚合算法模块
包含三种聚合策略：
1. FedAVG - 基于样本数量的加权平均
2. FedMEAN - 简单平均
3. FedRWA - 引入风险权值的联邦聚合（考虑欺诈数据比例和金额）
"""

import copy
import torch
from typing import Dict, List, OrderedDict


def FedAVG(
    w: OrderedDict,
    dw: List[OrderedDict],
    sample_nums: List[int]
) -> OrderedDict:
    """基于样本数量的加权平均聚合算法
    
    Args:
        w: 全局模型的当前参数
        dw: 各客户端的参数更新列表
        sample_nums: 各客户端的样本数量列表
    
    Returns:
        更新后的全局模型参数
    """
    # 计算所有客户端样本数量比例
    num_ratio = [sample_nums[i] / sum(sample_nums) for i in range(len(sample_nums))]
    
    # 计算第0个客户端待聚合的参数更新（加权）
    dw_avg = copy.deepcopy(dw[0])
    for k in dw_avg.keys():
        dw_avg[k] = num_ratio[0] * dw_avg[k]
    
    # 聚合所有客户端的参数更新，加上样本数量比例为权值
    for k in dw_avg.keys():
        for i in range(1, len(dw)):
            dw_avg[k] += num_ratio[i] * dw[i][k]
    
    # 更新全局模型的参数
    w_updated = copy.deepcopy(w)
    for k in w_updated.keys():
        if k in dw_avg:
            w_updated[k] = w_updated[k] + dw_avg[k]
    
    return w_updated


def FedMEAN(
    w: OrderedDict,
    dw: List[OrderedDict]
) -> OrderedDict:
    """简单平均聚合算法
    
    Args:
        w: 全局模型的当前参数
        dw: 各客户端的参数更新列表
    
    Returns:
        更新后的全局模型参数
    """
    dw_mean = copy.deepcopy(dw[0])
    
    # 对所有客户端的参数更新求和并取平均
    for k in dw_mean.keys():
        for i in range(1, len(dw)):
            dw_mean[k] += dw[i][k]
        dw_mean[k] = torch.div(dw_mean[k], len(dw))
    
    # 更新全局模型的参数
    w_updated = copy.deepcopy(w)
    for k in w_updated.keys():
        if k in dw_mean:
            w_updated[k] = w_updated[k] + dw_mean[k]
    
    return w_updated


def FedRWA(
    w: OrderedDict,
    dw: List[OrderedDict],
    a: List[float],
    s: List[float]
) -> OrderedDict:
    """引入风险权值的联邦聚合算法
    
    该算法根据客户端数据集的风险特征进行加权聚合，风险数值与数据集大小、
    欺诈样本比例、欺诈金额大小正相关。
    
    Args:
        w: 全局模型的当前参数
        dw: 各客户端的参数更新列表
        a: 各客户端的本地测试准确率列表
        s: 各客户端的数据集风险数值列表
           风险数值的计算方式：
           - risk_type=1: 数据集大小
           - risk_type=2: 欺诈样本数 + 正常样本数/M (M为平衡因子)
           - risk_type=3: 欺诈金额总和
    
    Returns:
        更新后的全局模型参数
    """
    # 计算第0个客户端融合风险权值的参数更新
    # k代表模型中的每种参数，例如 'conv1.weight', 'conv1.bias' 等
    dw_avg = copy.deepcopy(dw[0])
    
    # 使用风险权值进行加权
    for k in dw_avg.keys():
        dw_avg[k] = (s[0] / sum(s)) * dw_avg[k]
        # 可选：同时考虑准确率
        # dw_avg[k] = a[0] * (s[0] / sum(s)) * dw_avg[k]
    
    # 累积所有客户端融合风险权值的参数更新
    for k in dw_avg.keys():
        for i in range(1, len(dw)):
            dw_avg[k] += (s[i] / sum(s)) * dw[i][k]
            # 可选：同时考虑准确率
            # dw_avg[k] += a[i] * (s[i] / sum(s)) * dw[i][k]
    
    # 更新全局模型的参数
    w_updated = copy.deepcopy(w)
    for k in w_updated.keys():
        if k in dw_avg:
            w_updated[k] = w_updated[k] + dw_avg[k]
    
    return w_updated


def compute_risk_score(
    dataset_size: int,
    fraud_count: int,
    normal_count: int,
    fraud_amount: float,
    risk_type: int = 1,
    balance_factor: float = 576.0
) -> float:
    """计算客户端数据集的风险数值
    
    Args:
        dataset_size: 数据集总大小
        fraud_count: 欺诈样本数量
        normal_count: 正常样本数量
        fraud_amount: 欺诈样本金额总和
        risk_type: 风险计算类型
            1 - 仅考虑数据集大小
            2 - 欺诈样本数 + 正常样本数/平衡因子
            3 - 欺诈金额总和
        balance_factor: 平衡因子（默认576，用于调整正常样本的权重）
    
    Returns:
        风险数值
    """
    if risk_type == 1:
        return float(dataset_size)
    elif risk_type == 2:
        # P + N/M (P=欺诈样本数, N=正常样本数, M=平衡因子)
        return float(fraud_count + (normal_count / balance_factor))
    elif risk_type == 3:
        # 欺诈金额总和
        # 或可选：(P + N/M) * 平均欺诈金额
        return float(fraud_amount)
    else:
        return 1.0
