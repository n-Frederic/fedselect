import random
import numpy as np
import torch
from typing import Dict, List, Set, Union, Tuple
from torch.utils.data import Dataset


def iid(dataset: Dataset, num_users: int) -> Dict[int, Set[int]]:
    """从数据集中按 I.I.D.（独立同分布）随机均匀划分客户端数据。

    参数:
        dataset: 完整数据集
        num_users: 要划分的客户端数量

    返回:
        一个字典，将每个客户端 ID 映射到分配给该客户端的数据索引集合
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def noniid(
    dataset: Dataset,
    num_users: int,
    shard_per_user: int,
    server_data_ratio: float = 0.0,
    size: Union[int, None] = None,
    rand_set_all: List = [],
) -> Tuple[Dict[Union[int, str], Union[np.ndarray, Set[int]]], np.ndarray]:
    """通过按类别划分数据来生成非 I.I.D. 的客户端数据划分。

    参数:
        dataset: 完整数据集
        num_users: 客户端数量
        shard_per_user: 每个客户端分配多少个类别分片
        server_data_ratio: 预留给服务器的数据比例（默认 0.0）
        size: 可选的限制每个客户端数据量大小
        rand_set_all: 可传入用于划分的随机标签分配数组

    返回:
        一个元组:
            - 客户端 ID 到索引数组的映射字典
            - 本轮划分使用的随机类别分配序列
    """
    dict_users, all_idxs = {i: np.array([], dtype="int64") for i in range(num_users)}, [
        i for i in range(len(dataset))
    ]

    targets = None
    targets = [elem[1] for elem in dataset]

    # 为每个 label 建立一个索引表
    idxs_dict = {}
    for i in range(len(dataset)):
        label = torch.tensor(targets[i]).item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    num_classes = len(np.unique(targets))
    shard_per_class = int(shard_per_user * num_users / num_classes)

    # 将每个类别的数据分成若干 shard
    for label in idxs_dict.keys():
        x = idxs_dict[label]
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x

    # 若没有预定义随机序列，则生成一个
    if len(rand_set_all) == 0:
        rand_set_all = list(range(num_classes)) * shard_per_class
        random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))

    # 分配给各个客户端
    for i in range(num_users):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            idx = np.random.choice(len(idxs_dict[label]), replace=False)
            rand_set.append(idxs_dict[label].pop(idx))
        dict_users[i] = np.concatenate(rand_set)

    # 正确性检查：每个数据索引必须只出现一次
    test = []
    for key, value in dict_users.items():
        x = np.unique(torch.tensor(targets)[value])
        assert (len(x)) <= shard_per_user
        test.append(value)
    test = np.concatenate(test)
    assert len(test) == len(dataset)
    assert len(set(list(test))) == len(dataset)

    # 若有服务器数据比例，额外分配一份给 server
    if server_data_ratio > 0.0:
        dict_users["server"] = set(
            np.random.choice(
                all_idxs, int(len(dataset) * server_data_ratio), replace=False
            )
        )

    # 若 size 不为空，按分片大小切割每个用户的数据
    for i in range(num_users):
        num_elem = len(dict_users[i])
        dict_users[i] = np.concatenate(
            [
                dict_users[i][k : k + size]
                for k in range(0, num_elem, num_elem // shard_per_user + 1)
            ]
        )

    return dict_users, rand_set_all


def creditcard_iid(dataset, num_users):
    """
    以 I.I.D. 方式从信用卡数据集中划分用户数据
    :param dataset:
    :param num_users:
    :return: 用户 ID 到数据索引集合的字典
    """
    np.random.seed(0)
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, np.arange(len(dataset))
    for i in range(num_users-1):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    dict_users[i+1] = set(all_idxs)
    return dict_users


def creditcard_noniid(dataset, num_users, type, list_ratio):
    """
    以非 I.I.D. 方式从信用卡数据集中划分客户端数据

    :param dataset: 数据集
    :param num_users: 客户端数量（划分份数）
    :param type: 划分方式:
                 1 - 按样本数量比例划分
                 2 - 按欺诈样本数量比例划分
                 3 - 按欺诈金额进行划分（仅支持 3 份）
    :param list_ratio: 各客户端的划分比例
    :return: 每个客户端分配到的样本索引集合
    """
    if type not in (1,2,3):
        raise Exception("creditcard_noniid type param error")
    if num_users!= len(list_ratio):
        raise Exception("creditcard_noniid scale param error")

    all_idxs = np.arange(len(dataset))
    dict_users = {}
    dict_users0 = {}
    dict_users1 = {}

    if type==1: # 按数据集样本数量比例划分
        np.random.seed(123)
        for i in range(len(list_ratio)-1):
            num_items = int(len(dataset)*list_ratio[i]/sum(list_ratio))
            dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])
        dict_users[i+1] = set(all_idxs)

    elif type==2: # 按欺诈样本数量划分
        if num_users != 3 or len(list_ratio) != 3:
            raise Exception("creditcard_noniid scale param error")
        data = dataset.data
        data0 = data[data.Class == 0]
        data1 = data[data.Class == 1]
        idxs_0 = np.where(dataset.labels == 0)[0]
        idxs_1 = np.where(dataset.labels == 1)[0]

        # 先划分欺诈样本
        for i in range(len(list_ratio) - 1):
            num_items1 = int(len(data1) * list_ratio[i] / sum(list_ratio))
            dict_users1[i] = set(np.random.choice(idxs_1, num_items1, replace=False))
            idxs_1 = list(set(idxs_1) - dict_users1[i])
        dict_users1[i + 1] = set(idxs_1)

        # 再划分正常样本，使每个客户端样本数量接近
        for i in range(len(list_ratio) - 1):
            num_items0 = int(len(data)/num_users - len(dict_users1[i]))
            dict_users0[i] = set(np.random.choice(idxs_0, num_items0, replace=False))
            idxs_0 = list(set(idxs_0) - dict_users0[i])
        dict_users0[i + 1] = set(idxs_0)

        dict_users = [set(list(dict_users0[i]) + list(dict_users1[i])) for i in range(num_users)]

    elif type==3: # 按欺诈金额划分（仅支持 3 个客户端）
        if num_users != 3 or len(list_ratio) != 3:
            raise Exception("creditcard_noniid scale param error")

        data = dataset.data
        data0 = data[data.Class == 0]
        data1 = data[data.Class == 1].sort_values(by='Amount')

        all_amount = sum(data1['Amount'])  # 总欺诈金额
        list_amount = [n / sum(list_ratio) * all_amount for n in list_ratio]  # 每个节点应分配的欺诈金额
        list_amount0 = list_amount.copy()

        dict_users1 = {0: [], 1: [], 2: []}  # 每个节点的欺诈样本

        # 逐个样本按金额填充
        while (1):
            if len(data1) == 0:
                break
            if list_amount[0] > 0 and len(data1) > 0:
                dict_users1[0].append(data1.iloc[0].name)
                list_amount[0] -= data1.iloc[0].Amount
                data1 = data1.drop(index=data1.iloc[0].name)
            if list_amount[1] > 0 and len(data1) > 0:
                dict_users1[1].append(data1.iloc[0].name)
                list_amount[1] -= data1.iloc[0].Amount
                data1 = data1.drop(index=data1.iloc[0].name)
            if list_amount[2] > 0 and len(data1) > 0:
                dict_users1[2].append(data1.iloc[0].name)
                list_amount[2] -= data1.iloc[0].Amount
                data1 = data1.drop(index=data1.iloc[0].name)

        list_amount1 = [list_amount0[i] - list_amount[i] for i in range(3)]
        print("欺诈金额分配比例：", list_ratio)
        print("实际每个节点分配到的欺诈金额:", list_amount1)

        dict_users0 = {0: [], 1: [], 2: []}
        sample_num = int(len(data)/3)
        data0_indexs = data0.index

        for i in range(2):
            dict_users0[i] = set(np.random.choice(data0_indexs, sample_num-len(dict_users1[i]), replace=False))
            data0_indexs = list(set(data0_indexs) - dict_users0[i])
        dict_users0[2] = data0_indexs

        dict_users = [set(list(dict_users0[i]) + list(dict_users1[i])) for i in range(3)]

    else:
        raise Exception("creditcard 数据集 non-iid 分割：type 参数错误")

    return dict_users
