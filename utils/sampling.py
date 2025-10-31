import random
import numpy as np
import torch
from typing import Dict, List, Set, Union, Tuple
from torch.utils.data import Dataset


def iid(dataset: Dataset, num_users: int) -> Dict[int, Set[int]]:
    """Sample I.I.D. client data from dataset by randomly dividing into equal parts.

    Args:
        dataset: The full dataset to sample from
        num_users: Number of clients to divide data between

    Returns:
        Dict mapping client IDs to sets of data indices assigned to that client
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
    """Sample non-I.I.D client data from dataset by dividing data by class labels.

    Args:
        dataset: The full dataset to sample from
        num_users: Number of clients to divide data between
        shard_per_user: Number of class shards to assign to each user
        server_data_ratio: Fraction of data to reserve for server (default: 0.0)
        size: Optional size to limit each user's data to
        rand_set_all: Optional pre-defined random class assignments

    Returns:
        Tuple containing:
            - Dict mapping client IDs to arrays of assigned data indices
            - Array of random class assignments used for the split
    """
    dict_users, all_idxs = {i: np.array([], dtype="int64") for i in range(num_users)}, [
        i for i in range(len(dataset))
    ]

    targets = None
    targets = [elem[1] for elem in dataset]

    # dictionary of indices in the dataset for each label
    idxs_dict = {}
    for i in range(len(dataset)):
        label = torch.tensor(targets[i]).item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    num_classes = len(np.unique(targets))
    shard_per_class = int(shard_per_user * num_users / num_classes)
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

    if len(rand_set_all) == 0:
        rand_set_all = list(range(num_classes)) * shard_per_class
        random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))

    # divide and assign
    for i in range(num_users):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            idx = np.random.choice(len(idxs_dict[label]), replace=False)
            rand_set.append(idxs_dict[label].pop(idx))
        dict_users[i] = np.concatenate(rand_set)

    test = []
    for key, value in dict_users.items():
        x = np.unique(torch.tensor(targets)[value])
        assert (len(x)) <= shard_per_user
        test.append(value)
    test = np.concatenate(test)
    assert len(test) == len(dataset)
    assert len(set(list(test))) == len(dataset)

    if server_data_ratio > 0.0:
        dict_users["server"] = set(
            np.random.choice(
                all_idxs, int(len(dataset) * server_data_ratio), replace=False
            )
        )

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
    Sample I.I.D. client data from creditcard dataset
    :param dataset:
    :param num_users:
    :return: dict of sample index
    """
    np.random.seed(0)
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, dataset.data.index #[i for i in range(len(dataset))]
    for i in range(num_users-1):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    dict_users[i+1] = set(all_idxs)
    return dict_users


def creditcard_noniid(dataset, num_users, type, list_ratio):
    """
    Sample non-I.I.D client data from creditcard dataset
    :param dataset: 数据集
    :param num_users: 划分的用户数量，即划分为多少份
    :param type: 划分类别，1-按照样本数量划分，2-按照欺诈样本数量划分，3-按照欺诈金额划分
    :param list_ratio: 划分比例
    :return: 划分后每个客户端分配的样本index集合
    """
    if type not in (1,2,3):
        raise Exception("creditcard_noniid type param error")
    if num_users!= len(list_ratio):
        raise Exception("creditcard_noniid scale param error")
    # all_idxs = [i for i in range(len(dataset))]
    all_idxs = dataset.data.index
    dict_users = {}
    dict_users0 = {}
    dict_users1 = {}
    if type==1: #按数据集大小比例进行分割
        np.random.seed(123)
        for i in range(len(list_ratio)-1):
            num_items = int(len(dataset)*list_ratio[i]/sum(list_ratio))
            dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])
        dict_users[i+1] = set(all_idxs)
    elif type==2: #按欺诈样本比例进行分割
        if num_users != 3 or len(list_ratio) != 3:
            raise Exception("creditcard_noniid scale param error")
        data = dataset.data
        data0 = data[data.Class == 0]
        data1 = data[data.Class == 1]
        idxs_0 = data0.index
        idxs_1 = data1.index
        for i in range(len(list_ratio) - 1): #先划分欺诈样本
            num_items1 = int(len(data1) * list_ratio[i] / sum(list_ratio))
            dict_users1[i] = set(np.random.choice(idxs_1, num_items1, replace=False))
            idxs_1 = list(set(idxs_1) - dict_users1[i])
        dict_users1[i + 1] = set(idxs_1)
        for i in range(len(list_ratio) - 1): #再划分正常样本
            num_items0 = int(len(data)/num_users-len(dict_users1[i]))
            dict_users0[i] = set(np.random.choice(idxs_0, num_items0, replace=False))
            idxs_0 = list(set(idxs_0) - dict_users0[i])
        dict_users0[i + 1] = set(idxs_0)
        dict_users = [set(list(dict_users0[i]) + list(dict_users1[i])) for i in range(num_users)]
    elif type==3: #按欺诈金额进行分割，这个方式目前仅支持将数据集划分为3份
        if num_users != 3 or len(list_ratio) != 3:
            raise Exception("creditcard_noniid scale param error")
        data = dataset.data
        data0 = data[data.Class == 0]
        data1 = data[data.Class == 1]
        data1 = data1.sort_values(by='Amount')
        all_amount = sum(data1['Amount']) #总共的欺诈金额
        list_amount = [n / sum(list_ratio) * all_amount for n in list_ratio] #欺诈金额分配到三个节点
        list_amount0 = list_amount.copy() #备份应分配的金额
        dict_users1 = {0: [], 1: [], 2: []} #每个节点分配到的欺诈样本
        while (1):
            if len(data1) == 0: #若欺诈数据已分配完则退出循环
                break
            if list_amount[0] > 0 and len(data1) > 0:
                dict_users1[0].append(data1.iloc[0].name)  # 将data1中第0行数据的索引加入到集合中
                list_amount[0] = list_amount[0] - (data1.iloc[0]).Amount  # 减去已分配的金额
                data1 = data1.drop(index=data1.iloc[0].name)  # 删除data1的第0行数据
            if list_amount[1] > 0 and len(data1) > 0:
                dict_users1[1].append(data1.iloc[0].name)
                list_amount[1] = list_amount[1] - (data1.iloc[0]).Amount
                data1 = data1.drop(index=data1.iloc[0].name)
            if list_amount[2] > 0 and len(data1) > 0:
                dict_users1[2].append(data1.iloc[0].name)
                list_amount[2] = list_amount[2] - (data1.iloc[0]).Amount
                data1 = data1.drop(index=data1.iloc[0].name)
        list_amount1 = [list_amount0[i] - list_amount[i] for i in range(3)] #实际分配到每个节点的欺诈金额
        print("欺诈金额分配比例：",list_ratio)
        print("实际分配到每个节点的欺诈金额:",list_amount1)
        dict_users0 = {0: [], 1: [], 2: []} #每个节点分配到的正常样本
        sample_num = int(len(data)/3) #每个节点应分配的样本数量
        data0_indexs = data0.index
        for i in range(2): #将正常样本分配到各个节点，正常样本数量和异常样本加起来是数量平均分配的
            dict_users0[i] = set(np.random.choice(data0_indexs, sample_num-len(dict_users1[i]), replace=False))
            data0_indexs = list(set(data0_indexs) - dict_users0[i])
        dict_users0[2] = data0_indexs
        dict_users = [set(list(dict_users0[i]) + list(dict_users1[i])) for i in range(3)] #实际分配到每个节点的样本=正常样本+欺诈样本
    else:
        raise Exception("creditcard数据集non-iid分割，type参数error")
    return dict_users
