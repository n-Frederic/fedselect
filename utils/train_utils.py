from torchvision import datasets, transforms
from utils.sampling import iid, noniid, creditcard_iid, creditcard_noniid
from utils.dataprocess import DatasetFromCSV
import numpy as np
import torch
from typing import Dict, List, Tuple, Any


class DatasetSplit(torch.utils.data.Dataset):
    """自定义 Dataset 类，用于根据给定索引返回原始数据集的一个子集。

    参数:
        dataset: 原始基础数据集
        idxs: 要从原始数据集中采样的索引列表
    """

    def __init__(self, dataset: torch.utils.data.Dataset, idxs: List[int]) -> None:
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, int]:
        image, label = self.dataset[self.idxs[item]]
        return image, label


# trans_mnist = transforms.Compose(
#     [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
# )
# trans_cifar10_train = transforms.Compose(
#     [
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ]
# )
# trans_cifar10_val = transforms.Compose(
#     [
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ]
# )
# trans_cifar100_train = transforms.Compose(
#     [
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
#     ]
# )
# trans_cifar100_val = transforms.Compose(
#     [
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
#     ]
# )


def get_data(
    args: Any,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, Dict, Dict, np.ndarray]:
    """根据参数获取训练集、测试集以及用于联邦学习的用户数据划分。

    参数:
        args: 包含数据集配置的参数

    返回:
        dataset_train: 训练数据集
        dataset_test: 测试数据集
        dict_users_train: 每个用户对应的训练数据索引字典
        dict_users_test: 每个用户对应的测试数据索引字典
        rand_set_all: 非IID划分时的随机分配结果
    """
    if args.dataset == 'creditcard':
        dataset_train = DatasetFromCSV('./data/creditcard/creditcard.csv', train=True)
        dataset_test = DatasetFromCSV('./data/creditcard/creditcard.csv', train=False)

        if args.iid:
            dict_users_train = creditcard_iid(dataset_train, args.num_users)
        else:
            dict_users_train = creditcard_noniid(dataset_train, args.num_users, type=args.split_dataset_type,
                                                 list_ratio=args.split_dataset_ratio)

        print("\n--- 客户端数据分布验证 ---")
        # 从 dataset_train 对象中获取真实 DataFrame 用于统计
        stats_df = dataset_train.data
        for client_id in range(args.num_users):
            # 获取当前客户端的数据索引
            client_indices = list(dict_users_train[client_id])
            # 从 DataFrame 提取该客户端的真实数据
            user_data = stats_df.loc[client_indices]

            total_samples = len(user_data)
            fraud_data = user_data[user_data.Class == 1]
            fraud_count = len(fraud_data)

            # 避免没有欺诈样本时出现除 0 错误
            if fraud_count == 0:
                avg_fraud_amount = 0
            else:
                avg_fraud_amount = fraud_data.Amount.mean()

            print(
                f"第{client_id}个客户端: 总样本 {total_samples}，欺诈样本 {fraud_count}，平均欺诈金额 {avg_fraud_amount}")
        print("--- 验证结束 ---\n")

        # --- 测试集划分 ---
        print("为测试集创建用户数据划分...")
        if args.iid:
            dict_users_test = creditcard_iid(dataset_test, args.num_users)
        else:
            dict_users_test = creditcard_noniid(dataset_test, args.num_users,
                                                type=getattr(args, 'split_dataset_type', 'type1'),
                                                list_ratio=getattr(args, 'split_dataset_ratio', [0.1, 0.2, 0.7]))

        # --- 从 DataFrame 中获取标签 ---
        labels = dataset_train.data['Class'].values

        return dataset_train, dataset_test, dict_users_train, dict_users_test, labels

    # elif args.dataset == 'cifar10':
    #     dataset_train = datasets.CIFAR10(
    #         "data/cifar10", train=True, download=True, transform=trans_cifar10_train
    #     )
    #     dataset_test = datasets.CIFAR10(
    #         "data/cifar10", train=False, download=True, transform=trans_cifar10_val
    #     )
    #     if args.iid:
    #         dict_users_train = iid(dataset_train, args.num_users)
    #         dict_users_test = iid(dataset_test, args.num_users)
    #         rand_set_all = np.array([])
    #     else:
    #         dict_users_train, rand_set_all = noniid(
    #             dataset_train,
    #             args.num_users,
    #             args.shard_per_user,
    #             args.server_data_ratio,
    #             size=args.num_samples,
    #         )
    #         dict_users_test, rand_set_all = noniid(
    #             dataset_test,
    #             args.num_users,
    #             args.shard_per_user,
    #             args.server_data_ratio,
    #             size=args.test_size,
    #             rand_set_all=rand_set_all,
    #         )
    #
    #     return dataset_train, dataset_test, dict_users_train, dict_users_test, rand_set_all


def prepare_dataloaders(
    dataset_train: torch.utils.data.Dataset,
    dict_users_train: Dict,
    dataset_test: torch.utils.data.Dataset,
    dict_users_test: Dict,
    args: Any,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """为某个用户构建训练集和测试集的 DataLoader。

    参数:
        dataset_train: 全局训练数据集
        dict_users_train: 用户对应的训练数据索引
        dataset_test: 全局测试数据集
        dict_users_test: 用户对应的测试数据索引
        args: 包含 batch 大小等配置的参数

    返回:
        ldr_train: 用户的训练 DataLoader
        ldr_test: 用户的测试 DataLoader
    """
    ldr_train = torch.utils.data.DataLoader(
        DatasetSplit(dataset_train, dict_users_train),
        batch_size=args.local_bs,
        shuffle=True,
    )
    ldr_test = torch.utils.data.DataLoader(
        DatasetSplit(dataset_test, dict_users_test),
        batch_size=args.local_bs,
        shuffle=False,
    )
    return ldr_train, ldr_test
