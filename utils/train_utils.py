from torchvision import datasets, transforms
from utils.sampling import iid, noniid, creditcard_iid, creditcard_noniid
from utils.dataprocess import DatasetFromCSV
import numpy as np
import torch
from typing import Dict, List, Tuple, Any


class DatasetSplit(torch.utils.data.Dataset):
    """Custom Dataset class that returns a subset of another dataset based on indices.

    Args:
        dataset: The base dataset to sample from
        idxs: Indices to use for sampling from the base dataset
    """

    def __init__(self, dataset: torch.utils.data.Dataset, idxs: List[int]) -> None:
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, int]:
        image, label = self.dataset[self.idxs[item]]
        return image, label


trans_mnist = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
trans_cifar10_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
trans_cifar10_val = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
trans_cifar100_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    ]
)
trans_cifar100_val = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    ]
)


def get_data(
    args: Any,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, Dict, Dict, np.ndarray]:
    """Get train and test datasets and user splits for federated learning.

    Args:
        args: Arguments containing dataset configuration

    Returns:
        dataset_train: Training dataset
        dataset_test: Test dataset
        dict_users_train: Dictionary mapping users to training data indices
        dict_users_test: Dictionary mapping users to test data indices
        rand_set_all: Random set assignments for non-iid splitting
    """
    if args.dataset == 'creditcard':
        dataset_train = DatasetFromCSV('./data/creditcard/creditcard.csv', train=True)
        dataset_test = DatasetFromCSV('./data/creditcard/creditcard.csv', train=False)

        if args.iid:
            dict_users_train = creditcard_iid(dataset_train, args.num_users)
        else:
            dict_users_train = creditcard_noniid(dataset_train, args.num_users, type=args.split_dataset_type,
                                                 list_ratio=args.split_dataset_ratio)

        print("\n--- Client Data Distribution Verification ---")
        # 直接从 dataset_train 对象中获取用于统计的原始 DataFrame
        stats_df = dataset_train.data
        for client_id in range(args.num_users):
            # 获取分配给该客户端的数据索引
            client_indices = list(dict_users_train[client_id])
            # 从 DataFrame 中选出该客户端的实际数据
            user_data = stats_df.loc[client_indices]

            total_samples = len(user_data)
            fraud_data = user_data[user_data.Class == 1]
            fraud_count = len(fraud_data)

            # 避免在没有欺诈样本时出现除以零的错误
            if fraud_count == 0:
                avg_fraud_amount = 0
            else:
                avg_fraud_amount = fraud_data.Amount.mean()

            print(
                f"第{client_id}个客户端: 总样本数量为{total_samples}，欺诈样本数量为{fraud_count}，平均欺诈样本金额为{avg_fraud_amount}")
        print("--- End of Verification ---\n")
        # ---  为测试集划分数据  ---
        print("为测试集创建用户数据划分...")
        if args.iid:
            dict_users_test = creditcard_iid(dataset_test, args.num_users)
        else:
            dict_users_test = creditcard_noniid(dataset_test, args.num_users,
                                                type=getattr(args, 'split_dataset_type', 'type1'),
                                                list_ratio=getattr(args, 'split_dataset_ratio', [0.1, 0.2, 0.7]))

        # ---  从 DataFrame 中获取标签  ---
        labels = dataset_train.data['Class'].values

        return dataset_train, dataset_test, dict_users_train, dict_users_test, labels
    elif args.dataset == 'cifar10':
        dataset_train = datasets.CIFAR10(
            "data/cifar10", train=True, download=True, transform=trans_cifar10_train
        )
        dataset_test = datasets.CIFAR10(
            "data/cifar10", train=False, download=True, transform=trans_cifar10_val
        )
        if args.iid:
            dict_users_train = iid(dataset_train, args.num_users)
            dict_users_test = iid(dataset_test, args.num_users)
            rand_set_all = np.array([])
        else:
            dict_users_train, rand_set_all = noniid(
                dataset_train,
                args.num_users,
                args.shard_per_user,
                args.server_data_ratio,
                size=args.num_samples,
            )
            dict_users_test, rand_set_all = noniid(
                dataset_test,
                args.num_users,
                args.shard_per_user,
                args.server_data_ratio,
                size=args.test_size,
                rand_set_all=rand_set_all,
            )

        return dataset_train, dataset_test, dict_users_train, dict_users_test, rand_set_all


def prepare_dataloaders(
    dataset_train: torch.utils.data.Dataset,
    dict_users_train: Dict,
    dataset_test: torch.utils.data.Dataset,
    dict_users_test: Dict,
    args: Any,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Prepare train and test data loaders for a user.

    Args:
        dataset_train: Training dataset
        dict_users_train: Dictionary mapping users to training data indices
        dataset_test: Test dataset
        dict_users_test: Dictionary mapping users to test data indices
        args: Arguments containing batch size configuration

    Returns:
        ldr_train: Training data loader
        ldr_test: Test data loader
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

