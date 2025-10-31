# 导入库
import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, OrderedDict, Tuple, Optional, Any

# 自定义库
from utils.options import lth_args_parser
from utils.train_utils import prepare_dataloaders, get_data
from pflopt.optimizers import MaskLocalAltSGD, local_alt
from lottery_ticket import init_mask_zeros, delta_update
from broadcast import (
    FusionModule,
    broadcast_server_to_client_initialization,
    div_server_weights,
    add_masks,
    add_server_weights,
)
import random
from torchvision.models import resnet18


def evaluate(
    model: nn.Module, ldr_test: torch.utils.data.DataLoader, args: Any
) -> float:
    """在测试数据集上评估模型准确率。

    参数:
        model: 要评估的神经网络模型
        ldr_test: 测试集 DataLoader
        args: 包含设备信息的参数

    返回:
        float: 测试集平均准确率
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    average_accuracy = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(ldr_test):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            acc = pred.eq(target.view_as(pred)).sum().item() / len(data)
            average_accuracy += acc
        average_accuracy /= len(ldr_test)
    return average_accuracy


def train_personalized(
    model: nn.Module,
    ldr_train: torch.utils.data.DataLoader,
    mask: OrderedDict,
    args: Any,
    initialization: Optional[OrderedDict] = None,
    verbose: bool = False,
    eval: bool = True,
) -> Tuple[nn.Module, float]:
    """使用个性化的局部交替优化训练模型。

    参数:
        model: 要训练的神经网络模型
        ldr_train: 训练数据 DataLoader
        mask: 参数的二值 mask
        args: 训练参数
        initialization: 可选的初始模型状态
        verbose: 是否输出训练过程
        eval: 是否在训练中评估模型

    返回:
        包含：
            - 训练后的模型
            - 最终训练损失
    """
    if initialization is not None:
        model.load_state_dict(initialization)
    optimizer = MaskLocalAltSGD(model.parameters(), mask, lr=args.lr)
    epochs = args.la_epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    train_loss = 0
    with tqdm(total=epochs) as pbar:
        for i in range(epochs):
            train_loss = local_alt(
                model,
                criterion,
                optimizer,
                ldr_train,
                device,
                clip_grad_norm=args.clipgradnorm,
            )
            if verbose:
                print(f"Epoch: {i} \tLoss: {train_loss}")
            pbar.update(1)
            pbar.set_postfix({"Loss": train_loss})
    return model, train_loss


def fedselect_algorithm(
    model: nn.Module,
    args: Any,
    dataset_train: torch.utils.data.Dataset,
    dataset_test: torch.utils.data.Dataset,
    dict_users_train: Dict[int, np.ndarray],
    dict_users_test: Dict[int, np.ndarray],
    labels: np.ndarray,
    idxs_users: List[int],
) -> Dict[str, Any]:
    """FedSelect 联邦学习主算法。

    参数:
        model: 神经网络模型
        args: 训练参数
        dataset_train: 训练数据集
        dataset_test: 测试数据集
        dict_users_train: 用户到训练数据索引的映射
        dict_users_test: 用户到测试数据索引的映射
        labels: 数据标签
        idxs_users: 用户 ID 列表

    返回:
        字典，包含：
            - client_accuracies: 每轮每个客户端的准确率
            - labels: 数据标签
            - client_masks: 最终客户端 mask
            - args: 训练参数
            - cross_client_acc: 跨客户端准确率矩阵
            - lth_convergence: 彩票票据收敛历史
    """
    # 初始化模型
    initial_state_dict = copy.deepcopy(model.state_dict())
    com_rounds = args.com_rounds
    # 初始化服务器
    client_accuracies = [{i: 0 for i in idxs_users} for _ in range(com_rounds)]
    client_state_dicts = {i: copy.deepcopy(initial_state_dict) for i in idxs_users}
    client_state_dict_prev = {i: copy.deepcopy(initial_state_dict) for i in idxs_users}
    client_masks = {i: None for i in idxs_users}
    client_masks_prev = {i: init_mask_zeros(model) for i in idxs_users}
    server_accumulate_mask = OrderedDict()
    server_weights = OrderedDict()
    lth_iters = args.lth_epoch_iters
    prune_rate = args.prune_percent / 100
    prune_target = args.prune_target / 100
    lottery_ticket_convergence = []
    client_delta_tensors = {i: None for i in idxs_users}

    # 开始联邦学习
    for round_num in range(com_rounds):
        round_loss = 0
        for i in idxs_users:
            # 初始化模型
            model.load_state_dict(client_state_dicts[i])
            # 获取数据
            ldr_train, _ = prepare_dataloaders(
                dataset_train,
                dict_users_train[i],
                dataset_test,
                dict_users_test[i],
                args,
            )
            # 本地更新 LTN_i
            client_mask = client_masks_prev.get(i)
            # 本地更新 u_i 参数（0 为全局，1 为本地）
            client_model, loss = train_personalized(model, ldr_train, client_mask, args)
            round_loss += loss
            # 发送 u_i 更新给服务器
            if round_num < com_rounds - 1:
                server_accumulate_mask = add_masks(server_accumulate_mask, client_mask)
                server_weights = add_server_weights(
                    server_weights, client_model.state_dict(), client_mask
                )
            client_state_dicts[i] = copy.deepcopy(client_model.state_dict())
            client_masks[i] = copy.deepcopy(client_mask)

            if round_num % lth_iters == 0 and round_num != 0:
                client_mask, delta_tensor_dict = delta_update(
                    prune_rate,
                    client_state_dicts[i],
                    client_state_dict_prev[i],
                    client_masks_prev[i],
                    bound=prune_target,
                    invert=True,
                )
                client_state_dict_prev[i] = copy.deepcopy(client_state_dicts[i])
                client_masks_prev[i] = copy.deepcopy(client_mask)
                client_delta_tensors[i] = copy.deepcopy(delta_tensor_dict)

        round_loss /= len(idxs_users)
        cross_client_acc = cross_client_eval(
            model,
            client_state_dicts,
            dataset_train,
            dataset_test,
            dict_users_train,
            dict_users_test,
            args,
        )

        accs = torch.diag(cross_client_acc)
        for i in range(len(accs)):
            client_accuracies[round_num][i] = accs[i]
        print("Client Accs: ", accs, " | Mean: ", accs.mean())

        if round_num < com_rounds - 1:
            # 服务器对 u_i 求平均
            server_weights = div_server_weights(server_weights, server_accumulate_mask)
            # 服务器将非 Lottery Ticket 的参数广播到每个设备
            print(f"round_num is {round_num}")
            for i in idxs_users:
                fushion_module = FusionModule(client_state_dicts[i],client_delta_tensors[i])
                client_state_dicts[i] = broadcast_server_to_client_initialization(
                    server_weights, client_masks[i], client_state_dicts[i], client_delta_tensors[i], fusion_module=fushion_module
                )
            server_accumulate_mask = OrderedDict()
            server_weights = OrderedDict()

    cross_client_acc = cross_client_eval(
        model,
        client_state_dicts,
        dataset_train,
        dataset_test,
        dict_users_train,
        dict_users_test,
        args,
        no_cross=False,
    )

    out_dict = {
        "client_accuracies": client_accuracies,
        "labels": labels,
        "client_masks": client_masks,
        "args": args,
        "cross_client_acc": cross_client_acc,
        "lth_convergence": lottery_ticket_convergence,
    }

    return out_dict


def cross_client_eval(
    model: nn.Module,
    client_state_dicts: Dict[int, OrderedDict],
    dataset_train: torch.utils.data.Dataset,
    dataset_test: torch.utils.data.Dataset,
    dict_users_train: Dict[int, np.ndarray],
    dict_users_test: Dict[int, np.ndarray],
    args: Any,
    no_cross: bool = True,
) -> torch.Tensor:
    """跨客户端评估模型。

    参数:
        model: 神经网络模型
        client_state_dicts: 每个客户端的模型状态
        dataset_train: 训练数据集
        dataset_test: 测试数据集
        dict_users_train: 客户端到训练数据的映射
        dict_users_test: 客户端到测试数据的映射
        args: 参数
        no_cross: 是否只在自己的数据上评估

    返回:
        torch.Tensor: 跨客户端准确率矩阵
    """
    cross_client_acc_matrix = torch.zeros(
        (len(client_state_dicts), len(client_state_dicts))
    )
    idx_users = list(client_state_dicts.keys())
    for _i, i in enumerate(idx_users):
        model.load_state_dict(client_state_dicts[i])
        for _j, j in enumerate(idx_users):
            if no_cross:
                if i != j:
                    continue
            # 用客户端 i 的模型评估客户端 j 的数据
            _, ldr_test = prepare_dataloaders(
                dataset_train,
                dict_users_train[j],
                dataset_test,
                dict_users_test[j],
                args,
            )
            acc = evaluate(model, ldr_test, args)
            cross_client_acc_matrix[_i, _j] = acc
    return cross_client_acc_matrix


def get_cross_correlation(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """使用 F.conv2d 计算两个张量的互相关。

    参数:
        A: 第一个张量
        B: 第二个张量

    返回:
        torch.Tensor: 互相关结果
    """
    A = A.cuda() if torch.cuda.is_available() else A
    B = B.cuda() if torch.cuda.is_available() else B
    A = A.unsqueeze(0).unsqueeze(0)
    B = B.unsqueeze(0).unsqueeze(0)
    A = A / (A.max() - A.min()) if A.max() - A.min() != 0 else A
    B = B / (B.max() - B.min()) if B.max() - B.min() != 0 else B
    return F.conv2d(A, B)


def run_base_experiment(model: nn.Module, args: Any) -> None:
    """运行基础联邦学习实验。

    参数:
        model: 神经网络模型
        args: 实验参数
    """
    dataset_train, dataset_test, dict_users_train, dict_users_test, labels = get_data(
        args
    )

    idxs_users = np.arange(args.num_users * args.frac)
    m = max(int(args.frac * args.num_users), 1)
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    idxs_users = [int(i) for i in idxs_users]
    fedselect_algorithm(
        model,
        args,
        dataset_train,
        dataset_test,
        dict_users_train,
        dict_users_test,
        labels,
        idxs_users,
    )


def load_model(args: Any) -> nn.Module:
    """加载并初始化模型。

    参数:
        args: 模型参数

    返回:
        nn.Module: 初始化后的模型
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    model = resnet18(pretrained=args.pretrained_init)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, args.num_classes)
    model = model.to(device)
    return model.to(device)


def setup_seed(seed: int) -> None:
    """设置随机种子以确保可复现性。

    参数:
        seed: 随机种子值
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    # 参数解析器
    args = lth_args_parser()

    # 设置随机种子
    setup_seed(args.seed)
    model = load_model(args)

    run_base_experiment(model, args)
