# 导入库
import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, OrderedDict, Tuple, Optional, Any

from utils.drift_process import drift_detect, compute_z, apply_hypernet_delta
# 自定义库
from utils.options import lth_args_parser
from utils.train_utils import prepare_dataloaders, get_data
from pflopt.optimizers import MaskLocalAltSGD, local_alt
from lottery_ticket import init_mask_zeros, delta_update
from broadcast import (
    broadcast_server_to_client_initialization,
    div_server_weights,
    add_masks,
    add_server_weights,
)
import random
from torchvision.models import resnet18
from models.nets import get_model
from models.fed_aggregation import FedAVG, FedMEAN, FedRWA, compute_risk_score
from utils.train_utils import get_client_fraud_stats
from models.hypernetwork import HyperNetwork


from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

def evaluate(
    model: nn.Module, ldr_test: torch.utils.data.DataLoader, args: Any
) -> Dict[str, Any]:
    """
    在测试数据集上评估模型，并返回包括混淆矩阵、F1、召回率等在内的多种指标。

    参数:
        model: 要评估的神经网络模型
        ldr_test: 测试集 DataLoader
        args: 包含设备信息的参数

    返回:
        一个包含多种评估指标的字典
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for data, target in ldr_test:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # 获取预测为正类(类别1)的概率，用于计算AUC
            probs = F.softmax(output, dim=1)[:, 1]
            # 获取预测的类别
            preds = output.argmax(dim=1)

            all_targets.extend(target.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 将列表转换为numpy数组以便于计算
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    metrics = {}
    try:
        # 计算混淆矩阵
        cm = confusion_matrix(all_targets, all_preds)
        # 从混淆矩阵中提取 TN, FP, FN, TP
        tn, fp, fn, tp = cm.ravel()

        metrics['cm'] = cm
        metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
        # zero_division=0 防止在某些批次中没有正样本时出现警告
        metrics['precision'] = precision_score(all_targets, all_preds, zero_division=0)
        metrics['recall'] = recall_score(all_targets, all_preds, zero_division=0)
        metrics['f1'] = f1_score(all_targets, all_preds, zero_division=0)
        metrics['tnr'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['auc'] = roc_auc_score(all_targets, all_probs)

    except ValueError:
        # 如果测试集为空或只包含一个类别，则指标计算可能会失败
        print("警告: 无法计算指标，数据集可能为空或只包含一个类别。")
        # 返回默认值
        default_metrics = {k: 0.0 for k in ['accuracy', 'precision', 'recall', 'f1', 'tnr', 'auc']}
        default_metrics['cm'] = np.zeros((2, 2))
        return default_metrics

    return metrics



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
    z_dim = 64
    delta_dim = 402
    hypernet = HyperNetwork(z_dim=z_dim, delta_dim=delta_dim).to(args.device)
    hypernet_optimizer = torch.optim.Adam(hypernet.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    client_old_data = {}
    # client_delta_tensors = {i: None for i in idxs_users}

    # 开始联邦学习
    for round_num in range(com_rounds):
        round_loss = 0
        
        # 用于风险加权聚合的变量
        # client_param_updates = {}  # 存储每个客户端的参数更新
        # client_sample_nums = {}  # 存储每个客户端的样本数量
        # client_accuracies_dict = {}  # 存储每个客户端的准确率
        # client_risk_scores = {}  # 存储每个客户端的风险分数
        
        for i in idxs_users:
            # 保存训练前的状态（用于计算参数更新）
            # state_before_training = copy.deepcopy(model.state_dict()) if hasattr(args, 'dataset') and args.dataset == 'creditcard' and getattr(args, 'fed_type', 0) in [1, 2, 3] else None
            
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
            
            # 收集客户端的风险统计信息（仅对 creditcard 数据集）
            # if hasattr(args, 'dataset') and args.dataset == 'creditcard':
            #     fraud_stats = get_client_fraud_stats(dataset_train, dict_users_train[i])
            #     client_sample_nums[i] = fraud_stats['dataset_size']
            #
            #     # 计算客户端准确率（这里简化，使用损失的倒数作为代理）
            #     # 在实际应用中，应该在验证集上评估准确率
            #     client_accuracies_dict[i] = 1.0 / (1.0 + loss)
            #
            #     # 计算风险分数
            #     risk_type = getattr(args, 'risk_type', 1)
            #     client_risk_scores[i] = compute_risk_score(
            #         dataset_size=fraud_stats['dataset_size'],
            #         fraud_count=fraud_stats['fraud_count'],
            #         normal_count=fraud_stats['normal_count'],
            #         fraud_amount=fraud_stats['fraud_amount'],
            #         risk_type=risk_type
            #     )
            #
            #     # 存储参数更新（训练后 - 训练前）
            #     param_update = OrderedDict()
            #     if state_before_training is not None:
            #         for k in client_model.state_dict().keys():
            #             param_update[k] = client_model.state_dict()[k] - state_before_training[k]
            #         client_param_updates[i] = param_update
            # else:
            #     # 非 creditcard 数据集，使用默认值
            #     client_sample_nums[i] = len(dict_users_train[i])
            #     client_accuracies_dict[i] = 1.0 / (1.0 + loss)
            #     client_risk_scores[i] = float(len(dict_users_train[i]))
            

            # try:
            #     batch = next(iter(ldr_train))
            # except StopIteration:
            #     batch = None
            #
            # if batch is not None:
            #     # 假设 batch 是 (inputs, labels)
            #     batch_x = batch[0].to(args.device)
            #     # 展平（如果输入已经是向量可以跳过）
            #     batch_x = batch_x.view(batch_x.size(0), -1)  # (N, 29)
            #     batch_y = batch[1].to(args.device)
            #     # 获取旧的 batch（第0轮没有旧数据）
            #     if round_num == 0 or (i not in client_old_data):
            #         old_batch = batch_x.clone()
            #     else:
            #         old_batch = client_old_data[i]
            #
            #     # 保存本轮 batch 供下一轮比较
            #     client_old_data[i] = batch_x.clone()
            #
            #     # 漂移检测
            #     drift_flag, mean_diff, std_diff = drift_detect(old_batch, batch_x, threshold=0.05)
            #
            #     if drift_flag:
            #         delta_vec, applied_w_delta, applied_b_delta = apply_hypernet_delta(
            #             client_model=client_model,
            #             hypernet=hypernet,
            #             mean_diff=mean_diff,
            #             std_diff=std_diff,
            #             drift_flag=drift_flag,
            #             scale=0.01,
            #             clip_val_w=1.0,
            #             clip_val_b=0.1,
            #             train_hypernet=True,
            #             criterion=criterion,
            #             batch_x=batch_x.to(args.device),
            #             batch_y=batch_y.to(args.device),
            #             optimizer=hypernet_optimizer,
            #             device=args.device
            #         )
            #
            #         print(
            #             f"[Round {round_num}] Client {i}: Δw.norm={applied_w_delta.norm():.4f}, bias Δ.norm={(applied_b_delta.norm() if applied_b_delta is not None else 0):.4f}")

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
                # client_delta_tensors[i] = copy.deepcopy(delta_tensor_dict)

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
                # fushion_module = FusionModule(client_state_dicts[i],client_delta_tensors[i])
                client_state_dicts[i] = broadcast_server_to_client_initialization(
                    server_weights, client_masks[i], client_state_dicts[i]
                )
            # 选择聚合算法
            # fed_type = getattr(args, 'fed_type', 0)  # 默认使用原始方法
            #
            # if fed_type in [1, 2, 3] and hasattr(args, 'dataset') and args.dataset == 'creditcard':
            #     # 使用新的聚合算法（FedAVG, FedMEAN, FedRWA）
            #     print(f"使用聚合算法类型: {fed_type} ({'FedAVG' if fed_type == 1 else 'FedMEAN' if fed_type == 2 else 'FedRWA'})")
            #
            #     # 准备聚合所需的数据
            #     dw_list = [client_param_updates[i] for i in idxs_users]
            #     sample_nums = [client_sample_nums[i] for i in idxs_users]
            #     accuracies = [client_accuracies_dict[i] for i in idxs_users]
            #     risk_scores = [client_risk_scores[i] for i in idxs_users]
            #
            #     # 获取一个参考客户端的状态作为基准
            #     reference_state = client_state_dicts[idxs_users[0]]
            #
            #     # 执行聚合
            #     if fed_type == 1:
            #         # FedAVG: 基于样本数量加权
            #         aggregated_state = FedAVG(reference_state, dw_list, sample_nums)
            #     elif fed_type == 2:
            #         # FedMEAN: 简单平均
            #         aggregated_state = FedMEAN(reference_state, dw_list)
            #     elif fed_type == 3:
            #         # FedRWA: 风险加权
            #         print(f"风险分数: {risk_scores}")
            #         aggregated_state = FedRWA(reference_state, dw_list, accuracies, risk_scores)
            #
            #     # 将聚合后的参数广播到所有客户端
            #     for i in idxs_users:
            #         # 应用 mask：只更新非本地参数（mask==0 的部分）
            #         for key in aggregated_state.keys():
            #             if "weight" in key or "bias" in key:
            #                 if client_masks[i] is not None and key in client_masks[i]:
            #                     # 只在 mask 为 0（全局参数）的位置更新
            #                     client_state_dicts[i][key] = torch.where(
            #                         client_masks[i][key] == 0,
            #                         aggregated_state[key],
            #                         client_state_dicts[i][key]
            #                     )
            #                 else:
            #                     # 如果没有 mask，直接使用聚合后的参数
            #                     client_state_dicts[i][key] = aggregated_state[key]
            #             else:
            #                 # 其他参数（如 BN 的 running_mean 等）直接复制
            #                 client_state_dicts[i][key] = aggregated_state[key]
            # else:
            #     # 使用原始的聚合方法（FedSelect 默认方法）
            #     server_weights = div_server_weights(server_weights, server_accumulate_mask)
            #     # 服务器将非 Lottery Ticket 的参数广播到每个设备
            #     print(f"round_num is {round_num}")
            #     for i in idxs_users:
            #         client_state_dicts[i] = broadcast_server_to_client_initialization(
            #             server_weights, client_masks[i], client_state_dicts[i]
            #         )
            #
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
    """
    跨客户端评估模型，并在评估每个客户端自身数据时打印详细指标。
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

            # 如果某个客户端没有测试数据，则跳过
            if len(ldr_test.dataset) == 0:
                metrics = {'accuracy': 0.0}
            else:
                # 调用新的evaluate函数获取所有指标
                metrics = evaluate(model, ldr_test, args)

            # 当客户端在自己的测试集上评估时，打印详细信息
            if i == j:
                cm = metrics.get('cm', np.array([['N/A', 'N/A'], ['N/A', 'N/A']]))
                # 将numpy数组格式化为单行字符串以便打印
                cm_str = np.array2string(cm, separator=', ').replace('\n', '')
                print(f"Client_{_i} test results -> cm: {cm_str}")
                print(f"  └─> Accuracy: {metrics.get('accuracy', 0.0):.4f}, "
                      f"Recall: {metrics.get('recall', 0.0):.4f}, "
                      f"F1: {metrics.get('f1', 0.0):.4f}, "
                      f"AUC: {metrics.get('auc', 0.0):.4f}")

            # 将准确率存入矩阵
            cross_client_acc_matrix[_i, _j] = metrics['accuracy']

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
    # model = resnet18(pretrained=args.pretrained_init)
    model = get_model(name=args.model, input_dim=29, num_classes=2)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, args.num_classes)
    # model = model.to(device)
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
