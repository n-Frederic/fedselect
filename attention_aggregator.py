import torch
import torch.nn.functional as F
from collections import OrderedDict
from typing import List


def _flatten_weights(state_dict: OrderedDict, relevant_keys: List[str]) -> torch.Tensor:
    """将模型权重展平为一维张量。"""
    flat_weights = []
    for key in relevant_keys:
        if key in state_dict:
            flat_weights.append(torch.flatten(state_dict[key].cpu()))
    return torch.cat(flat_weights)


def attention_based_aggregation(
        server_weights: OrderedDict,
        client_weights_list: List[OrderedDict],
        client_masks_list: List[OrderedDict]
) -> OrderedDict:
    """
    使用基于注意力机制的加权平均来聚合客户端权重。
    注意力分数基于客户端和服务器模型之间全局参数的余弦相似度。

    Args:
        server_weights (OrderedDict): 服务器当前的全局模型状态字典。
        client_weights_list (List[OrderedDict]): 来自客户端的状态字典列表。
        client_masks_list (List[OrderedDict]): 来自客户端的掩码列表。

    Returns:
        OrderedDict: 更新后的新全局模型状态字典。
    """
    num_clients = len(client_weights_list)
    if num_clients == 0:
        return server_weights

    # 我们只关心全局参数（掩码中为0的部分）。
    # 首先，获取所有可训练参数（权重和偏置）的键名。
    all_param_keys = [key for key in server_weights.keys() if "weight" in key or "bias" in key]

    # 将服务器模型的全局参数展平，作为参考向量。
    # 注意：这里的掩码是“局部参数掩码”，所以值为0表示全局参数。
    # 我们用第一个客户端的掩码来确定哪些是全局参数键。
    global_param_keys = [key for key in all_param_keys if torch.all(client_masks_list[0][key] == 0)]
    server_global_flat = _flatten_weights(server_weights, global_param_keys)

    attention_scores = []
    for client_weights in client_weights_list:
        # 将每个客户端模型的全局参数展平
        client_global_flat = _flatten_weights(client_weights, global_param_keys)

        # 计算余弦相似度作为注意力分数
        score = F.cosine_similarity(server_global_flat.unsqueeze(0), client_global_flat.unsqueeze(0))
        attention_scores.append(score)

    # 使用Softmax归一化分数，得到最终的注意力权重
    attention_scores_tensor = torch.tensor(attention_scores)
    attention_weights = F.softmax(attention_scores_tensor, dim=0)

    print(f"服务器端注意力权重: {attention_weights.tolist()}")

    # 使用注意力权重进行加权平均，计算新的全局模型
    new_server_weights = OrderedDict()

    # 1. 累加所有客户端的加权全局参数
    for key in server_weights.keys():
        if key in global_param_keys:
            # 初始化为零
            weighted_sum = torch.zeros_like(server_weights[key])
            for i in range(num_clients):
                weighted_sum += attention_weights[i] * client_weights_list[i][key]
            new_server_weights[key] = weighted_sum

    # 2. 从旧的服务器模型中复制非参数层（如BatchNorm的running_mean）和所有客户端的局部参数
    # 因为广播函数只会覆盖全局参数，所以客户端自己的局部参数不会受影响。
    # 我们只需要确保新的server_weights字典是完整的。
    for key in server_weights.keys():
        if key not in new_server_weights:
            new_server_weights[key] = server_weights[key]

    return new_server_weights