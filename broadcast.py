import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import OrderedDict


class FusionModule(nn.Module):
    def __init__(self, client_state_dict, delta_tensor_dict=None):
        super().__init__()
        self.alpha = nn.ParameterDict()
        self.key_map = {}  # 保存原始 key 和合法 key 的映射关系

        for key, param in client_state_dict.items():
            if "weight" in key or "bias" in key:
                safe_key = key.replace(".", "__")  # 替换掉点号
                self.key_map[safe_key] = key

                if delta_tensor_dict is not None and key in delta_tensor_dict:
                    delta = delta_tensor_dict[key]
                    mean, std = delta.mean(), delta.std() + 1e-8
                    normalized = (delta - mean) / std
                    init_alpha = normalized
                else:
                    init_alpha = torch.zeros_like(param)

                self.alpha[safe_key] = nn.Parameter(init_alpha)

    def forward(self, local_state_dict, global_state_dict):
        fused_state_dict = {}
        for safe_key, orig_key in self.key_map.items():
            local_param = local_state_dict[orig_key]
            global_param = global_state_dict[orig_key]
            weight_factor = torch.sigmoid(self.alpha[safe_key])
            fused_state_dict[orig_key] = weight_factor * global_param + (1 - weight_factor) * local_param

        # 其它非 weight/bias 的参数保持不变
        for key in local_state_dict.keys():
            if key not in self.key_map.values():
                fused_state_dict[key] = local_state_dict[key]

        return fused_state_dict


def broadcast_server_to_client_initialization(
        server_weights: OrderedDict[str, torch.Tensor],
        mask: OrderedDict[str, torch.Tensor],
        client_initialization: OrderedDict[str, torch.Tensor],
        delta_tensor_dict: OrderedDict[str, torch.Tensor],
        epsilon: float = 1e-8,
        fusion_module = None,
) -> OrderedDict[str, torch.Tensor]:
    """Broadcasts server weights to client initialization for non-masked parameters.

    Args:
        fusion_module:
        delta_tensor_dict:
        epsilon:
        server_weights: Server model state dict
        mask: Binary mask indicating which parameters are local (1) vs global (0)
        client_initialization: Client model state dict to update

    Returns:
        Updated client model state dict with server weights broadcast to non-masked parameters
    """
    if fusion_module is not None:
        print(">>> USING MODIFIED broadcast <<<")
        return fusion_module(client_initialization, server_weights)
    for key in client_initialization.keys():
        # only override client_initialization where mask is non-zero
        if "weight" in key or "bias" in key:
            # local_param = client_initialization[key]
            # global_param = server_weights[key]
            #
            # if delta_tensor_dict is None or key not in delta_tensor_dict:
            #     client_initialization[key][mask[key] == 0] = global_param[mask[key] == 0]
            # else:
            #     print(">>> USING MODIFIED fedselect.py <<<")
            #     delta_tensor = delta_tensor_dict[key].to(local_param.device)
            #     mean = delta_tensor.mean()
            #     std = delta_tensor.std() + epsilon
            #     normalized = (delta_tensor - mean) / std
            #     weight_factor = torch.sigmoid(normalized)
            #     weight_factor = weight_factor * (1 - mask[key].float())
            #     client_initialization[key] = weight_factor * local_param + (1 - weight_factor) * global_param
            #     print(
            #         key,
            #         weight_factor.min().item(),
            #         weight_factor.max().item(),
            #         weight_factor.mean().item()
            #     )

            client_initialization[key][mask[key] == 0] = server_weights[key][
                mask[key] == 0
            ]

    return client_initialization


def div_server_weights(
        server_weights: OrderedDict[str, torch.Tensor],
        server_mask: OrderedDict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Divides server weights by mask values where mask is non-zero.

    Args:
        server_weights: Server model state dict
        server_mask: Mask indicating number of contributions to each parameter

    Returns:
        Server weights normalized by number of contributions
    """
    for key in server_weights.keys():
        # only divide where server_mask is non-zero
        if "weight" in key or "bias" in key:
            server_weights[key][server_mask[key] != 0] /= server_mask[key][
                server_mask[key] != 0
                ]
    return server_weights


def add_masks(
        server_dict: OrderedDict[str, torch.Tensor],
        client_dict: OrderedDict[str, torch.Tensor],
        invert: bool = True,
) -> OrderedDict[str, torch.Tensor]:
    """Accumulates client masks into server mask dictionary.

    Args:
        server_dict: Server mask accumulator
        client_dict: Client mask to add
        invert: Whether to invert client mask before adding

    Returns:
        Updated server mask accumulator
    """
    for key in client_dict.keys():
        if "weight" in key or "bias" in key:
            if key not in server_dict.keys():
                server_dict[key] = 1 - client_dict[key] if invert else client_dict[key]
            else:
                server_dict[key] += (
                    (1 - client_dict[key]) if invert else client_dict[key]
                )
    return server_dict


def add_server_weights(
        server_weights: OrderedDict[str, torch.Tensor],
        client_weights: OrderedDict[str, torch.Tensor],
        client_mask: OrderedDict[str, torch.Tensor],
        invert: bool = True,
) -> OrderedDict[str, torch.Tensor]:
    """Accumulates masked client weights into server weights.

    Args:
        server_weights: Server weights accumulator
        client_weights: Client model weights to add
        client_mask: Binary mask indicating which parameters to add
        invert: Whether to invert mask before applying

    Returns:
        Updated server weights accumulator
    """
    for key in client_weights.keys():
        if "weight" in key or "bias" in key:
            mask = 1 - client_mask[key] if invert else client_mask[key]
            if key not in server_weights.keys():
                server_weights[key] = client_weights[key] * mask
            else:
                server_weights[key] += client_weights[key] * mask
    return server_weights
