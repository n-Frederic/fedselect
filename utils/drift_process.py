import torch
import numpy as np
import torch.nn as nn


def drift_detect(old_data, new_data, threshold=0.1):
    """
    返回：是否漂移, 以及构造 z 所需的统计量
    """
    old_mean = old_data.mean(dim=0)
    new_mean = new_data.mean(dim=0)
    mean_diff = (new_mean - old_mean)
    old_std = old_data.std(dim=0)
    new_std = new_data.std(dim=0)
    std_diff = (new_std - old_std)

    drift_score = mean_diff.abs().mean() + std_diff.abs().mean()
    drift_flag = drift_score.item() > threshold

    return drift_flag, mean_diff, std_diff


def compute_z(mean_diff, std_diff, z_dim=64):
    z_raw = torch.cat([mean_diff, std_diff], dim=0)   # shape (2*input_dim)
    # 降维
    if z_raw.numel() > z_dim:
        idx = torch.linspace(0, z_raw.numel()-1, z_dim).long()
        z = z_raw[idx]
    else:
        # padding
        z = torch.cat([z_raw, torch.zeros(z_dim - z_raw.numel())])
    return z


def apply_hypernet_delta(
        client_model: nn.Module,
        hypernet: nn.Module,
        mean_diff: torch.Tensor,
        std_diff: torch.Tensor,
        drift_flag: bool,
        z_dim: int = 64,
        scale: float = 0.01,
        clip_val_w: float = 1.0,
        clip_val_b: float = 0.1,
        train_hypernet: bool = False,
        criterion=None,
        batch_x=None,
        batch_y=None,
        optimizer=None,
        device='cpu'
):
    """
    给 client_model 的最后一层应用超网络生成的 Δw。
    可选训练超网络。

    返回：
        delta_vec: torch.Tensor, 超网络生成的 Δw向量
        applied_w_delta: torch.Tensor, 实际应用到权重的 Δw
        applied_b_delta: torch.Tensor or None, 实际应用到 bias 的 Δw
    """
    if not drift_flag:
        return None, None, None

    # ===== 构造 z 并自动补齐到 z_dim =====
    z_raw = torch.cat([mean_diff, std_diff], dim=0)  # shape (58,)
    if z_raw.numel() < z_dim:
        pad = torch.zeros(z_dim - z_raw.numel(), device=z_raw.device)
        z = torch.cat([z_raw, pad], dim=0).unsqueeze(0)  # shape (1, z_dim)
    else:
        z = z_raw.unsqueeze(0)

    # ===== 超网络生成 Δw =====
    delta_vec = hypernet(z).squeeze(0)  # shape (delta_dim,)
    last_layer = client_model.net[-1]

    # ===== 解析 delta_vec =====
    w = last_layer.weight.view(-1)
    num_w = w.numel()
    w_delta = delta_vec[:num_w]
    b_delta = delta_vec[num_w:num_w + last_layer.bias.numel()] if last_layer.bias is not None else None

    # ===== scale =====
    w_delta = w_delta * scale
    if b_delta is not None:
        b_delta = b_delta * scale

    # ===== clip =====
    torch.clamp_(w_delta, -clip_val_w, clip_val_w)
    if b_delta is not None:
        torch.clamp_(b_delta, -clip_val_b, clip_val_b)

    # ===== 应用到模型 =====
    with torch.no_grad():
        last_layer.weight.copy_((w + w_delta).view_as(last_layer.weight))
        if b_delta is not None:
            last_layer.bias.copy_(last_layer.bias + b_delta)

    applied_w_delta = w_delta.clone()
    applied_b_delta = b_delta.clone() if b_delta is not None else None

    # ===== Debug 信息 =====
    print("\n========== DEBUG Δw BEGIN ==========")
    print(f"[DEBUG-DELTA] z.shape = {z.shape}, z.norm = {z.norm().item():.4f}")
    print(f"[DEBUG-DELTA] delta_vec.shape = {delta_vec.shape}, delta_vec.norm = {delta_vec.norm().item():.4f}")
    print(f"[DEBUG-DELTA] w_delta.shape = {w_delta.shape}, w_delta.norm = {w_delta.norm().item():.4f}")
    print(f"[DEBUG-DELTA] w_delta[:10] = {w_delta[:10].detach().cpu().numpy()}")
    if applied_b_delta is not None:
        print(f"[DEBUG-DELTA] b_delta.shape = {b_delta.shape}, b_delta.norm = {b_delta.norm().item():.4f}")
        print(f"[DEBUG-DELTA] b_delta[:10] = {b_delta[:10].detach().cpu().numpy()}")
    print(f"[DEBUG-DELTA] last_layer.weight.norm = {last_layer.weight.norm().item():.4f}")
    if last_layer.bias is not None:
        print(f"[DEBUG-DELTA] last_layer.bias.norm = {last_layer.bias.norm().item():.4f}")
    print("========== DEBUG Δw END ==========\n")

    # ===== 可选训练超网络 =====
    if train_hypernet and batch_x is not None and batch_y is not None and criterion is not None and optimizer is not None:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = client_model(batch_x)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(hypernet.parameters(), max_norm=1.0)
        optimizer.step()
        print(f"[DEBUG-HYPER] Trained hypernet on this batch, loss = {loss.item():.6f}")

    return delta_vec, applied_w_delta, applied_b_delta
