import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from corruption_detector import (
    save_weights,
    monitor_patch_mahalanobis_change,
    adjust_learning_rate,
)

# Dice 系数计算
def dice_coeff(input, target, epsilon=1e-6):
    input = (input > 0.5).float()
    intersection = (input * target).sum(dim=(1, 2, 3))
    union = input.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + epsilon) / (union + epsilon)
    return dice.mean().item()

# Dice 损失
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()

        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()

# 组合损失
def combined_loss(epoch, epochs, outputs, masks):
    alpha = 0.8 - 0.5 * (epoch / epochs)
    bce = nn.BCEWithLogitsLoss()(outputs, masks)
    dice = DiceLoss()(outputs, masks)
    return alpha * bce + (1 - alpha) * dice

# 训练函数
def train(
    model, train_loader, optimizer, device, epoch, epochs, criterion, scheduler=None, pca=None, train_mean=None,
    train_cov_inv=None, global_threshold=None, selected_feature="second_up", base_lr=0.01, scale_factor=0.1
):
    model.train()
    epoch_bce_loss, epoch_dice_loss, epoch_total_loss, epoch_dice_coeff = 0, 0, 0, 0
    total_batches = len(train_loader)

    anomalous_batch_count = 0
    normal_lr_batches = 0
    normal_lr_sum = 0
    max_anomalous_distance = -float('inf')
    anomalous_distances = []  # 用于存储异常 batch 的 Mahalanobis 距离

    for batch_idx, (images, masks, _) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}")):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs, _, _, _, _, _ = model(images)
        total_loss = criterion(epoch, epochs, outputs, masks)

        # **确保 is_anomalous 变量始终定义**
        is_anomalous = False  # 默认假设 batch 没有异常
        distances = []  # 避免 distances 未定义

        if pca is not None and train_mean is not None and train_cov_inv is not None and global_threshold is not None:
            is_anomalous, distances = monitor_patch_mahalanobis_change(
                model, images, device, pca, train_mean, train_cov_inv, selected_feature, global_threshold
            )

            # **如果是异常 batch，更新统计信息**
            if is_anomalous:
                anomalous_batch_count += 1
                anomalous_distances.extend(distances)  # 记录所有异常 batch 的距离
                max_anomalous_distance = max(max_anomalous_distance, max(distances))  # 记录最大异常值

                adjust_learning_rate(optimizer, base_lr, scale_factor)  # 动态调整学习率
            else:
                normal_lr_batches += 1
                normal_lr_sum += base_lr  # 记录正常学习率的 batch

        # 反向传播
        total_loss.backward()

        # 更新模型参数
        optimizer.step()

        # 仅当 batch 不是异常时恢复学习率
        if not is_anomalous:
            adjust_learning_rate(optimizer, base_lr)

        # 记录损失
        bce_loss = nn.BCEWithLogitsLoss()(outputs, masks)
        dice_loss = DiceLoss()(outputs, masks)

        epoch_bce_loss += bce_loss.item()
        epoch_dice_loss += dice_loss.item()
        epoch_total_loss += total_loss.item()

        outputs = torch.sigmoid(outputs)
        epoch_dice_coeff += dice_coeff(outputs, masks)

    # **计算平均异常 Mahalanobis 距离**
    mean_anomalous_distance = np.mean(anomalous_distances) if anomalous_distances else 0

    # 计算平均损失
    avg_bce_loss = epoch_bce_loss / total_batches
    avg_dice_loss = epoch_dice_loss / total_batches
    avg_total_loss = epoch_total_loss / total_batches
    avg_dice_coeff = epoch_dice_coeff / total_batches

    # 学习率调度器
    if scheduler:
        scheduler.step()

    avg_normal_lr = normal_lr_sum / normal_lr_batches if normal_lr_batches > 0 else 0

    # 训练日志
    print(f"Epoch {epoch + 1}/{epochs} - Anomalous Batches: {anomalous_batch_count}/{total_batches}")
    if anomalous_batch_count > 0:
        print(f"  Mean Anomalous Distance: {mean_anomalous_distance:.4f}")
        print(f"  Max Anomalous Distance: {max_anomalous_distance:.4f}")
    print(f"Epoch {epoch + 1}/{epochs} - Normal Learning Rate Batches: {normal_lr_batches}/{total_batches}")
    print(f"Epoch {epoch + 1}/{epochs} - Average Normal Learning Rate: {avg_normal_lr:.6f}")
    print(f"Epoch {epoch + 1}/{epochs} - Avg Loss: {avg_total_loss:.4f}, Avg Dice: {avg_dice_coeff:.4f}")

    return avg_bce_loss, avg_dice_loss, avg_total_loss, avg_dice_coeff
