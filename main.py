import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from unet3 import UNet
from dataloader import PatchDataset
from train2 import train, combined_loss
from test2 import test
from corruption_detector import (
    extract_specific_features,
    initialize_mahalanobis,
    calculate_threshold,
    monitor_patch_mahalanobis_change,
    calculate_mahalanobis_distances,
    load_weights
)
import random


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def plot_dice_curves(akoya, scanner_b, augmented, save_dir):
    """
    绘制三个阶段的测试 Dice 曲线，Akoya 覆盖所有 epoch，Scanner B 从 100 开始，Augmented Scanner B 从 200 开始。
    """
    # 设置每个阶段的 x 轴范围
    epochs_akoya = range(0, len(akoya))  # Akoya 从 0 到 300
    epochs_scanner_b = range(70, 70 + len(scanner_b))  # Scanner B 从 100 到 300
    epochs_augmented = range(140, 140 + len(augmented))  # Augmented Scanner B 从 200 到 300

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(epochs_akoya, akoya, label="Test Dice - Akoya")
    plt.plot(epochs_scanner_b, scanner_b, label="Test Dice - Scanner B")
    plt.plot(epochs_augmented, augmented, label="Test Dice - Augmented Scanner B")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Coefficient")
    plt.title("Test Dice Coefficient Across Training Phases")
    plt.legend()
    plt.grid(True)

    # 保存曲线图
    save_path = os.path.join(save_dir, "test_dice_curves_batch01_3.png")
    plt.savefig(save_path)
    print(f"Test Dice curves saved to {save_path}")
    plt.show()
def plot_mahalanobis_distances(epochs, distances, save_dir):
    """
    绘制每个 epoch 的平均马氏距离曲线（基于 Akoky 统计信息计算），
    不同阶段使用不同数据集：
      - 第一阶段：Akoky，
      - 第二阶段：Scanner B，
      - 第三阶段：Scanner B 增强版。
    """
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, distances, marker='o', linestyle='-', label='Mahalanobis Distance')
    plt.xlabel("Epoch")
    plt.ylabel("Average Mahalanobis Distance (vs. Akoky)")
    plt.title("Mahalanobis Distance Evolution (vs. Akoky baseline)")
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(save_dir, "mahalanobis_distance_evolution1.png")
    plt.savefig(save_path)
    print(f"Mahalanobis distance evolution plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    # 路径配置
    train_dir_akoya = "/local/scratch/jliu/aggc/train_patches/Akoya"
    train_dir_scanner_b = "/local/scratch/jliu/aggc/train_patches/KFBio"
    test_dir_akoya = "/local/scratch/jliu/aggc/test_patches/Akoya"
    test_dir_scanner_b = "/local/scratch/jliu/aggc/test_patches/KFBio"
    save_dir = "/gris/gris-f/homelv/jliu/pvc/predictions"
    # checkpoint_path = "/gris/gris-f/homelv/jliu/pvc/predictions/unet_model_batch.pth"
    checkpoint_path = "/gris/gris-f/homelv/jliu/pvc/predictions/unet_model_batch_seed2.pth"

    os.makedirs(save_dir, exist_ok=True)

    # 参数配置
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    epochs = 210
    n_components = 50  # PCA 降维维度

    # 数据预处理和增强
    data_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    augment_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ColorJitter(brightness=2.0, contrast=0.8, saturation=0.8, hue=0.2),
        transforms.ToTensor(),
    ])

    # 数据集加载
    train_loader_akoya = torch.utils.data.DataLoader(
        PatchDataset(train_dir_akoya, transform=data_transform, mask_transform=mask_transform),
        batch_size=batch_size, shuffle=True, num_workers=4
    )

    train_loader_scanner_b = torch.utils.data.DataLoader(
        PatchDataset(train_dir_scanner_b, transform=data_transform, mask_transform=mask_transform),
        batch_size=batch_size, shuffle=True, num_workers=4
    )

    train_loader_augmented_scanner_b = torch.utils.data.DataLoader(
        PatchDataset(train_dir_scanner_b, transform=augment_transform, mask_transform=mask_transform),
        batch_size=batch_size, shuffle=True, num_workers=4
    )

    test_loader_akoya = torch.utils.data.DataLoader(
        PatchDataset(test_dir_akoya, transform=data_transform, mask_transform=mask_transform),
        batch_size=batch_size, shuffle=False, num_workers=4
    )

    test_loader_scanner_b = torch.utils.data.DataLoader(
        PatchDataset(test_dir_scanner_b, transform=data_transform, mask_transform=mask_transform),
        batch_size=batch_size, shuffle=False, num_workers=4
    )

    test_loader_augmented_scanner_b = torch.utils.data.DataLoader(
        PatchDataset(test_dir_scanner_b, transform=augment_transform, mask_transform=mask_transform),
        batch_size=batch_size, shuffle=False, num_workers=4
    )

    # 初始化模型
    model = UNet(n_channels=3, n_classes=1).to(device)
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=50)

    # 加载模型
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully.")
    else:
        print("No checkpoint found. Training from scratch.")

    # 初始化变量
    pca, train_mean, train_cov_inv = None, None, None
    pca_akoya, train_mean_akoya, train_cov_inv_akoya, global_threshold_akoya = None, None, None, None
    temporary_threshold = None
    global_threshold = None
    accumulated_distances = []  # 存储当前阶段的马氏距离
    test_dice_akoya, test_dice_scanner_b, test_dice_augmented = [], [], []
    test_dice_akoya2, test_dice_scanner_b2, test_dice_augmented2 = [], [], []
    test_dice_akoya_during_scanner_b, test_dice_akoya_during_augmented = [], []
    test_dice_scanner_b_during_augmented = []
    best_test_dice = 0.0
    best_test_loss = float('inf')
    threshold_list_akoya, threshold_list_scanner_b = [], []
    akoya_thresholds = []
    global_threshold_akoya = None  # Akoya 阶段的阈值（不用于 Mahalanobis 计算）
    global_threshold_scanner_b = None  # Scanner B 阶段的阈值
    global_threshold_augmented = None  # Scanner B 强化阶段的阈值
    mahalanobis_distances = []


    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # 切换训练阶段和数据集
        if epoch < epochs // 3:
            current_loader = train_loader_akoya
            current_test_loader = test_loader_akoya
            print("Training on Akoya data.")
            print("Initializing PCA and Mahalanobis detector...")

        elif epochs // 3 <= epoch < 2 * epochs // 3:
            current_loader = train_loader_scanner_b
            current_test_loader = test_loader_scanner_b
            print("Switching to Scanner B data.")
            print("Reinitializing PCA and Mahalanobis detector for Scanner B...")

        elif 2 * epochs // 3 <= epoch <= epochs:
            current_loader = train_loader_augmented_scanner_b
            current_test_loader = test_loader_augmented_scanner_b
            print("Switching to Augmented Scanner B data.")
            print("Reinitializing PCA and Mahalanobis detector for Augmented Scanner B...")

        if epoch < epochs // 3:
            print("Finalizing Akoya phase. Computing PCA and Mahalanobis statistics...")
            train_features_akoya, _ = extract_specific_features(model, current_loader, device, "second_up")
            pca_akoya, train_mean_akoya, train_cov_inv_akoya = initialize_mahalanobis(train_features_akoya,
                                                                                      n_components)
            temp_threshold = calculate_threshold(
                model, current_loader, device, pca_akoya, train_mean_akoya, train_cov_inv_akoya,
                selected_feature="second_up", threshold_multiplier=1.0
            )
            akoya_thresholds.append(temp_threshold)
            print(f"Epoch {epoch + 1}: Temporary Akoya Threshold = {temp_threshold:.4f}")

        # **在 Akoya 训练阶段的最后一个 epoch 计算最终的全局阈值**
        if epoch == epochs // 3 - 1:
            print(f"All Akoya Thresholds: {akoya_thresholds}")
            global_threshold_akoya = np.mean(akoya_thresholds)
            print(f"Final Global Akoya Threshold: {global_threshold_akoya:.4f}")

        if epoch == epochs // 3 - 1:
            print("Finalizing Akoya phase. Computing PCA and Mahalanobis statistics...")

            # **提取 Akoya 训练集的特征**
            train_features_akoya, _ = extract_specific_features(model, current_loader, device, "second_up")
            # **计算 PCA 变换并获得 Mahalanobis 统计信息**
            pca_akoya, train_mean_akoya, train_cov_inv_akoya = initialize_mahalanobis(train_features_akoya,
                                                                                      n_components)
            #  计算 Scanner B 和 Akoya 之间的 Mahalanobis 距离**
            print("Computing Scanner B threshold based on Akoya statistics...")
            train_features_scanner_b, _ = extract_specific_features(model, train_loader_scanner_b, device, "second_up")
            global_threshold_scanner_b = calculate_threshold(
                model, train_loader_scanner_b, device, pca_akoya, train_mean_akoya, train_cov_inv_akoya,
                selected_feature="second_up", threshold_multiplier=3.0  # 你可以调整 multiplier
            )
            print(f"Global Threshold for Scanner B: {global_threshold_scanner_b:.4f}")

            #  计算 Scanner B 强化 和 Akoya 之间的 Mahalanobis 距离**
            print("Computing Augmented Scanner B threshold based on Akoya statistics...")
            train_features_augmented, _ = extract_specific_features(model, train_loader_augmented_scanner_b, device,"second_up")
            global_threshold_augmented = calculate_threshold(
                model, train_loader_augmented_scanner_b, device, pca_akoya, train_mean_akoya, train_cov_inv_akoya,
                selected_feature="second_up", threshold_multiplier=3.0  # 你可以调整 multiplier
            )
            print(f"Global Threshold for Augmented Scanner B: {global_threshold_augmented:.4f}")

        # **在 Akoya 阶段，Mahalanobis 计算无效，因为我们还没有统计信息**
        if epoch < epochs // 3:
            current_threshold = None  # Akoya 阶段不计算 Mahalanobis 距离
        elif epochs // 3 <= epoch < 2 * epochs // 3:
            current_threshold = global_threshold_scanner_b  # **Scanner B 阶段使用 Scanner B 的阈值**
        else:
            current_threshold = global_threshold_augmented  # **Scanner B 强化阶段使用强化的阈值**

            # Training step
        train_bce_loss, train_dice_loss, train_total_loss, train_dice = train(
            model, current_loader, optimizer, device, epoch, epochs, criterion=combined_loss, scheduler=scheduler,
            pca=pca_akoya, train_mean=train_mean_akoya, train_cov_inv=train_cov_inv_akoya,
            global_threshold=current_threshold, selected_feature="second_up"
        )


        # Record test Dice coefficients for different stages
        if epoch < epochs // 3:
            test_dice_current = test(model, current_test_loader, device, epoch, epochs)
            test_dice_akoya.append(test_dice_current)
        elif epochs // 3 <= epoch < 2 * epochs // 3:
            test_dice_current = test(model, current_test_loader, device, epoch, epochs)
            test_dice_akoya_current = test(model, test_loader_akoya, device, epoch, epochs)
            test_dice_scanner_b.append(test_dice_current)
            test_dice_akoya.append(test_dice_akoya_current)  # 继续记录 Akoya 的结果
        else:
            test_dice_current = test(model, current_test_loader, device, epoch, epochs)
            test_dice_akoya_current = test(model, test_loader_akoya, device, epoch, epochs)
            test_dice_scanner_b_current = test(model, test_loader_scanner_b, device, epoch, epochs)
            test_dice_augmented.append(test_dice_current)
            test_dice_akoya.append(test_dice_akoya_current)  # 继续记录 Akoya 的结果
            test_dice_scanner_b.append(test_dice_scanner_b_current)  # 继续记录 Scanner B 的结果

        dummy_threshold = -1  # 阈值无实际影响

        if epoch < epochs / 3:
            # 第一阶段：使用 Akoky 测试集
            features, labels = extract_specific_features(model, test_loader_akoya, device, selected_feature="second_up")
            distances_arr = calculate_mahalanobis_distances(features, pca_akoya, train_mean_akoya, train_cov_inv_akoya)
            avg_distance = np.mean(distances_arr)
            print(f"Epoch {epoch + 1} [Stage 1 - Akoky]: Average Mahalanobis Distance = {avg_distance:.4f}")

        elif epoch < 2 * epochs / 3:
            # 第二阶段：使用 Scanner B 测试集
            features, labels = extract_specific_features(model, test_loader_scanner_b, device,
                                                         selected_feature="second_up")
            distances_arr = calculate_mahalanobis_distances(features, pca_akoya, train_mean_akoya, train_cov_inv_akoya)
            avg_distance = np.mean(distances_arr)
            print(f"Epoch {epoch + 1} [Stage 2 - Scanner B]: Average Mahalanobis Distance = {avg_distance:.4f}")

        else:
            # 第三阶段：使用 Scanner B 增强版测试集
            features, labels = extract_specific_features(model, test_loader_augmented_scanner_b, device,
                                                         selected_feature="second_up")
            distances_arr = calculate_mahalanobis_distances(features, pca_akoya, train_mean_akoya, train_cov_inv_akoya)
            avg_distance = np.mean(distances_arr)
            print(
                f"Epoch {epoch + 1} [Stage 3 - Scanner B Enhanced]: Average Mahalanobis Distance = {avg_distance:.4f}")

        # 将当前 epoch 的平均马氏距离保存到列表中
        mahalanobis_distances.append(avg_distance)


        # Save the best model
        if test_dice_current > best_test_dice:
            best_test_dice = test_dice_current
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Best model saved with test dice: {best_test_dice:.4f}")

    # 绘制并保存曲线图
    # 3. 训练结束后，调用绘图函数进行可视化（在主训练循环之后）
    epochs_list = list(range(1, epochs + 1))
    plot_mahalanobis_distances(epochs_list, mahalanobis_distances, save_dir)

    plot_dice_curves(test_dice_akoya, test_dice_scanner_b, test_dice_augmented, save_dir)

    print("Training and testing completed.")
