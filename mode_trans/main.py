import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from unet3 import UNet
from dataloader import PatchDataset
from train import train, combined_loss
from test2 import test
from corruption_detector import (
    extract_specific_features,
    initialize_mahalanobis,
    monitor_batch_mahalanobis_change,
    adjust_model_updates,
    save_weights,
)

def plot_dice_curves(akoya, scanner_b, augmented, save_dir):
    """
    绘制三个阶段的测试 Dice 曲线。
    Plot test Dice curves for the three stages.
    """
    epochs_akoya = range(1, len(akoya) + 1)
    epochs_scanner_b = range(len(akoya) + 1, len(akoya) + len(scanner_b) + 1)
    epochs_augmented = range(len(akoya) + len(scanner_b) + 1, len(akoya) + len(scanner_b) + len(augmented) + 1)

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
    # Save the curve plot
    save_path = os.path.join(save_dir, "test_dice_curves_batch.png")
    plt.savefig(save_path)
    print(f"Test Dice curves saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    # 路径配置
    # Path configuration
    train_dir_akoya = "/train_patches/Akoya"
    train_dir_scanner_b = "/train_patches/KFBio"
    test_dir_akoya = "/test_patches/Akoya"
    test_dir_scanner_b = "/test_patches/KFBio"
    save_dir = "/visualizations"
    checkpoint_path = "/unet_model_batch.pth"

    os.makedirs(save_dir, exist_ok=True)

    # 参数配置
    # Parameter configuration
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    epochs = 300
    n_components = 50  # PCA 降维维度 (PCA dimensionality reduction)

    # 数据预处理和增强
    # Data preprocessing and augmentation
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
        transforms.ColorJitter(brightness=0.5),
        transforms.ToTensor(),
    ])

    # 数据集加载
    # Dataset loading
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
    # Initialize model
    model = UNet(n_channels=3, n_classes=1).to(device)
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=50)

    # 加载模型
    # Load model
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully.")
    else:
        print("No checkpoint found. Training from scratch.")

        # 初始化马氏距离检测器
        # Initialise the Marginal Distance Detector
    print("Initializing Mahalanobis detector with Akoya data...")
    train_features, _ = extract_specific_features(model, train_loader_akoya, device, "second_up")
    pca, train_mean, train_cov_inv = initialize_mahalanobis(train_features, n_components)

    # 记录测试 Dice 系数
    # Record test Dice coefficient
    test_dice_akoya, test_dice_scanner_b, test_dice_augmented = [], [], []

    # 训练和测试
    # Training and testing
    best_test_loss = float('inf')
    previous_weights = save_weights(model)
    previous_distance = None

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        current_loader, current_test_loader = None, None

        if epoch < epochs // 3:
            current_loader = train_loader_akoya
            current_test_loader = test_loader_akoya
            print("Training on Akoya data.")
        elif epoch < 2 * epochs // 3:
            current_loader = train_loader_scanner_b
            current_test_loader = test_loader_scanner_b
            print("Training on Scanner B data.")
        else:
            current_loader = train_loader_augmented_scanner_b
            current_test_loader = test_loader_augmented_scanner_b
            print("Training on Augmented Scanner B data.")

        # 训练
        # Training
        train_bce_loss, train_dice_loss, train_total_loss, train_dice = train(
            model, current_loader, optimizer, device, epoch, epochs, criterion=combined_loss, scheduler=scheduler
        )

        test_dice = test(model, current_test_loader, device, epoch, epochs)

        # 记录不同阶段的测试 Dice 系数
        # Record Dice coefficients for different stages of testing
        if epoch < epochs // 3:
            test_dice_akoya.append(test_dice)
        elif epoch < 2 * epochs // 3:
            test_dice_scanner_b.append(test_dice)
        else:
            test_dice_augmented.append(test_dice)

        # 检测马氏距离变化
        # Detecting changes in the Mahalanobis distance
        current_distance, distance_change_rate = monitor_mahalanobis_change(
            epoch, model, current_loader, device, pca, train_mean, train_cov_inv, "second_up", previous_distance,
            threshold
        )

        # 根据变化率调整权重
        # Weights adjusted for rate of change
        adjusted = adjust_model_updates(
            model, optimizer, previous_weights, save_weights(model),
            distance_change_rate, threshold
        )

        # 日志
        # Logs
        if adjusted:
            print(f"Epoch {epoch}: Model updates scaled due to significant Mahalanobis distance change.")
        else:
            print(f"Epoch {epoch}: Model updates applied normally.")

        # 更新权重和距离
        # Update weights and distances
        previous_weights = save_weights(model)
        previous_distance = current_distance

        # 打印日志
        # Print logs
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Train Loss: {train_total_loss:.4f}, Train Dice: {train_dice:.4f}, Test Dice: {test_dice:.4f}, LR: {current_lr:.6f}")

        # 保存最佳模型
        # Keep the best model
        if train_total_loss < best_test_loss:
            best_test_loss = train_total_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Best model saved with train loss: {best_test_loss:.4f}")

    # 绘制并保存曲线图
    # Plot and save graphs
    plot_dice_curves(test_dice_akoya, test_dice_scanner_b, test_dice_augmented, save_dir)
    print("Training and testing completed.")

