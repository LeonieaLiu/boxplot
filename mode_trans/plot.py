import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
from unet3 import UNet


class PatchDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for img_group in os.listdir(root_dir):
            img_group_path = os.path.join(root_dir, img_group)
            if os.path.isdir(img_group_path):
                for region in ["G4"]:
                    images_dir = os.path.join(img_group_path, region, "images")
                    masks_dir = os.path.join(img_group_path, region, "masks")
                    if os.path.exists(images_dir) and os.path.exists(masks_dir):
                        for patch_file in os.listdir(images_dir):
                            if patch_file.endswith((".tif", ".tiff")):
                                image_path = os.path.join(images_dir, patch_file)
                                mask_path = os.path.join(masks_dir, patch_file)
                                if os.path.exists(mask_path):
                                    label = 1 if region == "G4" else 0
                                    self.samples.append((image_path, mask_path, label))

        print(f"Loaded {len(self.samples)} samples from {root_dir}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = transforms.ToTensor()(mask)  # Convert mask to Tensor

        return image, mask, img_path


def visualize_predictions(model, data_loader, device, num_samples=5):
    model.eval()
    displayed = 0  # Counter for displayed samples # 用于计数已显示的样本数
    with torch.no_grad():
        for i, (images, masks, img_paths) in enumerate(data_loader):
            images = images.to(device)
            masks = masks.to(device)

            # Obtain model predictions # 获取模型预测
            outputs, *_ = model(images)
            predictions = torch.sigmoid(outputs)  # Convert to probabilities
            predictions = (predictions > 0.5).float()  # Apply threshold

            # Visualize results # 可视化结果
            batch_size = images.size(0)
            for j in range(batch_size):
                if displayed >= num_samples:  # Stop if enough samples have been displayed
                    return

                fig, ax = plt.subplots(1, 3, figsize=(15, 5))

                # Original input image # 原始输入图像
                img_np = images[j].cpu().numpy().transpose(1, 2, 0)
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())  # Normalize to [0, 1]
                ax[0].imshow(img_np)
                ax[0].set_title("Input Image")
                ax[0].axis("off")

                # Ground truth mask # 真实掩码
                mask_np = masks[j].cpu().numpy().squeeze()
                ax[1].imshow(mask_np, cmap="gray")
                ax[1].set_title("Ground Truth")
                ax[1].axis("off")

                # Predicted mask # 预测掩码
                pred_np = predictions[j].cpu().numpy().squeeze()
                ax[2].imshow(pred_np, cmap="gray")
                ax[2].set_title("Prediction")
                ax[2].axis("off")

                plt.tight_layout()
                plt.show()

                displayed += 1  # Update counter

if __name__ == "__main__":
    # Path to the test set  # 测试集路径
    test_dir = ("/Akoya")
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

    # Load the model # 加载模型
    model = UNet(n_channels=3, n_classes=1).to(device)
    checkpoint_path = "/unet_model3.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully.")
    else:
        raise FileNotFoundError("Model checkpoint not found.")

    # Data loading # 数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset = PatchDataset(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

    # Visualize prediction results (display 10 samples) # 可视化预测结果 (显示10组)
    visualize_predictions(model, test_loader, device, num_samples=10)
