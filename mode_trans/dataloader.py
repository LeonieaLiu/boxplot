import os
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class PatchDataset(Dataset):
    def __init__(self, root_dir, transform=None, mask_transform=None, neg_sample_ratio=1):
        """
        Initialize the dataset by loading samples from the given directory.
        根据给定目录加载样本初始化数据集。

        Args:
            root_dir (str): Path to the root directory containing the data.
            transform (callable, optional): Transformation to apply to the images.
            mask_transform (callable, optional): Transformation to apply to the masks.
            neg_sample_ratio (float, optional): Ratio of negative samples to positive samples.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.samples = []
        self.positive_samples = []
        self.negative_samples = []

        # Traverse the directory to load data
        # 遍历目录加载数据
        for img_group in os.listdir(root_dir):
            img_group_path = os.path.join(root_dir, img_group)
            if os.path.isdir(img_group_path):
                for region in ["G4", "G4_neg"]:  # G4 as positive samples, G4_neg as negative samples
                    # G4为正样本，G4_neg为负样本
                    images_dir = os.path.join(img_group_path, region, "images")
                    masks_dir = os.path.join(img_group_path, region, "masks")
                    if os.path.exists(images_dir) and os.path.exists(masks_dir):
                        for patch_file in os.listdir(images_dir):
                            if patch_file.endswith((".tif", ".tiff")):
                                image_path = os.path.join(images_dir, patch_file)
                                mask_path = os.path.join(masks_dir, patch_file)
                                if os.path.exists(mask_path):
                                    label = 1 if region == "G4" else 0
                                    sample = (image_path, mask_path, label)
                                    if label == 1:
                                        self.positive_samples.append(sample)
                                    else:
                                        self.negative_samples.append(sample)

        # Ensure the positive-to-negative sample ratio
        # 确保正负样本比例
        neg_sample_count = int(len(self.positive_samples) * neg_sample_ratio)
        self.negative_samples = random.sample(self.negative_samples, min(neg_sample_count, len(self.negative_samples)))
        self.samples = self.positive_samples + self.negative_samples
        random.shuffle(self.samples)

    def __len__(self):
        """
        Return the total number of samples.
        返回样本总数。
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieve a sample by index.
        根据索引获取一个样本。

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image, mask, and label.
        """
        img_path, mask_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask, label
