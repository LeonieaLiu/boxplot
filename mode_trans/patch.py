import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import argparse
from enum import Enum

class Scanner(Enum):
    AKOYA = "Akoya"
    OLYMPUS = "Olympus"
    ZEISS = "Zeiss"
    KFBIO = "KFBio"
    LEICA = "Leica"
    PHILIPS = "Philips"

class SegmentationMask(Enum):
    G3 = "G3"
    G4 = "G4"
    G5 = "G5"
    Normal = "Normal"
    Stroma = "Stroma"

def dynamic_extract_contour_patches(image, mask, patch_size, overlap_ratio=0.1, threshold=0.1, negative_samples=0,
                                    plot=False, not_plot_patches=False):
    patches = []
    patch_masks = []

    bg_patches = []
    bg_patch_masks = []

    step = int(patch_size * (1 - overlap_ratio))  # 根据重叠比例计算步长 (Calculate step size based on overlap ratio)
    extracted_patches = []  # 存储正样本块坐标 (Store patch coordinates)
    extracted_patches_bg = []  # 存储负样本块坐标 (Store patch coordinates for background/negative samples)

    for y in range(0, image.shape[0], step):
        for x in range(0, image.shape[1], step):
            img_patch = image[y:y + patch_size, x:x + patch_size]
            mask_patch = mask[y:y + patch_size, x:x + patch_size]

            if img_patch.shape[0] < patch_size or img_patch.shape[1] < patch_size:
                img_patch = np.pad(img_patch, ((0, patch_size - img_patch.shape[0]),
                                               (0, patch_size - img_patch.shape[1]),
                                               (0, 0)), mode='constant')
                mask_patch = np.pad(mask_patch, ((0, patch_size - mask_patch.shape[0]),
                                                 (0, patch_size - mask_patch.shape[1])),
                                    mode='constant')

            patch_coords = (y, x, y + patch_size, x + patch_size)
            # 如果块中至少有 threshold% 的区域 (If the patch contains at least threshold% of the region)
            if np.sum(mask_patch) > threshold * patch_size ** 2:
                patches.append(img_patch)
                patch_masks.append(mask_patch)
                extracted_patches.append(patch_coords)
            else:
                bg_patches.append(img_patch)
                bg_patch_masks.append(mask_patch)
                extracted_patches_bg.append(patch_coords)
    # 抽取负样本 (Sample negative samples)
    if negative_samples > 0:
        print("Collecting " + str(int(negative_samples * len(patches))) + " negative samples")
        bg_indices = np.random.choice(len(bg_patches), int(negative_samples * len(patches)), replace=False)
        bg_patches = np.array(bg_patches)[bg_indices]
        bg_patch_masks = np.array(bg_patch_masks)[bg_indices]
        extracted_patches_bg = np.array(extracted_patches_bg)[bg_indices].tolist()

    print(len(patch_masks), len(bg_patches))

    if negative_samples > 0:
        return patches, patch_masks, bg_patches, bg_patch_masks
    return patches, patch_masks

def save_patches_and_masks(patches, masks, output_dir, scanner_name, base_name, region_type):
    base_name_dir = os.path.join(output_dir, scanner_name, base_name)
    region_dir = os.path.join(base_name_dir, region_type)

    img_dir = os.path.join(region_dir, "images")
    mask_dir = os.path.join(region_dir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    for i, (patch, mask) in enumerate(zip(patches, masks)):
        img_path = os.path.join(img_dir, f"patch_{i}.tif")
        mask_path = os.path.join(mask_dir, f"patch_{i}.tif")
        Image.fromarray(patch).save(img_path)
        Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)

    print(f"Saved {len(patches)} patches to {region_dir}")

if __name__ == "__main__":
    Image.MAX_IMAGE_PIXELS = 10000000000
    parser = argparse.ArgumentParser(description='PyTorch code: Segmentation Training for Mahalanobis detector')

    parser.add_argument('--patch_size', type=int, help='Patch size')
    parser.add_argument('--overlap', type=float, help='of overlap set (10%)')
    parser.add_argument('--plot', action='store_true', help='Whether to plot the patches')
    parser.add_argument('--threshold', type=float,
                        help='How much of the region is required to be masked to be included')
    parser.add_argument('--negative-samples', type=float, help='Relative number of negative samples')
    parser.add_argument('--no-save', dest="save", action='store_false', help='Save patches')
    args = parser.parse_args()

    input_dir_train = "/Subset3_Train_image/"
    mask_dir_train = "/Subset3_Train_annotations_new/"
    output_dir_train = "/train_patches/"
    input_dir_test = "/Subset3_Test_image/"
    mask_dir_test = "/Subset3_Test_annotations_new/"
    output_dir_test = "/test_patches/"

    input_dir_dict = {"train": input_dir_train, "test": input_dir_test}
    mask_dir_dict = {"train": mask_dir_train, "test": mask_dir_test}
    output_dir_dict = {"train": output_dir_train, "test": output_dir_test}

    mask_type = SegmentationMask.G4.value
    mask_types = [mask_type]

    patch_size = args.patch_size if args.patch_size is not None else 2048
    overlap_ratio = args.overlap if args.overlap is not None else 0
    threshold = args.threshold if args.threshold is not None else 0.3
    negative_samples = args.negative_samples if args.negative_samples is not None else 1
    save = args.save if args.save is not None else True
    plot = args.plot if args.plot is not None else False

    print(negative_samples)
    for dataset_type in ["train"]:
        print("Dataset type: ", dataset_type)
        input_dir = input_dir_dict[dataset_type]
        mask_dir = mask_dir_dict[dataset_type]
        output_dir = output_dir_dict[dataset_type]

        print(f"Input directory: {input_dir}")
        print(f"Mask directory: {mask_dir}")
        print(f"Output directory: {output_dir}")
        print("Starting patch extraction")

        allowed_scanners = {Scanner.OLYMPUS.value, Scanner.ZEISS.value, Scanner.LEICA.value, Scanner.PHILIPS.value}
        #allowed_scanners = {Scanner.KFBIO.value}
        for scanner_name in os.listdir(input_dir):
            if scanner_name not in allowed_scanners:
                continue
            scanner_path = os.path.join(input_dir, scanner_name)
            mask_scanner_path = os.path.join(mask_dir, scanner_name)
            if not os.path.isdir(scanner_path) or not os.path.isdir(mask_scanner_path):
                continue
            for image_name in os.listdir(scanner_path):
                if not image_name.lower().endswith((".tif", ".tiff")):
                    continue

                print(f"Processing image: {image_name}")
                image_path = os.path.join(scanner_path, image_name)
                image = np.array(Image.open(image_path))
                base_name = os.path.splitext(image_name)[0]

                region_masks = {}
                mask_folder = os.path.join(mask_scanner_path, base_name)
                if not os.path.isdir(mask_folder):
                    print(f"Mask folder not found for {image_name}")
                    continue

                for mask_file in os.listdir(mask_folder):
                    for mask_type in mask_types:
                        if mask_file == mask_type + "_Mask.tif":
                            region_name = mask_file.split("_")[0]
                            mask_path = os.path.join(mask_folder, mask_file)
                            mask = np.array(Image.open(mask_path))
                            region_masks[region_name] = mask

                for region_type, mask in region_masks.items():
                    print(f"Processing region: {region_type}")
                    patch_data = dynamic_extract_contour_patches(image, mask, patch_size, overlap_ratio,
                                                                 threshold=threshold, negative_samples=negative_samples,
                                                                 plot=plot, not_plot_patches=True)
                    if save:
                        patches, patch_masks = patch_data[:2]
                        print("Saves " + str(len(patches)) + " positive patches")
                        save_patches_and_masks(patches, patch_masks, output_dir, scanner_name, base_name, region_type)
                        if negative_samples > 0:
                            patches_bg, patch_masks_bg = patch_data[2:]
                            print("Saves " + str(len(patches_bg)) + " negative patches")
                            save_patches_and_masks(patches_bg, patch_masks_bg, output_dir, scanner_name, base_name,
                                                   region_type)
