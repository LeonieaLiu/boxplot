import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
import torch

@torch.no_grad()
def extract_specific_features(model, data_loader, device, selected_feature):
    """
    Extract features from a specified layer of the model and save the corresponding labels.
    """
    model.eval()
    features = []
    labels = []

    for images, _, batch_labels in tqdm(data_loader, desc="Extracting Features"):
        images = images.to(device)
        logits, first_up, second_up, second_last_down, last_combined, bottleneck = model(images)

        feature_map = {
            "first_up": first_up,
            "second_up": second_up,
            "second_last_down": second_last_down,
            "last_combined": last_combined,
            "bottleneck": bottleneck,
        }

        if selected_feature not in feature_map:
            raise ValueError(f"Invalid feature name: {selected_feature}")

        extracted = feature_map[selected_feature]

        for i in range(extracted.size(0)):
            features.append(extracted[i].detach().cpu().view(-1).numpy())
            labels.append(batch_labels[i].item())

    return np.vstack(features), labels

@torch.no_grad()
def initialize_mahalanobis(train_features, n_components=50):
    """
    Initialize PCA and compute the mean and inverse covariance matrix of the training data.
    """
    pca = PCA(n_components=n_components)
    train_features_pca = pca.fit_transform(train_features)

    train_mean = np.mean(train_features_pca, axis=0)
    train_cov = np.cov(train_features_pca, rowvar=False)
    train_cov_inv = np.linalg.inv(train_cov + np.eye(train_cov.shape[0]) * 1e-5)

    return pca, train_mean, train_cov_inv

@torch.no_grad()
def calculate_mahalanobis_distances(test_features, pca, train_mean, train_cov_inv, batch_size=1000):
    """
    Compute Mahalanobis distances in batches using PCA, mean, and inverse covariance matrix.
    """
    test_features_pca = pca.transform(test_features)

    distances = []
    for i in range(0, test_features_pca.shape[0], batch_size):
        batch = test_features_pca[i:i + batch_size]
        for feature in batch:
            diff = feature - train_mean
            distance = np.sqrt(np.dot(np.dot(diff.T, train_cov_inv), diff))
            distances.append(distance)

    return np.array(distances)


@torch.no_grad()
def calculate_threshold(
    model, data_loader, device, pca_akoya, train_mean_akoya, train_cov_inv_akoya,
    selected_feature=None, threshold_multiplier=3, accumulate=False
):
    """
    Compute the Mahalanobis distance threshold for anomaly detection based on Akoya training statistics.
    """
    print("Calculating Mahalanobis threshold...")
    all_distances = []

    for images, _, _ in tqdm(data_loader, desc="Calculating Mahalanobis distances"):
        images = images.to(device)
        logits, first_up, second_up, second_last_down, last_combined, bottleneck = model(images)
        feature_map = {
            "first_up": first_up,
            "second_up": second_up,
            "second_last_down": second_last_down,
            "last_combined": last_combined,
            "bottleneck": bottleneck,
        }

        if selected_feature and selected_feature not in feature_map:
            raise ValueError(f"Invalid feature name: {selected_feature}")

        batch_features = feature_map[selected_feature].detach().cpu().numpy() if selected_feature else logits.detach().cpu().numpy()
        batch_features = batch_features.reshape(batch_features.shape[0], -1)

        # Always compute Mahalanobis distance based on Akoya training statistics
        batch_distances = calculate_mahalanobis_distances(
            batch_features, pca_akoya, train_mean_akoya, train_cov_inv_akoya
        )
        all_distances.extend(batch_distances)

    print(f"Mahalanobis Distance Stats: min={np.min(all_distances):.4f}, max={np.max(all_distances):.4f}, mean={np.mean(all_distances):.4f}, std={np.std(all_distances):.4f}")

    if accumulate:
        return all_distances

    # Compute threshold
    sorted_distances = np.sort(all_distances)
    lower_index = int(0.05 * len(sorted_distances))
    upper_index = int(0.95 * len(sorted_distances))
    filtered_distances = sorted_distances[lower_index:upper_index]

    mean_distance = np.mean(filtered_distances)
    std_distance = np.std(filtered_distances)
    threshold = mean_distance + threshold_multiplier * std_distance

    print(f"Calculated threshold: {threshold:.4f}")
    return threshold


@torch.no_grad()
def monitor_patch_mahalanobis_change(
    model, batch_images, device, pca_akoya, train_mean_akoya, train_cov_inv_akoya,
    selected_feature, current_threshold
):
    """
    Monitor whether the current batch contains anomalous patches, always using Akoya statistics for Mahalanobis distance.
    """
    model.eval()
    batch_images = batch_images.to(device)

    # Extract features from the current batch
    logits, first_up, second_up, second_last_down, last_combined, bottleneck = model(batch_images)
    feature_map = {
        "first_up": first_up,
        "second_up": second_up,
        "second_last_down": second_last_down,
        "last_combined": last_combined,
        "bottleneck": bottleneck,
    }

    if selected_feature not in feature_map:
        raise ValueError(f"Invalid feature name: {selected_feature}")

    # Get required features
    batch_features = feature_map[selected_feature].detach().cpu().numpy()
    batch_features = batch_features.reshape(batch_features.shape[0], -1)  # Flatten for PCA

    # Always compute Mahalanobis distance based on Akoya training statistics
    distances = calculate_mahalanobis_distances(
        batch_features, pca_akoya, train_mean_akoya, train_cov_inv_akoya
    )

    # Detect if any patch is anomalous
    is_anomalous = distances > current_threshold

    return is_anomalous.any(), distances


@torch.no_grad()
def adjust_learning_rate(optimizer, base_lr, scale_factor=0.5):
    """
    Dynamically adjust the learning rate.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = base_lr * scale_factor

@torch.no_grad()
def restore_learning_rate(optimizer, base_lr):
    """
    Restore the initial learning rate.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = base_lr

@torch.no_grad()
def save_weights(model):
    """
    Save the current model weights.
    """
    return [param.data.clone() for param in model.parameters()]

@torch.no_grad()
def load_weights(model, saved_weights):
    """
    Restore the model weights.
    """
    for param, saved in zip(model.parameters(), saved_weights):
        param.data.copy_(saved)
