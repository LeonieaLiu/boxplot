import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json


def create_boxplot(feature_name):
    with open(f'/Users/leoniealiu/Desktop/pvc/openset_CL/zxy/confidences_{feature_name}.json',
              'r') as file:
        data = json.load(file)

   # Initialize groups for all scanners list
    scanner_groups = []

    # Filter data and separate into groups
    for scanner_type, distances in data.items():
        for distance in distances:
            scanner_groups.append({"scanner": scanner_type, "distance": distance})

    # Combine data for boxplot
    combined_scanner = []
    combined_distances = []
    for scanner_group in scanner_groups:
        combined_scanner.append(scanner_group["scanner"])
        combined_distances.append(scanner_group["distance"])

    data_frame = pd.DataFrame({
        'Scanner': combined_scanner,
        'Mahalanobis_Distance': combined_distances
    })

    # Calculate mean and standard deviation
    mean = data_frame['Mahalanobis_Distance'].mean()
    std_dev = data_frame['Mahalanobis_Distance'].std()

    # Define custom whisker range (3 times standard deviation)
    lower_limit = mean - 3 * std_dev
    upper_limit = mean + 3 * std_dev

    # Plot the boxplot without whiskers and customize whiskers manually
    plt.figure(figsize=(15, 10))
    sns.boxplot(
        x='Scanner',
        y='Mahalanobis_Distance',
        data=data_frame,
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "red",
            "markeredgecolor": "black",
            "markersize": "10"
        },
        showfliers=False  # Disable default whiskers and outliers
    )

    # Add custom whiskers and outliers
    for scanner in data_frame['Scanner'].unique():
        scanner_data = data_frame[data_frame['Scanner'] == scanner]['Mahalanobis_Distance']
        outliers = scanner_data[
            (scanner_data < lower_limit) | (scanner_data > upper_limit)
            ]

        # Scatter plot for outliers
        plt.scatter(
            x=[scanner] * len(outliers),
            y=outliers,
            color='blue',
            edgecolor='black',
            s=80,
            label='Outliers (> 3 Std Dev)' if scanner == data_frame['Scanner'].unique()[0] else ""
        )

    # Customize plot
    plt.title(f'Boxplot of Mahalanobis Distances by Scanner ({feature_name})')
    plt.xlabel('Scanner Type')
    plt.ylabel('Mahalanobis Distance')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')

    # Save the plot with the feature name in the filename
    save_path = f'/Users/leoniealiu/Desktop/pvc/openset_CL/zxy/confidences_{feature_name}.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Boxplot for {feature_name} saved to {save_path}")

    plt.show()


if __name__ == '__main__':
    # Add feature selection logic
    selected_feature = input(
        "Enter the feature layer to use (e.g., first_up0.01, second_up0.01, second_up0.001): ")
    valid_features = ["first_up0.01", "second_up0.01", "second_up0.001"]
    if selected_feature not in valid_features:
        raise ValueError(f"Invalid feature name. Please choose from {valid_features}")

    # Create boxplot for the selected feature
    create_boxplot(selected_feature)