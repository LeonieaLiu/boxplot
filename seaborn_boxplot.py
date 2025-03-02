import pandas as pd
import matplotlib .pyplot as plt
import seaborn as sns
import numpy as np
import csv
import json
import yaml

def create_boxplot():
    with open('/Users/leoniealiu/Desktop/pvc/openset_CL/params.yaml','r') as file:
        scanner_type = yaml.FullLoader(file)
    akoya_group = {"Scanner": [], "Image_group": [], "Region":[], "Distance": []}
    KFBio_group = {"Scanner": [], "Image_group": [], "Region":[], "Distance": []}

    with open('/Users/leoniealiu/Desktop/pvc/openset_CL/patch_mahalanobis_distances_1.json', 'r') as file:
        data = json.load(file)
    # open up the csv file and read
    for i in range(len(data)):
        Scanner = data[i]['scanner']
        Image_group = data[i]['image_group']
        M_distance = data[i]['distance']
        if Scanner == 'Akoya':
            akoya_group['Scanner'].append(Scanner)
            akoya_group['Image_group'].append(Image_group)
            akoya_group['Distance'].append(M_distance)
        elif Scanner == 'KFBio':
            KFBio_group['Scanner'].append(Scanner)
            KFBio_group['Image_group'].append(Image_group)
            KFBio_group['Distance'].append(M_distance)

    print(f"akoya_group: {akoya_group}")
    print(f"KFBio_group: {KFBio_group}")
    # Calculate the mean
    distance_akoya = akoya_group['Distance']
    distance_KFBio = KFBio_group['Distance']
    mean_akoya = np.mean(distance_akoya) # get the mean
    std_dev_akoya = np.std(distance_akoya) # get the standard deviation
    mean_KFBio = np.mean(distance_KFBio)
    std_dev_KFBio = np.std(distance_KFBio)

    # determine the outlier
    # condition setting
    if scanner_type == 'Akoya':
        outlier_criteria = 3 * std_dev_akoya
        for dist in distance_akoya:
            error = np.abs(dist - mean_akoya)
            if error > outlier_criteria:  # outlier
                print(f"Outlier: {dist}")
            else:
                print(f"Non-outlier: {dist}")
    elif scanner_type == 'KFBio':
        outlier_criteria = 3 * std_dev_KFBio
        for dist in distance_KFBio:
            error = np.abs(dist - mean_KFBio)
            if error > outlier_criteria: # outlier
                print(f"Outlier: {dist}")
            else:
                print(f"Non-outlier: {dist}")

    # define the boxplot
    scanner_num = ['Akoya'] * len(distance_akoya) + ['KFBio'] * len(distance_KFBio)
    data = {
        'Scanner': scanner_num,
        'Mahalanobis_Distance': distance_akoya + distance_KFBio
    }

    df = pd.DataFrame(data)
    # Plot the boxplot using Seaborn
    plt.figure(figsize=(15, 10))  # Set the figure size
    sns.boxplot(x='Scanner', y='Mahalanobis_Distance', data=df, showmeans=True,
                meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "black", "markersize": "10"})

    # Customize plot
    plt.title('Boxplot of Mahalanobis Distances by Scanner')
    plt.xlabel('Scanner Type')
    plt.ylabel('Mahalanobis Distance')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the plot
    plt.show()

    return Scanner
if __name__ == '__main__':
    Scanner = create_boxplot()

