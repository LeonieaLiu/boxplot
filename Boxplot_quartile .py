import pandas as pd
# from lib.data.aggc_dataset import Scanner
import seaborn as sns
import matplotlib.pyplot as plt
from enum import Enum

import os



class Scanner:
    AKOYA = "Akoya"
    OLYMPUS = "Olympus"
    ZEISS = "Zeiss"
    KFBIO = "KFBio"
    LEICA = "Leica"
    PHILIPS = "Philips"


def plot_mahal(scores: dict[str, list[float]], magnitude, bottleneck):
    # Create pd.DataFrame from the scores
    # df = pd.DataFrame(scores)
    # print(df)
    
    data = [
        {"Scanner": key, "Score": value}
        for key, values in scores.items()
        for value in values
    ]
    df = pd.DataFrame(data)
    
    # Clipping variant
    df['ClippedScore'] = df['Score'].clip(lower=df['Score'].quantile(0.03), upper=df['Score'].quantile(0.97))
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    # # Original Scores
    # plt.subplot(1, 2, 1)
    # sns.boxplot(x="Scanner", y="Score", data=df)
    # plt.title('Original Scores')
    
    # Clipped Scores
    #plt.subplot(1, 2, 2)
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)#gpt

    #df['Scanner'] = df['Scanner'].apply(lambda x: x.name )#x.name)# Filter out scanner names

    sns.boxplot(x="Scanner", y="ClippedScore", data=df)
    means = df.groupby('Scanner')['ClippedScore'].mean()
    variances = df.groupby('Scanner')['ClippedScore'].var()

    x_ticks = plt.gca().get_xticks()#gpt


    for scanner in df['Scanner'].unique():
        idx = df['Scanner'].unique().tolist().index(scanner)#gpt
        mean = means[scanner]
        variance = variances[scanner]
        plt.text(x_ticks[idx], df['ClippedScore'].max() + 0.1, f'Mean: {mean:.2f}\nVar: {variance:.2f}',#gpt
                 horizontalalignment='center', size='medium', color='black', weight='semibold')#gpt

        #plt.text(scanner, df['ClippedScore'].max(), f'Mean: {mean:.2f}\nVar: {variance:.2f}',
                 #horizontalalignment='center', size='medium', color='black', weight='semibold')
    plt.title(f'Mahalanobis Confidence Scores, Magnitude: {magnitude}, Bottleneck-Layers: {bottleneck}')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    scanners = [Scanner.AKOYA, Scanner.KFBIO, Scanner.LEICA, Scanner.ZEISS, Scanner.OLYMPUS, Scanner.PHILIPS]
    
    bottleneck_layers = [[1], [2], [-1], [-2], [-1, 1]]
    
    for bottleneck in bottleneck_layers:
        bottleneck_str = '_'.join(map(str, bottleneck))
        # Calculate Mahalanobis scores
        score_path = f"scores/{bottleneck_str}/"
        m_list = [0.0, 0.001, 0.01]#, 0.005, 0.002, 0.0014, 0.001, 0.0005]

        for magnitude in m_list:
            print(f"Processing magnitude {magnitude}")#gpt
            scores = {}
            for scanner in scanners:
                if os.path.exists(score_path + f"mahalanobis_scores_{magnitude}_{scanner}.csv"):
                    scores[scanner] = pd.read_csv(score_path + f"mahalanobis_scores_{magnitude}_{scanner}.csv").iloc[:, 1].tolist()
                else:
                    print(f"File not found: {score_path + f'mahalanobis_scores_{magnitude}_{scanner}.csv'}")
            #print(scores)
            print(f"Scores for magnitude {magnitude}: {scores}")


            plot_mahal(scores, magnitude=magnitude, bottleneck=bottleneck_str)


