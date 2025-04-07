# import necessary libraries
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

# ==== CONFIGURATION ====
PATH_Data = os.path.join(Path(__file__).resolve().parent.parent, 'Data')
# ==== LOAD DATA ====
# Load the labels and connection data
labels = pd.read_csv(os.path.join(PATH_Data, 'data_atlas.csv'))
con_all = pd.read_csv(os.path.join(PATH_Data, 'data_con_figures.csv'))

# Define colors for plotting
c_P = '#4A96C8'  # Color for Sig (Response Probability)
c_DI = '#73BC75'  # Color for DI (Directionality Index)

# Group labels for x-axis in the violin plot
group_label = ['local', 'direct', 'indirect']


def plot_P_DI_violin(con_all):
    """
    Plot a split violin plot showing the distribution of two metrics:
    - Sig: Response Probability
    - DI: Directionality Index (DI is transformed for plotting purposes)

    Parameters:
    - con_all: DataFrame containing the connection data with 'Group', 'Sig', and 'DI' columns
    """
    # Prepare data for 'Sig' (Response Probability)
    data_P = con_all[['Group', 'Sig']].rename(columns={'Sig': 'm'})
    data_P = data_P[data_P.m > 0].reset_index(drop=True)  # Filter out non-positive values
    data_P['Condition'] = 'Sig'  # Label the data as 'Sig'

    # Prepare data for 'DI' (Directionality Index)
    data_DI = con_all[['Group', 'DI']].rename(columns={'DI': 'm'})
    data_DI['m'] = abs(data_DI.m)  # Take the absolute value of DI
    data_DI['m'] = abs(data_DI.m - 1)  # Transform DI for visualization (m-1)
    data_DI['Condition'] = 'DI'  # Label the data as 'DI'

    # Concatenate the 'Sig' and 'DI' data for combined plotting
    data_concat = pd.concat([data_P, data_DI]).reset_index(drop=True)

    # Plot the split violin plot
    fig, ax = plt.subplots(figsize=(3.5 , 2.8))  # Size in cm (convert to inches for plt)

    # Create the violin plot with different colors for 'Sig' and 'DI'
    sns.violinplot(x='Group', y='m', hue='Condition', data=data_concat, split=True,
                   inner="quart", palette={'Sig': c_P, 'DI': c_DI}, ax=ax,
                   order=group_label, density_norm='width', cut=0, linecolor='black', linewidth=0.75)

    # Calculate and annotate the number of samples (N) for each group
    group_counts = data_P['Group'].value_counts().reindex(group_label)
    for i, (group, count) in enumerate(group_counts.items()):
        ax.text(i, 1.05, f'#{count}', ha='center', va='center', color='black', fontsize=8)

    ax.set_xlabel("")  # No label for x-axis

    # Create a secondary y-axis for the 'DI' metric
    ax2 = ax.twinx()

    # Set y-axis limits for both axes
    ax.set_ylim([-0.01, 1.01])  # Y-axis for 'Sig'
    ax2.set_ylim([1.01, -0.01])  # Y-axis for 'DI'

    # Set y-axis labels with color-coding
    ax.set_ylabel("Response Probability", color=c_P)
    ax2.set_ylabel("Directionality Index", color=c_DI)

    plt.tight_layout()

    plt.show()


plot_P_DI_violin(con_all)
print('Fig2b')
