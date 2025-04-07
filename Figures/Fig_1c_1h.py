# connectogram_plot.py

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from pathlib import Path
# ==== CONFIGURATION ====
PATH_Data = os.path.join(Path(__file__).resolve().parent.parent, 'Data')
FIGSIZE = (5, 5)

# ==== LOAD DATA ====
labels = pd.read_csv(os.path.join(PATH_Data, 'data_atlas.csv'))
con_all = pd.read_csv(os.path.join(PATH_Data, 'data_con_figures.csv'))
ftract_df = pd.read_csv(os.path.join(PATH_Data, 'FTRACT_probability_table.csv'))

regions_all = np.unique(labels.drop_duplicates('region').region)
region_color_map = labels.drop_duplicates('region').set_index('region')['plot_color'].to_dict()
stim_colors = labels['plot_color'].tolist()
all_labels = labels['abbreviation'].tolist()
list_region = labels['region'].tolist()

# ==== PROCESS CONNECTO DATA ====
con_filtered = con_all[
    (con_all.Hemi != 'B') &
    np.isin(con_all.StimR, regions_all) &
    np.isin(con_all.ChanR, regions_all)
].reset_index(drop=True)

col_analysis = ['Sig', 'DI', 'LL_mean']
df_sig = con_filtered[(con_filtered.Sig > 0)].groupby(['Subj', 'StimA', 'ChanA', 'StimR', 'ChanR'], as_index=False)[col_analysis].mean()
df_sig = df_sig.groupby(['StimA', 'ChanA', 'StimR', 'ChanR'], as_index=False)[col_analysis].mean()

df_all = con_filtered.groupby(['Subj', 'StimA', 'ChanA', 'StimR', 'ChanR'], as_index=False)[col_analysis].mean()
df_all = df_all.groupby(['StimA', 'ChanA', 'StimR', 'ChanR'], as_index=False)[col_analysis].mean()
df_all['Sig0'] = df_all['Sig']

df_merged = df_sig.merge(df_all[['StimA', 'ChanA', 'Sig0']], on=['StimA', 'ChanA'], how='outer')
df_merged['Sig'].fillna(df_merged['Sig0'], inplace=True)

# ==== MATRIX PREPARATION ====
def create_matrix(df, row_col, value_col, all_labels, label_order):
    df_full = pd.DataFrame(index=all_labels, columns=all_labels)
    for _, row in df.iterrows():
        df_full.loc[row[row_col[0]], row[row_col[1]]] = row[value_col]
    df_full = df_full.astype(float)
    df_full = df_full.loc[label_order, label_order]
    return df_full

order_mapping = labels.set_index('abbreviation')['plot_order_float'].to_dict()
sorted_labels = sorted(all_labels, key=lambda x: order_mapping.get(x, float('inf')))

connecto_matrix = create_matrix(df_merged, ['StimA', 'ChanA'], 'Sig', all_labels, sorted_labels)

# ==== PLOTTING FUNCTION ====
def plot_heatmap(matrix, colors, vmax, cmap, title=None):
    sns.set(style="white")
    norm = mcolors.Normalize(vmin=0, vmax=vmax)

    g = sns.clustermap(
        matrix.fillna(np.nan), cmap=cmap,
        row_cluster=False, col_cluster=False,
        row_colors=colors, col_colors=colors,
        figsize=FIGSIZE, vmin=0, vmax=vmax
    )
    g.ax_heatmap.collections[0].set_facecolor('white')

    # Add border
    rect = patches.Rectangle((0, 0), 1, 1, transform=g.ax_heatmap.transAxes,
                             linewidth=0.5, edgecolor='black', facecolor='none')
    g.ax_heatmap.add_patch(rect)

    g.ax_heatmap.set_xticks([])
    g.ax_heatmap.set_yticks([])

    if title:
        g.ax_heatmap.set_title(title)

    plt.show()


# ==== PLOT CONNECTO FIGURE ====
colors_gray_to_blue = [
    (0.8, 0.8, 0.8),
    (0.62, 0.79, 0.88),
    (0.42, 0.68, 0.84),
    (0.26, 0.57, 0.78),
    (0.13, 0.44, 0.71),
    (0.03, 0.32, 0.61),
    (0.03, 0.19, 0.42)
]
custom_cmap = mcolors.LinearSegmentedColormap.from_list("GrayToBlue", colors_gray_to_blue)

plot_heatmap(connecto_matrix, stim_colors, vmax=0.75, cmap=custom_cmap, title="Connecto Response Probability")
print('Fig 1h')
# ==== PROCESS F-TRACT ====
file_ftract = '/Volumes/vellen/PhD/EL_experiment/Analysis/FTRACT/probability_table.csv'
df = pd.read_csv(file_ftract)

# Step 1: Rename columns in df_data
df.rename(columns={'StimR': 'StimD', 'ChanR': 'ChanD'}, inplace=True)

# Step 2: Merge to fill StimR
df = df.merge(labels[['destrieux', 'region']], left_on='StimD', right_on='destrieux', how='left')
df.rename(columns={'region': 'StimR'}, inplace=True)
df.drop('destrieux', axis=1, inplace=True)

# Step 3: Merge to fill ChanR
df = df.merge(labels[['destrieux', 'region']], left_on='ChanD', right_on='destrieux', how='left')
df.rename(columns={'region': 'ChanR'}, inplace=True)
df.drop('destrieux', axis=1, inplace=True)
df = df.drop_duplicates().reset_index(drop=True)

# Convert color column to a list for plotting
stim_colors = labels['plot_color'].tolist()
# Convert region column to a list showing community
list_region = labels['region'].tolist()

# Ensure all rows/columns from tab_destrieux are included
all_labels = labels['destrieux'].tolist()

# Step 1: Fill missing rows and columns in df
df_full = pd.DataFrame(index=all_labels, columns=all_labels)  # Full matrix
for _, row in df.iterrows():
    stim = row['StimD']
    resp = row['ChanD']
    value = row['probability']
    df_full.loc[stim, resp] = value

# ==== PLOT F-TRACT FIGURE ====
# Create a normalization instance
vmax = 0.4
norm = mcolors.Normalize(vmin=0, vmax=vmax)

# Replace NaN values in the dataframe for plotting
df_full_plot = df_full.astype(float).fillna(np.nan)

# Plot the heatmap
sns.set(style="white")
g = sns.clustermap(df_full_plot, cmap='binary',
                   row_cluster=False, col_cluster=False,
                   row_colors=stim_colors, col_colors=stim_colors,
                   figsize=(5, 5), vmin=0, vmax=vmax)

# Ensure NaN values are plotted as white
g.ax_heatmap.collections[0].set_facecolor('white')

# Add a black box around the heatmap
rect = patches.Rectangle(
    (0, 0), 1, 1,  # Position and size (0,0) to (1,1) in relative coordinates
    transform=g.ax_heatmap.transAxes,  # Use axis coordinates for positioning
    linewidth=0.5, edgecolor='black', facecolor='none'
)
g.ax_heatmap.add_patch(rect)

# Remove x-axis and y-axis ticks
g.ax_heatmap.set_xticks([])
g.ax_heatmap.set_yticks([])

# Step: Add one square per region on the diagonal
ax = g.ax_heatmap

# Initialize variables to track regions
current_region = list_region[0]
start_idx = 0
region_color = stim_colors[0]

for i, region in enumerate(list_region + [None]):  # Add None at the end to trigger the last rectangle
    if region != current_region:
        # Calculate the size of the block
        block_size = i - start_idx

        # Draw a square for the current region
        rect = patches.Rectangle(
            (start_idx, start_idx), block_size, block_size,  # (x, y), width, height
            linewidth=1, edgecolor=region_color, facecolor='none'
        )
        ax.add_patch(rect)

        # Update variables for the next region
        current_region = region
        if i < len(list_region):  # Avoid index error for the last None
            region_color = stim_colors[i]
        start_idx = i

# plt.savefig(os.path.join(PATH_fig, 'Response_incidence_FTRACT_map.svg'))
plt.show()
print('Fig 1c')
