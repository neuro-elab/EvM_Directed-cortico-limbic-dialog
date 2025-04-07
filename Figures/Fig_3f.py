import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# ==== CONFIGURATION ====

PATH_Data = os.path.join(Path(__file__).resolve().parent.parent, 'Data')
# ==== LOAD DATA ====
labels = pd.read_csv(os.path.join(PATH_Data, 'data_atlas.csv'))
con_all = pd.read_csv(os.path.join(PATH_Data, 'data_con_figures.csv'))

# Load and preprocess ExI data
df_ExI = pd.read_csv(os.path.join(PATH_Data, 'ExI_connections.csv'))
df_ExI = df_ExI[(df_ExI.Group != 'local') & (df_ExI.Hemi != 'B') & (df_ExI.SleepState == 'Wake')].reset_index(drop=True)

# Get color map and region information
region_color_map = labels.drop_duplicates('region').set_index('region')['plot_color'].to_dict()
regions_all = labels.drop_duplicates('region').sort_values('plot_order')['region'].values
regions_MT = ['Mesiotemporal', 'Amygdala', 'Hippocampus']
regions_CX = ['Orbitofrontal', 'Operculo-insular', 'Dorsofrontal', 'Central', 'Cingular', 'Parietal', 'Insulo-temporal', 'Inferotemporal', 'Occipital']

# Get color for regions of interest
regions_MT_col = [labels.loc[labels.region == lb, 'plot_color'].values[0] for lb in regions_MT]
regions_CX_col = [labels.loc[labels.region == lb, 'plot_color'].values[0] for lb in regions_CX]

# ==== REGION SELECTION ====
# Choose a region for analysis (e.g., 'Hippocampus')
region_sel = 'Hippocampus'
color_reg = labels.loc[labels.region == region_sel, 'plot_color'].values[0]

# Filter data based on selected region and connection types
data_in = df_ExI[np.isin(df_ExI['StimR'], regions_CX) & (df_ExI['ChanR'] == region_sel)].reset_index(drop=True)
data_out = df_ExI[np.isin(df_ExI['ChanR'], regions_CX) & (df_ExI['StimR'] == region_sel)].reset_index(drop=True)

# Add a condition column to differentiate in and out data
data_in['Condition'] = 'in'
data_out['Condition'] = 'out'
data_plot = pd.concat([data_in, data_out]).reset_index(drop=True)

# Get the number of unique subjects in the 'Wake' sleep state
n_subj = len(np.unique(data_plot.loc[data_plot.SleepState == 'Wake', 'Subj']))

# ==== DATA FOR PLOTTING ====
# Prepare data for the boxplot
data_in_box = con_all[~np.isnan(con_all.ExI) & np.isin(con_all['StimR'], regions_CX) & (con_all['ChanR'] == region_sel)].reset_index(drop=True)
data_out_box = con_all[~np.isnan(con_all.ExI) & np.isin(con_all['ChanR'], regions_CX) & (con_all['StimR'] == region_sel)].reset_index(drop=True)

# Add condition column for boxplot data
data_out_box['Condition'] = 'out'
data_in_box['Condition'] = 'in'
data_plot_box = pd.concat([data_in_box, data_out_box]).reset_index(drop=True)

# ==== PLOTTING ====
# Create the figure with custom width ratios for side-by-side plots
fig, ax = plt.subplots(1, 2, figsize=(5.8 / 2.54, 4.1 / 2.54), gridspec_kw={'width_ratios': [4, 1]})
plt.suptitle(f'Excitability across afferent and efferent connection ({region_sel})', fontsize=6)

# ==== Line Plot (Condition vs. Stimulation Intensity) ====
ax1 = ax[0]
sns.lineplot(
    x='Int_norm', y='LL_norm', hue='Condition', hue_order=['in', 'out'],
    estimator=lambda x: np.median(x), data=data_plot,
    palette=['gray', color_reg], ax=ax1, lw=0.5, legend=False
)
ax1.set_xlim([-0.05, 1.05])
ax1.set_ylim([-0.05, 1.05])
ax1.set_yticks([0, 0.5, 1])
ax1.set_xticks([0, 0.5, 1])
ax1.set_xticklabels([0, 50, 100], fontsize=6)
ax1.set_yticklabels([0, 50, 100], fontsize=6)
ax1.set_xlabel('Stimulation intensity [%]', fontsize=6)
ax1.set_ylabel('Response magnitude [%]', fontsize=6)
ax1.set_aspect(1)  # Equal scaling for x and y axes

# ==== Violin Plot (Excitability Index) ====
ax2 = ax[1]
sns.violinplot(y='ExI', hue='Condition', hue_order=['in', 'out'], palette=['gray', color_reg],
    data=data_plot_box, ax=ax2, legend=False, linewidth=0.5)
ax2.set_yticks([0, 0.5, 1])
ax2.set_ylim([-0.05, 1.05])
ax2.set_yticklabels([0, 0.5, 1], fontsize=6)
ax2.set_ylabel('', fontsize=6)  # Hide y-label for the second plot

plt.tight_layout()
plt.show()
print('Fig3f')
