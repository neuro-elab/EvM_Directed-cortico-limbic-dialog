# Import necessary libraries
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# ==== CONFIGURATION ====
# Define paths for data and resources
PATH_Data = os.path.join(Path(__file__).resolve().parent.parent, 'Data')
# Load data from CSV files
labels = pd.read_csv(os.path.join(PATH_Data, 'data_atlas.csv'))
con_all = pd.read_csv(os.path.join(PATH_Data, 'data_con_figures.csv'))

# ==== PREPROCESSING ====
# Extract the regions and their colors
regions_all = labels.drop_duplicates('region').sort_values('plot_order')['region'].values
regions_MT = ['Mesiotemporal', 'Amygdala', 'Hippocampus']
regions_CX = ['Orbitofrontal', 'Operculo-insular', 'Dorsofrontal', 'Central',
              'Cingular', 'Parietal', 'Insulo-temporal', 'Inferotemporal', 'Occipital']

# Get colors for the regions
regions_MT_col = [labels.loc[labels.region == lb, 'plot_color'].values[0] for lb in regions_MT]
regions_CX_col = [labels.loc[labels.region == lb, 'plot_color'].values[0] for lb in regions_CX]

# ==== DATA FILTERING ====
# Filter out unwanted connections (local, inter-hemispheric, or unknown regions)
con_filtere = con_all[
            (con_all.Hemi != 'B') & (con_all.Group != 'local')].reset_index(drop=True)

# Prepare reverse DataFrame to merge for comparison
con_filtere_reverse = con_filtere[['Subj','Stim', 'Chan', 'Num_trial']].reset_index(drop=True)
con_filtere_reverse = con_filtere_reverse.rename(columns={'Stim':'Chan', 'Chan':'Stim', 'Num_trial':'Num_trial_in'})
con_filtere = con_filtere.merge(con_filtere_reverse, on=['Subj','Stim', 'Chan'], how='left')

# Merge signal data for comparison
con_filtere_reverse = con_filtere[['Subj','Stim', 'Chan', 'Sig']].reset_index(drop=True)
con_filtere_reverse = con_filtere_reverse.rename(columns={'Stim':'Chan', 'Chan':'Stim', 'Sig':'Sig_in'})
con_filtere = con_filtere.merge(con_filtere_reverse, on=['Subj','Stim', 'Chan'], how='left')

# Calculate signal difference
con_filtere['Sig_diff'] = con_filtere['Sig'] - con_filtere['Sig_in']

# Handle NaN values when both signals are zero
con_filtere.loc[(con_filtere.Sig == 0) & (con_filtere.Sig_in == 0), 'Sig_diff'] = np.nan

# Apply trial number threshold to remove unreliable data
num_threshold = 10
con_filtere.loc[(con_filtere.Num_trial_in < num_threshold) | (con_filtere.Num_trial < num_threshold), ['Sig_diff', 'DI']] = np.nan

# ==== DATA SPLITTING ====
# Separate data based on region categories (Mesiotemporal and Cortical regions)
data_DI_Subj_MT = con_filtere[~np.isnan(con_filtere.Sig_diff) & ~np.isin(con_filtere.ChanR, regions_MT)].groupby(['Subj', 'StimR'], as_index=False)[['d_inf', 'Sig', 'DI']].mean()
data_DI_Node_MT = con_filtere[~np.isnan(con_filtere.Sig_diff) & ~np.isin(con_filtere.ChanR, regions_MT)].groupby(['Subj', 'StimR', 'ChanR'], as_index=False)[['d_inf', 'Sig', 'DI']].mean()

data_DI_Subj_CXT = con_filtere[~np.isnan(con_filtere.Sig_diff) & ~np.isin(con_filtere.StimR, regions_MT) & ~np.isin(con_filtere.ChanR, regions_MT)].groupby(['Subj', 'StimR'], as_index=False)[['d_inf', 'Sig', 'DI']].mean()
data_DI_Node_CXT = con_filtere[~np.isnan(con_filtere.Sig_diff) & ~np.isin(con_filtere.StimR, regions_MT) & ~np.isin(con_filtere.ChanR, regions_MT)].groupby(['Subj', 'StimR', 'ChanR'], as_index=False)[['d_inf', 'Sig', 'DI']].mean()

# ==== PLOTTING ====
# Define figure size (in cm)
cm = 1/2.54  # centimeters to inches
fig, axes = plt.subplots(1, 2, figsize=(13*cm, 8*cm), gridspec_kw={'width_ratios': [1, 3]}, sharey=True)

# Plot for Mesiotemporal Regions
ax = axes[0]
sns.violinplot(
    x='StimR', y='DI', data=data_DI_Node_MT, order=regions_MT, hue='StimR',
    hue_order=regions_MT, palette=regions_MT_col, ax=ax, legend=False,
    cut=0.1, inner="quart", density_norm='count'
)
sns.stripplot(
    x='StimR', y='DI', data=data_DI_Subj_MT, order=regions_MT,
    jitter=True, ax=ax, size=2.5, color='k', edgecolor='black', linewidth=0.5
)
ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
ax.set_xticklabels(regions_MT, rotation=90)
ax.set_title('Mesiotemporal to Neocortex', fontsize=8)
ax.set_ylim([-1.2, 1.2])
ax.legend_.remove() if ax.legend_ else None  # Remove legend if it exists
ax.set_xlabel('')
# Plot for Cortical Regions
ax = axes[1]
sns.violinplot(
    x='StimR', y='DI', data=data_DI_Node_CXT, order=regions_CX, hue='StimR',
    hue_order=regions_CX, palette=regions_CX_col, ax=ax, legend=False,
    cut=0.1, inner="quart", density_norm='count'
)
sns.stripplot(
    x='StimR', y='DI', data=data_DI_Subj_CXT, order=regions_CX,
    jitter=True, ax=ax, size=2.5, color='k', edgecolor='black', linewidth=0.5
)
ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
ax.set_xticklabels(regions_CX, rotation=90)
ax.set_title('Neocortex to Neocortex', fontsize=8)
ax.set_ylim([-1.05, 1.2])
ax.legend_.remove() if ax.legend_ else None  # Remove legend if it exists
ax.set_ylabel('')
ax.set_xlabel('')

# Display the plot
plt.tight_layout()
plt.show()
print('Fig3c and 3d')


# ==== PLOTTING: BAR PLOTS ====
# 1. Bar plot for neocortical-neocortical regions
df = con_filtere[~np.isnan(con_filtere.DI) & ~np.isin(con_filtere.ChanR, regions_MT)].reset_index(drop=True)
bins = [-1, -0.6, -0.2, 0.2, 0.6, 1]
colors = [plt.get_cmap('PRGn')(i) for i in np.linspace(0, 1, len(bins) - 1)]

fig, axes = plt.subplots(len(regions_all), 1, figsize=(2.3 * cm, 12 * cm), sharex=True)
plt.suptitle('Fig 3b i)')

for i, region in enumerate(regions_all):
    ax = axes[i]
    val = df.loc[df.StimR == region, 'DI'].values
    hist, _ = np.histogram(val, bins=bins)
    hist = hist / hist.sum()

    left = 0
    for j in range(len(hist)):
        ax.barh(region, hist[j], left=left, height=0.8, color=colors[j])
        left += hist[j]

    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.text(0.01, 0.02, f'#{len(val)}', color='white', fontsize=6)

ax.set_xticks([])

for ax in axes:
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

plt.show()
print('Fig 3b i)')

# 2. Bar plot for Mesiotemporal-neocortical regions
df = con_filtere[~np.isnan(con_filtere.DI) & np.isin(con_filtere.ChanR, regions_MT)].reset_index(drop=True)

fig, axes = plt.subplots(len(regions_all), 1, figsize=(2.3 * cm, 12 * cm), sharex=True)
plt.suptitle('Fig 3b ii)')

for i, region in enumerate(regions_all):
    ax = axes[i]
    val = df.loc[df.StimR == region, 'DI'].values
    hist, _ = np.histogram(val, bins=bins)
    hist = hist / hist.sum()

    left = 0
    for j in range(len(hist)):
        ax.barh(region, hist[j], left=left, height=0.8, color=colors[j])
        left += hist[j]

    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.text(0.01, 0.02, f'#{len(val)}', color='white', fontsize=6)

ax.set_xticks([])

for ax in axes:
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

plt.show()
print('Fig 3b ii)')