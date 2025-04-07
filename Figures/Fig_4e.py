import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================
# CONFIGURATION
# ==========================

PATH_Data = os.path.join(Path(__file__).resolve().parent.parent, 'Data')
# Region of interest: choose 'Hippocampus' or 'Amygdala'
REGION_SEL = 'Hippocampus'

# Cortex regions
regions_CX = [
    'Orbitofrontal', 'Operculo-insular', 'Dorsofrontal', 'Central',
    'Cingular', 'Parietal', 'Insulo-temporal', 'Inferotemporal', 'Occipital'
]

# Sleep state display settings
color_sleep = ['#2E2E2E', '#B89FCC', '#A49D82']
label_sleep = ['Wake', 'NREM', 'REM']


# ==========================
# DATA LOADING & CLEANING
# ==========================

labels = pd.read_csv(os.path.join(PATH_Data, 'data_atlas.csv'))
con_all = pd.read_csv(os.path.join(PATH_Data, 'data_con_figures.csv'))
df_ExI = pd.read_csv(os.path.join(PATH_Data, 'ExI_connections.csv'))

# Filter out unwanted subjects and local channels
df_ExI = df_ExI[
    (~np.isin(df_ExI.Subj, ['EL012', 'EL013'])) &
    (df_ExI.Group != 'local') &
    (df_ExI.Hemi != 'B')
].reset_index(drop=True)

df_ExI_results = pd.read_csv(os.path.join(PATH_Data, 'ExI_results.csv'))

# Keep only valid condition IDs
valid_ids = df_ExI.cond_id.unique()
df_ExI_results = df_ExI_results[df_ExI_results.cond_id.isin(valid_ids)].reset_index(drop=True)


# ==========================
# PLOTTING FUNCTION
# ==========================
def plot_excitability_curve(data, results, region_selected, direction_label):
    """
    Plot response curves and AUC violin plots for given subset of connections.
    """
    n_subj = data[data.SleepState == 'Wake']['Subj'].nunique()
    auc_data = results[(results.AUC_wake > 0)].reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(3.2, 2.6), gridspec_kw={'width_ratios': [3, 1]})
    plt.suptitle(f'{region_selected}: {direction_label} connections', fontsize=6)
    ax1, ax2 = axes

    # ---- Left plot: Response magnitude curves ----
    sns.lineplot(
        x='Int_norm', y='LL_norm', hue='SleepState',
        estimator=np.median, ci=90,
        data=data, palette=color_sleep, ax=ax1, lw=1.5
    )

    # Axis and legend settings
    ax1.set_ylim([0, 1.3])
    ax1.set_xticks([0, 0.5, 1])
    ax1.set_yticks([0, 0.5, 1])
    ax1.set_xlabel('Stimulation intensity [%]', fontsize=6)
    ax1.set_ylabel('Response magnitude [%]', fontsize=6)
    ax1.tick_params(axis='both', labelsize=6)
    ax1.set_aspect(1)

    # Legend counts
    legend_counts = [
        data[data.SleepState == state]['cond_id'].nunique()
        for state in ['Wake', 'NREM', 'REM']
    ]
    handles, _ = ax1.get_legend_handles_labels()
    ax1.legend(
        handles[:3],
        [f'{state} #{n}' for state, n in zip(label_sleep, legend_counts)],
        fontsize=6
    )
    ax1.text(0.05, 1, f'N={n_subj}', fontsize=6)

    # ---- Right plot: Normalized AUC violin plot ----
    sns.violinplot(
        x='SleepState', y='AUC_norm',
        data=auc_data, order=['NREM', 'REM'],
        palette=color_sleep[1:], ax=ax2, linewidth=0.5
    )

    ax2.set_ylim([0, 3])
    ax2.set_yticks([0.5, 1, 2])
    ax2.set_yticklabels(['0.5', '1', '2'])
    ax2.set_ylabel('')
    ax2.set_xlabel('')
    ax2.tick_params(axis='both', labelsize=6)
    ax2.axhline(1, ls='--', lw=0.5, color='k')

    plt.tight_layout()
    plt.show()


# ==========================
# AFFERENT CONNECTIONS
# ==========================
aff_data = df_ExI[
    (df_ExI['ChanR'] == REGION_SEL) &
    (df_ExI['StimR'].isin(regions_CX))
].reset_index(drop=True)

aff_results = df_ExI_results[
    (df_ExI_results['ChanR'] == REGION_SEL) &
    (df_ExI_results['StimR'].isin(regions_CX)) &
    (df_ExI_results['SleepState'] != 'Wake')
].reset_index(drop=True)

plot_excitability_curve(aff_data, aff_results, region_selected = REGION_SEL, direction_label='Afferent')


# ==========================
# EFFERENT CONNECTIONS
# ==========================
eff_data = df_ExI[
    (df_ExI['StimR'] == REGION_SEL) &
    (df_ExI['ChanR'].isin(regions_CX))
].reset_index(drop=True)

eff_results = df_ExI_results[
    (df_ExI_results['StimR'] == REGION_SEL) &
    (df_ExI_results['ChanR'].isin(regions_CX)) &
    (df_ExI_results['SleepState'] != 'Wake')
].reset_index(drop=True)

plot_excitability_curve(eff_data, eff_results, region_selected = REGION_SEL, direction_label='Efferent')

print('Fig 4e')
