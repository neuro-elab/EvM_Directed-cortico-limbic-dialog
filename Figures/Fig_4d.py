
import os
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from connectogram_helpers import connectogram_region_sleep_mod, reorder_letters
from pathlib import Path

# ==== CONFIGURATION ====

PATH_Data = os.path.join(Path(__file__).resolve().parent.parent, 'Data')

# ==== LOAD DATA ====
labels = pd.read_csv(os.path.join(PATH_Data, 'data_atlas.csv'))
df_con = pd.read_csv(os.path.join(PATH_Data, 'data_con_figures.csv'))
df_stats = pd.read_csv(os.path.join(PATH_Data, 'sleep_mod_stats.csv'))
# Filter out unwanted subjects and local channels
df_con = df_con[
    (~np.isin(df_con.Subj, ['EL012', 'EL013'])) &
    (df_con.Group != 'local') &
    (df_con.Hemi != 'B')
].reset_index(drop=True)

# === GLOBALS ====
# Get color map from table
region_color_map = labels.drop_duplicates('region').set_index('region')['plot_color'].to_dict()
# Filter and compute statistic
regions_all = labels.drop_duplicates('region').sort_values('plot_order')['region'].values
col_analysis  = ['Norm_NREM_Mag', 'Norm_REM_Mag', 'DI', 'Sig', 'peak_latency']

df_mean_R = df_con[df_con.Sig>0].groupby(['StimR','ChanR'], as_index=False)[col_analysis].mean().reset_index(drop=True)
df_mean_R['Subj'] ='EL000' #dummy subject used for connectogram format
df_mean_R['DI'] = df_mean_R['DI'].astype(float)
df_mean_R['Sig'] = df_mean_R['Sig'].astype(float)

# Calculate SigR
df_mean_R['SigR'] = np.where(df_mean_R['DI'] > 0,
                             df_mean_R['Sig'] - df_mean_R['DI'] * df_mean_R['Sig'],
                             df_mean_R['DI'] * df_mean_R['Sig'] - df_mean_R['Sig'])
for r in regions_all:
    a_label = labels.loc[labels.region==r, 'destrieux'].values[-1]
    df_mean_R.loc[df_mean_R.StimR ==r , 'StimA'] = a_label
    df_mean_R.loc[df_mean_R.ChanR ==r , 'ChanA'] = a_label
df_mean_R['Stim'] = df_mean_R.groupby(['StimR','ChanR']).ngroup()
df_mean_R['Chan'] = df_mean_R.groupby(['StimR','ChanR']).ngroup()+len(df_mean_R.groupby(['StimR','ChanR']).ngroup())+1

# calculate connectogram plot for each sleep state
value = 'Median_dif' # 'Median_dif', 'Median_norm'
for state in ['NREM', 'REM']:
    columns_to_keep = [
        'StimR', 'ChanR', 'Sig', 'peak_latency', f'Norm_{state}_Mag',
        'Subj', 'StimA', 'ChanA', 'Stim', 'Chan'
    ]

    stats_filtered = df_stats[df_stats.SleepState.str.lower() == state.lower()][
        ['StimR', 'ChanR', value, 'rejected']
    ]

    df_connectogram = (
        df_mean_R[columns_to_keep]
        .merge(stats_filtered, on=['StimR', 'ChanR'], how='left')
        .reset_index(drop=True)
    )

    df_connectogram.loc[df_connectogram.rejected == False, value] = 0
    df_connectogram['colored'] = 1  #  df_connectogram['rejected']
    df_connectogram['lw'] = df_connectogram['Sig']
    df_connectogram['value'] = df_connectogram[value]
    df_connectogram['dashed'] = df_connectogram['peak_latency'] > 0.065

    result_reverse = df_connectogram[['StimR', 'ChanR', 'colored', 'dashed', 'lw', 'value']].reset_index(drop=True)
    result_reverse = result_reverse.rename(columns={'StimR': 'ChanR', 'ChanR': 'StimR'})
    result_reverse['colored_out'] = result_reverse['colored']
    result_reverse['lw_out'] = result_reverse['lw']
    result_reverse['value_out'] = result_reverse['value']
    result_reverse['dashed_out'] = result_reverse['dashed']
    df_connectogram = df_connectogram.merge(
        result_reverse[['StimR', 'ChanR', 'colored_out', 'lw_out', 'value_out', 'dashed_out']],
        on=['StimR', 'ChanR']).reset_index(drop=True)

    # Create the plot
    fig, axes = plt.subplots(4, 3, figsize=(6, 6))
    axes = axes.flatten()

    for i, area_sel in enumerate(regions_all):
        ax = axes[i]
        df_plot = df_connectogram[(np.isin(df_connectogram.StimR, area_sel))].reset_index(drop=True)

        # Reorder the regions
        regions_all_ordered = reorder_letters(area_sel)
        df_plot['Region_order'] = df_plot.StimR
        df_plot.loc[(df_plot.StimR == area_sel), 'Region_order'] = df_plot.loc[(df_plot.StimR == area_sel), 'ChanR']
        df_plot['Region_order'] = pd.Categorical(df_plot['Region_order'], categories=regions_all_ordered, ordered=True)
        df_plot = df_plot.sort_values('Region_order').reset_index(drop=True)
        df_plot['plot_order'] = np.arange(len(df_plot)) + 2000
        df_plot.loc[(df_plot.StimR == area_sel), 'Stim'] = df_plot.loc[(df_plot.StimR == area_sel), 'plot_order']
        df_plot.loc[(df_plot.ChanR == area_sel), 'Chan'] = df_plot.loc[(df_plot.ChanR == area_sel), 'plot_order']

        # Plot using the helper function
        ax = connectogram_region_sleep_mod(df_plot, ax, area=area_sel,
                                 cs='colormap')  # Use the helper function to plot the connectogram
        ax.set_aspect(1.1)

    # Hide empty subplots (if there are any left)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
    print(f'All connectograms in {state}-modulation')


