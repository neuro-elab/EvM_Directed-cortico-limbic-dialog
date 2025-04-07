
import os
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from connectogram_helpers import connectogram_region, reorder_letters
from pathlib import Path

# ==== CONFIGURATION ====
PATH_Data = os.path.join(Path(__file__).resolve().parent.parent, 'Data')
# ==== LOAD DATA ====
labels = pd.read_csv(os.path.join(PATH_Data, 'data_atlas.csv'))
con_all = pd.read_csv(os.path.join(PATH_Data, 'data_con_figures.csv'))

# Filter connections and prepare the data
con_filtere = con_all[(con_all.Hemi != 'B') & (con_all.Group != 'local')].reset_index(drop=True)
con_filtere_reverse = con_filtere[['sc_id', 'rc_id', 'Num_trial']].reset_index(drop=True)
con_filtere_reverse = con_filtere_reverse.rename(columns={'sc_id': 'rc_id', 'rc_id': 'sc_id', 'Num_trial': 'Num_trial_in'})
con_filtere = con_filtere.merge(con_filtere_reverse, on=['sc_id', 'rc_id'], how='left')

con_filtere_reverse = con_filtere[['sc_id', 'rc_id', 'Sig']].reset_index(drop=True)
con_filtere_reverse = con_filtere_reverse.rename(columns={'sc_id': 'rc_id', 'rc_id': 'sc_id', 'Sig': 'Sig_in'})
con_filtere = con_filtere.merge(con_filtere_reverse, on=['sc_id', 'rc_id'], how='left')
con_filtere['Sig_diff'] = con_filtere['Sig'] - con_filtere['Sig_in']
con_filtere.loc[(con_filtere.Sig == 0) & (con_filtere.Sig_in == 0), 'Sig_diff'] = np.nan

# Threshold for filtering
num_threshold = 10
con_filtere.loc[(con_filtere.Num_trial_in < num_threshold) | (con_filtere.Num_trial < num_threshold), ['Sig_diff', 'DI']] = np.nan

# Get color map from table
region_color_map = labels.drop_duplicates('region').set_index('region')['plot_color'].to_dict()

# Filter and compute statistics
regions_all = np.unique(labels.drop_duplicates('region').region)

# Function to prepare the statistics of the connections
def prepare_statistics(df_con, regions_all):
    arr_connection = []
    df_con.loc[df_con.Sig == 0, 'Sig'] = np.nan
    for region_stim in regions_all:
        for region_resp in regions_all:
            df_region = df_con[(df_con.StimR == region_stim) & (df_con.ChanR == region_resp)].reset_index(drop=True)
            if len(df_region) > 0:
                md = np.median(df_region.DI)
                IQR_low = np.quantile(df_region.DI, 0.25)
                IQR_up = np.quantile(df_region.DI, 0.75)
                md_Sig = np.nanmedian(df_region.Sig)
                IQR_low_Sig = np.nanquantile(df_region.Sig, 0.25)
                IQR_up_Sig = np.nanquantile(df_region.Sig, 0.75)
                n = len(df_region.DI)
                n_pat = len(np.unique(df_region['Subj']))
                res = wilcoxon(df_region.Sig_diff, alternative='two-sided')
                arr_connection.append([region_stim, region_resp, md, f'{md:0.2f} [{IQR_low:0.2f},{IQR_up:0.2f}]',
                                      f'{md_Sig:0.2f} [{IQR_low_Sig:0.2f},{IQR_up_Sig:0.2f}]', n, n_pat, res.pvalue])
    df_region_con = pd.DataFrame(arr_connection, columns=['StimR', 'ChanR', 'Median', 'DI Median [IQR]', 'Prob Median [IQR]',
                                                          'N_connection', 'N_pat', 'p_value'])

    # FDR correction
    _, pvals_corrected, _, _ = multipletests(df_region_con['p_value'], method='fdr_bh')
    df_region_con['p_value_corrected'] = pvals_corrected
    return df_region_con

# Filter and compute statistic
regions_all = labels.drop_duplicates('region').sort_values('plot_order')['region'].values
df_region_con = prepare_statistics(con_filtere[~np.isnan(con_filtere.Sig_diff)].reset_index(drop=True), regions_all)


# prepare Connectogram DATA
metrics = ['DI', 'Sig', 'peak_latency']
df_mean_R = con_filtere.groupby(['StimR','ChanR'], as_index=False)[metrics].mean().reset_index(drop=True)
df_mean_R['Subj'] ='EL000'

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


metric = 'DI'
df_mean_R_connecto = df_mean_R.copy()
df_mean_R_connecto['pair'] = df_mean_R_connecto['StimR']+'_'+df_mean_R['ChanR']
# add stats for significane level
df_region_con['colored'] = 0
df_region_con.loc[(df_region_con.p_value_corrected<0.05), 'colored'] = 1
df_mean_R_connecto = df_mean_R_connecto.merge(df_region_con[['StimR','ChanR', 'colored']], on = ['StimR','ChanR'], how='left').reset_index(drop=True)
# add lw, value and dashed
df_mean_R_connecto['lw'] = df_mean_R_connecto['Sig']
df_mean_R_connecto['value'] = df_mean_R_connecto[metric]
df_mean_R_connecto['dashed'] = 0
df_mean_R_connecto.loc[df_mean_R_connecto.peak_latency>0.065, 'dashed'] = 1

result_reverse = df_mean_R_connecto[['StimR', 'ChanR', 'lw', 'value', 'dashed', 'colored']].reset_index(drop=True)
result_reverse = result_reverse.rename(columns={'StimR': 'ChanR','ChanR': 'StimR' })
result_reverse['lw_out'] = result_reverse['lw']
result_reverse['value_out'] = result_reverse['value']
result_reverse['colored_out'] = result_reverse['colored']
df_mean_R_connecto = df_mean_R_connecto.merge(result_reverse[['StimR', 'ChanR', 'lw_out', 'value_out', 'colored_out']], on=['StimR', 'ChanR']).reset_index(drop=True)

# Create the plot
fig, axes = plt.subplots(4, 3, figsize=(6, 6))
axes = axes.flatten()

for i, area_sel in enumerate(regions_all):
    ax = axes[i]
    df_plot = df_mean_R_connecto[(np.isin(df_mean_R_connecto.StimR, area_sel)) | (np.isin(df_mean_R_connecto.ChanR, area_sel))].reset_index(
        drop=True)

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
    ax = connectogram_region(df_plot, ax, area=area_sel, metric='DI',
                             cs='area')  # Use the helper function to plot the connectogram
    ax.set_aspect(1.1)

# Hide empty subplots (if there are any left)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
print('All connectograms')


