# connectogram_helpers.py
import pandas as pd
import numpy as np
import sys
from collections import deque
import os
from pathlib import Path
cwd = Path.cwd()
# ==== CONFIGURATION ====
# PATH_CONNECTO = '/Users/ellenvanmaren/Desktop/Insel/PhD_Projects/EL_experiment/Codes/Softwares/Connecto/Connecto/'
PATH_Connecto = os.path.join(cwd, 'Connecto')
## adding Conenctogram path
sys.path.append(PATH_Connecto)
from mock.shared_figures import Shared
shared_fake = Shared()

from views.plot_connectogram import ConnectogramPlotter

# Define the colormap and vlim based on the metric
cmap_dict = {
    'Sig': ('custom_Blues_gray', [0, 0.75]),
    'DI': ('custom_PRGn_gray', [-0.75, 0.75]),
    'peak_latency': ('custom_orange_to_gray', [0, 0.1]),
    'LL': ('custom_hot_gray', [0, 5])
}

# Function to reorder regions
def reorder_letters(selected_letter):
    # Calculate the reordered region list
    curve = ['Orbitofrontal', 'Operculo-insular', 'Dorsofrontal', 'Central', 'Cingular', 'Parietal', 'Insulo-temporal',
             'Inferotemporal', 'Occipital']
    straight = ['Mesiotemporal', 'Amygdala', 'Hippocampus']
    full_order = curve + straight[::-1]  # Reverse straight to go counterclockwise

    # Find the index of the selected letter in the combined list
    selected_index = full_order.index(selected_letter)

    # The letter to the left is one position before the selected letter
    left_index = (selected_index + 1) % len(full_order)

    # Rotate the list so that the left letter is first
    reordered = deque(full_order)
    reordered.rotate(-left_index)

    return list(reordered)


# Function to plot the connectogram for a given region
def connectogram_region(df_plot, ax, area='Amygdala', metric='Sig', cs='area', cmap_dict=None):
    # Define the colormap and vlim based on the metric
    cmap_dict = {
        'Sig': ('custom_Blues_gray', [0, 0.75]),
        'DI': ('custom_PRGn_gray', [-0.75, 0.75]),
        'peak_latency': ('custom_orange_to_gray', [0, 0.1]),
        'LL': ('custom_hot_gray', [0, 5])
    }

    cmap, vlim = cmap_dict.get(metric, ('viridis', [0, 1]))  # Default to 'viridis' if metric not in dictionary

    df_plot['show'] = 1
    df_plot = df_plot.loc[(df_plot.lw_out > 0) & (df_plot.lw > 0) & (df_plot.DI > 0) &
                          ((np.isin(df_plot.StimR, area)) | (np.isin(df_plot.ChanR, area)))].reset_index(drop=True)

    if np.sum(df_plot.show) > 0:
        plotter_connecto = ConnectogramPlotter(df_plot, ax)
        plotter_connecto.load_data(df_plot)
        plotter_connecto.show_plot_ax(color_style=cs, cmap=cmap, vmin=vlim[0], vmax=vlim[1], arrow_end=True, lw_scale=2)

    return ax

def connectogram_region_sleep_mod(df_plot, ax, area='Amygdala', cs='area'):
    import matplotlib.colors as mcolors
    import matplotlib as mpl

    # Register colormap only if not already registered
    if 'seismic_gray' not in mpl.colormaps:
        _seismic_gray = [
            (0.0, 0.0, 0.3), (0.0, 0.0, 0.9),
            (0.9, 0.9, 0.9), (0.9, 0.0, 0.0),
            (0.5, 0.0, 0.0)]
        seismic_gray = mcolors.LinearSegmentedColormap.from_list('seismic_gray', _seismic_gray, N=50)
        mpl.colormaps.register(seismic_gray)
    else:
        seismic_gray = mpl.colormaps['seismic_gray']

    # Define the colormap and vlim based on the metric
    vlim = [-0.4, 0.4]
    df_plot['show'] = 1

    df_plot.loc[np.isin(df_plot.ChanR, area), 'sort'] = 1
    df_plot = df_plot.sort_values(by=['sort']).reset_index(drop=True)

    if np.sum(df_plot.show) > 0:
        plotter_connecto = ConnectogramPlotter(df_plot, ax)
        plotter_connecto.load_data(df_plot)
        plotter_connecto.show_plot_ax(color_style=cs, cmap=seismic_gray, vmin=vlim[0], vmax=vlim[1], arrow_end=True,
                                      bi_dir=True)

    return ax