import numpy as np
import pandas as pd

import views.utils_connectogram_read_data as rd
from views.fig_connectogram import Connectogram


class ConnectogramPlotter:
    def __init__(self, data: pd.DataFrame, ax, show_label: bool = True):
        self.figure = Connectogram(ax=ax)
        self.show_label = show_label

        self.data_con = None
        self.data_nodes = None
        self.load_data(data)

    def load_data(self, data_con: pd.DataFrame):
        # todo: in the main data_con, add 'Stim_ID' and 'Chan_ID'
        data_con = data_con.reset_index(drop=True)

        chan0 = 0
        for subj in np.unique(data_con.Subj):
            data_con.loc[data_con.Subj == subj, 'Stim'] = data_con.loc[data_con.Subj == subj, 'Stim'] + chan0
            data_con.loc[data_con.Subj == subj, 'Chan'] = data_con.loc[data_con.Subj == subj, 'Chan'] + chan0
            chan0 = np.max(data_con.loc[data_con.Subj == subj, ['Stim', 'Chan']].values) + 1
        self.data_con = data_con
        self.data_nodes = rd.get_nodes(data_con)

    def show_plot(self, title, color_style: str, h: str = 'r', cmap: str = None, vmin: float = None, vmax: float = None,
                  lw_scale: int = 2):
        self.figure.setData(self.data_con, self.data_nodes, h)
        self.figure.plot_nodes(self.show_label)
        self.figure.show_con_all(self.data_con, color_style, cmap=cmap, vmin=vmin, vmax=vmax, lw_max=lw_scale)

    def show_plot_ax(self, color_style: str, h: str = 'r', cmap: str = None, vmin: float = None, vmax: float = None,
                     lw_scale: int = 2, bi_dir: bool = False, arrow_end: bool = False):
        # adding function to call from outside of GUI.
        self.figure.setData(self.data_con, self.data_nodes, h)
        self.figure.plot_nodes(self.show_label)
        self.figure.show_con_all(self.data_con, color_style, cmap=cmap, vmin=vmin, vmax=vmax, lw_max=lw_scale,bi_dir = bi_dir, arrow_end = arrow_end)

# # Example usage:
# plotter = ConnectogramPlotter()
# file = 'X:\\4 e-Lab\EvM\Projects\EL_experiment\Analysis\Patients\EL025\BrainMapping\CT\data\\C_A.csv'
# plotter.load_data(file)
# plotter.show_plot()
