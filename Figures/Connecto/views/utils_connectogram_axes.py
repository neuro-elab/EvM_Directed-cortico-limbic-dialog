import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import special as ssp

from source.globals import PLOT_ATLAS

plt.rcParams.update({
    'font.family': 'arial',
    'font.size': 12,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 9,
    'svg.fonttype': 'none',
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 9,
    'figure.titlesize': 10,
})


def despine(ax):
    for spine in ax.spines:
        ax.spines[spine].set_visible(False)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)


def B(i, N, t):
    val = ssp.comb(N, i) * t ** i * (1. - t) ** (N - i)
    return val


def P(t, X):
    """
     xx = P(t, X)

     Evaluates a Bezier curve for the points in X.

     Inputs:
      X is a list (or array) or 2D coords: start , references , end
      t is a number (or list of numbers) in [0,1] where you want to
        evaluate the Bezier curve

     Output:
      xx is the set of 2D points along the Bezier curve
    """
    X = np.array(X)
    N, d = np.shape(X)  # Number of points, Dimension of points
    N = N - 1
    xx = np.zeros((len(t), d))

    for i in range(N + 1):
        xx += np.outer(B(i, N, t), X[i])

    return xx


def initialize_data_edges(edges: pd.DataFrame, bi_dir: bool) -> pd.DataFrame:
    required_columns = ['value', 'lw', 'dashed', 'colored', 'effect']
    default_values = [1, 1, 0, 1, 0]
    for col, default in zip(required_columns, default_values):
        if col not in edges:
            edges[col] = default

    #if bi_dir:
    for col in required_columns[:-1]:  # Exclude 'effect' for bi-directional handling
        if f'{col}_out' not in edges:
            edges[f'{col}_out'] = edges[col]
    return edges


def get_line_style_and_width(data_edges: pd.DataFrame, index: int, bi_dir: bool = False) -> (float, str, float, str):
    lw = data_edges.lw.values[index]
    ls = '-' if data_edges.dashed.values[index] == 0 else '--'

    # if bi_dir:
    lw_out = data_edges.lw_out.values[index]
    ls_out = '-' if data_edges.dashed_out.values[index] == 0 else '--'
    return lw, ls, lw_out, ls_out

    #return lw, ls, None, None


def get_colors(color_style: str, edges: pd.DataFrame, index: int, r1, vmin: float, vmax: float,
               colormap, bi_dir=False) -> ([float], [float]):
    col = [169. / 255, 169. / 255, 169. / 255]
    col_out = [169. / 255, 169. / 255, 169. / 255]

    if color_style == 'area':
        if edges.colored.values[index]:
            if len(PLOT_ATLAS.loc[PLOT_ATLAS.region == r1, 'plot_color'].values) > 0:
                col = PLOT_ATLAS.loc[PLOT_ATLAS.region == r1, 'plot_color'].values[0]

    elif color_style == 'colormap':
        normalized_value = (edges.value.values[index] - vmin) / (vmax - vmin)
        col = colormap(normalized_value)
        if bi_dir and edges.colored_out.values[index]:
            normalized_value_out = (edges.value_out.values[index] - vmin) / (vmax - vmin)
            col_out = colormap(normalized_value_out)

    return col, col_out


def plot_curved_text(ax, radius: float, theta_start: float, theta_end: float, abbreviation: str):
    center_angle = (theta_start + theta_end + np.pi) / 2.
    x, y = radius * np.cos(center_angle), radius * np.sin(center_angle)
    rotation = np.degrees(center_angle) - 90

    ax.text(x, y, abbreviation + '.', rotation=rotation, rotation_mode='anchor', ha='center', va='center', fontsize=7,
            color='white')
