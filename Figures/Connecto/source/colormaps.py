import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np


def register_colormaps():
    _register_colormap(_white_to_darkmagenta, 'custom_darkmagenta_to_white')
    _register_colormap(_white_to_deepindigo, 'custom_deepindigo_to_white')
    _register_colormap(_grey_yellow_red, 'custom_grey_yellow_red')
    _register_colormap(_orange_to_gray, 'custom_orange_to_gray')
    _register_colormap(_PRGn_gray, 'custom_PRGn_gray')
    _register_colormap(_seismic_gray, 'custom_seismic_gray')
    _register_colormap(_Blues_gray, 'custom_Blues_gray')
    _register_colormap(_hot_gray, 'custom_hot_gray')
    _register_colormap(_destrieux, 'custom_destrieux')

_white_to_darkmagenta = [(.9, .9, .9), (0.5098, 0.0353, 0.2745)]  # for Sigurd paper, dark magenta, for Wake
_white_to_deepindigo = [(.9, .9, .9), (0.1922, 0.1765, 0.4039),]  # for Sigurd paper, deep indigo, for NREM
_grey_yellow_red = [(.9, .9, .9), (.9, .9, 0), (0.9, 0.55, 0), (0.9, 0.25, 0), (0.9, 0, 0)]


_orange_to_gray = [(0.77, 0.3, 0.0), (0.9, 0.9, 0.9)]
# modified from https://github.com/matplotlib/matplotlib/blob/main/lib/matplotlib/_cm.py
_PRGn_gray = [
    (0.25098039215686274,  0.0                ,  0.29411764705882354),
    (0.46274509803921571,  0.16470588235294117,  0.51372549019607838),
    (0.6                ,  0.4392156862745098 ,  0.6705882352941176 ),
    (0.76078431372549016,  0.6470588235294118 ,  0.81176470588235294),
    (0.9,  0.83137254901960789,  0.9),
    (0.9,  0.9,  0.9),
    (0.85098039215686272,  0.9,  0.82745098039215681),
    (0.65098039215686276,  0.85882352941176465,  0.62745098039215685),
    (0.35294117647058826,  0.68235294117647061,  0.38039215686274508),
    (0.10588235294117647,  0.47058823529411764,  0.21568627450980393),
    (0.0                ,  0.26666666666666666,  0.10588235294117647)
]
_seismic_gray = [
    (0.0, 0.0, 0.3), (0.0, 0.0, 0.9),
    (0.9, 0.9, 0.9), (0.9, 0.0, 0.0),
    (0.5, 0.0, 0.0)]
_Blues_gray = [
    (0.8,  0.8,  0.8),
    (0.61960784313725492,  0.792156862745098  ,  0.88235294117647056),
    (0.41960784313725491,  0.68235294117647061,  0.83921568627450982),
    (0.25882352941176473,  0.5725490196078431 ,  0.77647058823529413),
    (0.12941176470588237,  0.44313725490196076,  0.70980392156862748),
    (0.03137254901960784,  0.31764705882352939,  0.61176470588235299),
    (0.03137254901960784,  0.18823529411764706,  0.41960784313725491)
]
_hot_gray = [
    (0.37837482, 0.33958373, 0.33168836),
    (0.69578053, 0.21739553, 0.14538121),
    (0.79466276, 0.25579074, 0.17407871),
    (0.87757695, 0.32612717, 0.23534325),
    (0.95504391, 0.40945533, 0.15777387),
    (0.99950372, 0.52931618, 0.00421048),
    (1.        , 0.6646932 , 0.        ),
    (1.        , 0.79837804, 0.        ),
    (0.98740565, 0.92514676, 0.26546832)
]
_destrieux = [
    (0.4, 0.4, 0.4),
    (0.7725, 0.8980, 0.9607),
    (0.5294, 0.7098, 0.8784),
    (0.3215, 0.4078, 0.6745),
    (0.5294, 0.4313, 0.5960),
    (0.7529, 0.6901, 0.7882),
    (0.9294, 0.7176, 0.7529),
    (0.8862, 0.4627, 0.4078),
    (0.9294, 0.6588, 0.2549),
    (0.9764, 0.8509, 0.3450),
    (0.5450, 0.7607, 0.7686),
    (0.2156, 0.4980, 0.5019),
    (0.8196, 0.8745, 0.4784)
]


def _register_colormap(colors, cmap_name:str, n_bins=200):
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    mpl.colormaps.register(custom_cmap)


"""
import scicomap as sc
# Import the colour map
sc_map = sc.ScicoSequential(cmap="hot")
# Fix the colour map
sc_map.unif_sym_cmap(lift=40, bitonic=False, diffuse=True)

n_lines = 10
#cmap = mpl.colormaps['hot']

# Take colors at regular intervals spanning the colormap.

colors = cmap(np.linspace(0, 1, n_lines))
print('done')
"""
