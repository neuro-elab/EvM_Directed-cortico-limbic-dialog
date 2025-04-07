from os import path
from enum import Enum

import numpy as np
import pandas as pd
from visbrain.io import read_freesurfer_mesh

from source.colormaps import register_colormaps

VERSION = 'v0.3.0'

CODE_DIRECTORY = path.dirname(path.dirname(__file__))
IMG_BANNER = path.normpath(path.join(CODE_DIRECTORY, "resources", "logo_banner.png"))
ICON_CONF_SUBJECTS = path.normpath(path.join(CODE_DIRECTORY, "resources", "icons", "conf_subjects.svg"))
ICON_CONF_HEMISPHERES = path.normpath(path.join(CODE_DIRECTORY, "resources", "icons", "conf_hemispheres.svg"))
ICON_CONF_REGIONS = path.normpath(path.join(CODE_DIRECTORY, "resources", "icons", "conf_regions.svg"))
ICON_CONF_TRACTS = path.normpath(path.join(CODE_DIRECTORY, "resources", "icons", "conf_tracts.svg"))
ICON_CONF_STATES = path.normpath(path.join(CODE_DIRECTORY, "resources", "icons", "conf_states.svg"))

ICON_DISP_SIDES = path.normpath(path.join(CODE_DIRECTORY, "resources", "icons", "disp_sides.svg"))
ICON_DISP_PLOTS = path.normpath(path.join(CODE_DIRECTORY, "resources", "icons", "disp_plots.svg"))
ICON_DISP_MEASURES = path.normpath(path.join(CODE_DIRECTORY, "resources", "icons", "disp_measures.svg"))

PATH_PIAL_LH = path.normpath(path.join(CODE_DIRECTORY, "resources", "fs_surf", "lh.pial"))
PATH_PIAL_RH = path.normpath(path.join(CODE_DIRECTORY, "resources", "fs_surf", "rh.pial"))
PATH_BRAINMASK = path.normpath(path.join(CODE_DIRECTORY, "resources", "fs_surf", "brainmask.mgz"))
PATH_INFLATED_LH = path.normpath(path.join(CODE_DIRECTORY, "resources", "fs_surf", "lh.inflated"))
PATH_INFLATED_RH = path.normpath(path.join(CODE_DIRECTORY, "resources", "fs_surf", "rh.inflated"))
PATH_ANNOTS_LH = path.normpath(path.join(CODE_DIRECTORY, "resources", "fs_label", "lh.aparc.a2009s.annot"))
PATH_ANNOTS_RH = path.normpath(path.join(CODE_DIRECTORY, "resources", "fs_label", "rh.aparc.a2009s.annot"))
PATH_SULCUS = path.normpath(path.join(CODE_DIRECTORY, "resources", "fs_surf", "sulcus.npy"))
PATH_GII_SURF = path.normpath(path.join(CODE_DIRECTORY, "resources", "gii_surf"))

PATH_DATA_ATLAS = path.normpath(path.join(CODE_DIRECTORY, "resources", "tables", "data_atlas.csv"))
PATH_DATA_CONNECTIVITY = path.normpath(path.join(CODE_DIRECTORY, "resources", "tables", "data_connectivity.csv"))
PATH_DATA_COORDS = path.normpath(path.join(CODE_DIRECTORY, "resources", "tables", "data_labels_coords.csv"))
PATH_DATA_MEASUREMENTS = path.normpath(path.join(CODE_DIRECTORY, "resources", "tables", "measurements.csv"))
PATH_DATA_REGION_ABBR = path.normpath(path.join(CODE_DIRECTORY, "resources", "tables", "region_abbr.csv"))
REGION_ABBR = pd.read_csv(PATH_DATA_REGION_ABBR)
PATH_DATA_TRACTS = path.normpath(path.join(CODE_DIRECTORY, "resources", "tracts", "HCP1065"))
PATH_DESTRIEUX_BORDER = path.normpath(path.join(CODE_DIRECTORY, "resources", "destrieux_border.csv"))

DICT_TRACTS = {
    "All": "All",
    "AF": "Arcuate Fasciculus",
    "PAT": "Temporo-Parietal Aslant Tract",
    "C_FP": "Cingulum, Frontal Parietal Segment",
    "C_FPH": "Cingulum, Frontal Parahippocampal Segment",
    "C_PH": "Cingulum, Parahippocampal Segment",
    "C_PHP": "Cingulum, Parahippocampal Parietal Segment",
    "C_PO": "Cingulum, Parolfactory Segment",
    "EMC": "Extreme Capsule",
    "FAT": "Frontal Aslant Tract",
    "IFOF": "Inferior Fronto Occipital Fasciculus",
    "ILF": "Inferior Longitudinal Fasciculus",
    "MdLF": "Middle Longitudinal Fasciculus",
    "SLF1": "Superior Longitudinal Fasciculus 1",
    "SLF2": "Superior Longitudinal Fasciculus 2",
    "SLF3": "Superior Longitudinal Fasciculus 3",
    "VOF": "Vertical Occipital Fasciculus"
}

SULCUS = np.load(PATH_SULCUS)
PIAL_VERTICES_RIGHT, _, _ = read_freesurfer_mesh([PATH_PIAL_RH])
PIAL_VERTICES_LEFT, _, _ = read_freesurfer_mesh([PATH_PIAL_LH])
PIAL_VERTICES_BOTH, _, _ = read_freesurfer_mesh([PATH_PIAL_LH, PATH_PIAL_RH])
COLOR_GYRI = np.array([.9, .9, .9, 1.])
COLOR_SULCI = np.array([.6, .6, .6, 1.])
COLOR_MASK = np.array([.6, .6, 0, 1.])
BORDER_VERTICES = pd.read_csv(PATH_DESTRIEUX_BORDER)
BORDER_VERTICES = BORDER_VERTICES['border_vertices'].tolist()


class Site(Enum):
    STIMULATION = 'Outgoing'
    RESPONSE = 'Incoming'

    def isIncoming(self) -> bool:
        if self == Site.RESPONSE:
            return True
        else:
            return False

    def get_chan_to_filter(self) -> str:
        if self == Site.RESPONSE:
            return 'Chan'
        else:
            return 'Stim'

    def get_tract_to_filter(self) -> int:
        if self == Site.RESPONSE:
            return 1
        else:
            return -1

    def get_chan_to_show(self) -> str:
        if self == Site.RESPONSE:
            return 'Stim'
        else:
            return 'Chan'


class Hemisphere(Enum):
    BOTH = 'Both separate'
    MERGED = 'Both merged'
    LEFT = 'Left hemisphere'
    RIGHT = 'Right hemisphere'

    def get_plot_code(self) -> str:
        if self == Hemisphere.BOTH:
            return 'both'
        elif self == Hemisphere.RIGHT:
            return 'right'
        else:  # left and merged
            return 'left'


class Measure:
    def __init__(self, name: str, column_name: str, column_lw: str, colormap: str, vmin: float, vmax: float,
                 show_incoming_and_outgoing: int, description: str):
        self.name = name
        self.col_name = column_name
        self.to_column_lw = column_lw
        self.colormap = colormap
        self.vmin = vmin
        self.vmax = vmax
        self.show_incoming_and_outgoing = True if show_incoming_and_outgoing == 1 else False
        self.description = description


MEASURES: [str, Measure] = {}
measurements = pd.read_csv(PATH_DATA_MEASUREMENTS)
for index, row in measurements.iterrows():
    lw = '' if pd.isna(row.column_name_connectogram_lw) else row.column_name_connectogram_lw
    MEASURES[row.measurement] = Measure(row.measurement, row.column_name, lw,
                                        row.colormap, row.vmin, row.vmax, row.show_incoming_and_outgoing, row.description)


ATLAS = pd.read_csv(PATH_DATA_ATLAS)
PLOT_ATLAS = ATLAS[['region', 'plot_color', 'lobe', 'plot_pos', 'plot_order']].drop_duplicates()
REGION_ABBREVIATION = REGION_ABBR.set_index('region')['region_abbr'].to_dict()

CONNECTOGRAM_MEAN_THRESHOLD = 1000  # If more rows are present, averages are used
CCEP_MAX_PLAY_INDEX = 30  # number of frames for CCEP play
CCEP_MAX_PLAY_TIME = 0.4  # number of spanned seconds for CCEP play

# Register custom colormaps
register_colormaps()
