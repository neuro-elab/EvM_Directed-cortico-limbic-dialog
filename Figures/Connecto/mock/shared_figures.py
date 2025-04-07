import numpy as np
import pandas as pd

from source.globals import PATH_DATA_ATLAS, PATH_DATA_CONNECTIVITY


class Shared:
    def __init__(self):
        super().__init__()

        self.selectedSubjects = []  # a list of subjects ['Subject 2', 'Subject 10']
        self.selectedHemisphere = 'both'  # can be 'both', 'both - merged', 'left' or 'right'
        self.selectedRegions = []  # a list of Regions
        self.selectedTracts = []  # a list of Tracts

        self.selectedMeasure = 'Magnitude'  # can be ...
        self.selectedMeasureColumnName = 'LL'
        self.selectedMeasureColumnNameConnectogramLW = 'Sig'
        self.selectedActivationDisplay = 'Destrieux'  # can be 'Destrieux' or 'Sources'

        self.showSources: bool = False
        self.showCoverage: bool = False
        self.showDestrieux: bool = False
        self.thresholdByMagnitude: bool = False

    def setFilteredData(self):
        # filter your data by self.selectedSubjects, self.selectedHemisphere, self.selectedRegions, self.selectedTracts
        pass

    def setSelectedSubjects(self, subjects: [str]):
        self.selectedSubjects = subjects
        self.setFilteredData()

    def setSelectedHemisphere(self, hemi: str):
        self.selectedHemisphere = hemi.lower()
        self.setFilteredData()

    def setSelectedRegionsAndTracts(self, regions: [str], tracts: [str]):
        self.selectedRegions = regions
        self.selectedTracts = tracts
        self.setFilteredData()

    def setSelectedMeasure(self, measurement: str):
        self.selectedMeasure = measurement

        self.selectedMeasureColumnName = (
            MEASUREMENTS.loc[MEASUREMENTS['measurement'] == measurement, "column_name"]
            .to_list()[0]
        )
        self.selectedMeasureColumnNameConnectogramLW = (
            MEASUREMENTS.loc[MEASUREMENTS['measurement'] == measurement, "column_name_connectogram_lw"]
            .to_list()[0]
        )

    def setSelectedActivationDisplay(self, activationDisplay: str):
        self.selectedActivationDisplay = activationDisplay

    def setInflatedDisplay(self, showSources, showCoverage, showDestrieux, thresholdByMagnitude):
        self.showSources = showSources
        self.showCoverage = showCoverage
        self.showDestrieux = showDestrieux
        self.thresholdByMagnitude = thresholdByMagnitude

    """
    -- REQUEST DATA
    """
    def get_electrode_mni_coordinates(self) -> np.ndarray:
        # todo: return the electrode_mni_coordinates as np.ndarray
        return []

    def get_activations(self, isIncoming: bool, volume: str) -> (np.ndarray, np.ndarray):
        # todo: return the activations_per_sources as (np.ndarray, np.ndarray),
        #  based on
        return ([], [])

    def get_activations_per_region(self, side: str, isIncoming: bool, volume: str='pial'):
        # todo: return the activations_per_region as (np.ndarray, np.ndarray),
        #  based on self.selectedHemisphere, side, self.selectedRegions, self.selectedMeasure, isIncoming, volume
        return

    def get_atlas(self):
        return pd.read_csv(PATH_DATA_ATLAS)

    def get_plot_atlas(self):
        atlas = pd.read_csv(PATH_DATA_ATLAS)
        return atlas[['region', 'plot_color', 'lobe', 'plot_pos', 'plot_order']].drop_duplicates()

    def get_tracts(self) -> (list, list):
        if len(self.selectedTracts) < 1:
            return [], []
        else:
            # todo: return the tracts as (list, list), based on self.selectedTracts, self.selectedHemisphere
            return [], []  # return fibers_list, idx_list

    def get_connectivity(self, all_regions_and_tracts=False):
        if all_regions_and_tracts:
            return pd.read_csv(PATH_DATA_CONNECTIVITY)
        else:
            # todo: return the filtered connectivity
            return pd.read_csv(PATH_DATA_CONNECTIVITY)
