import os
import numpy as np
import scipy
import tqdm
import pandas as pd
import sys
import re
from scipy.spatial import distance
import scipy.io
import matplotlib.cm as cm
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
# Add the Figures directory to sys.path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Figures'))

# Now you can import plot_inflated from brain_plots
from brain_plots import plot_inflated

sub_path = '/Volumes/vellen/PhD/EL_experiment' #'X:\\4 e-Lab\\' # y:\\eLab
CIRC_AREAS_FILEPATH = 'X:\\4 e-Lab\EvM\Projects\EL_experiment\Analysis\Patients\Across\elab_labels.xlsx'

if os.name == "posix":  # For Linux/macOS
    CIRC_AREAS_FILEPATH = os.path.join(
        "/Users/ellenvanmaren/Desktop/Insel/PhD_Projects/EL_experiment/Codes/Softwares/Connectogram",
        "circ_areas.xlsx"
    )
else:  # For Windows
    CIRC_AREAS_FILEPATH = os.path.join("X:", "4 e-Lab", "e-Lab shared code", "Softwares", "Connectogram",
                                       "circ_areas.xlsx"
                                       )

# Function to compute the Euclidean distance between points in coord
def compute_distance(row, coord):
    point_A = coord[row['Stim']]
    point_B = coord[row['Chan']]
    return np.linalg.norm(point_A - point_B)


# Apply the function to each row in the dataframe to create the d_inf column df[‘d_inf’] = df.apply(compute_distance, axis=1)

def adding_distance_infalted(df, lbls, subj):
    if not "native_x"in lbls:
        lbls['native_x'] = lbls['x']
        lbls['native_y'] = lbls['y']
        lbls['native_z'] = lbls['z']
    # Assuming plot_inflated.get_inflated_coord(lbls, subj, mni=False) returns the coord array
    coord = plot_inflated.get_inflated_coord(lbls, subj, mni=True)
    # Apply the function to each row in the dataframe to create the d_inf column
    df['d_inf'] = df.apply(lambda row: compute_distance(row, coord), axis=1)
    return df


def adding_distance_tracts(df, df_atlas, atlas=False):
    # Correct renaming of columns with a dictionary and using inplace or assignment
    df_atlas = df_atlas.rename(columns={'Count': 'tract_num', 'Length': 'tract_dist'})
    for col in ['tract_num', 'tract_dist']:
        if col in df:
            df = df.drop(columns=col)
    # Ensure that 'Stim' and 'Chan' are columns to be merged on both DataFrames
    # Correct the usage of selecting multiple columns from df_atlas
    if atlas:
        df = df.merge(df_atlas, on=['Stim', 'Chan'],
                      how='left').reset_index(drop=True)

    else:
        df = df.merge(df_atlas[['Stim', 'Chan', 'tract_num', 'tract_dist']], on=['Stim', 'Chan'],
                      how='left').reset_index(drop=True)
    return df


def get_color(group='Dist'):
    cmap_org = 'winter'
    n = 45
    viridis = cm.get_cmap(cmap_org, n)
    viridis_org = viridis(np.linspace(0, 1, n))
    color_b = np.array([viridis_org[0], viridis_org[15], viridis_org[30], viridis_org[-1]])

    n = 100
    viridis = cm.get_cmap(cmap_org, n)
    color_d = viridis(np.linspace(0, 1, n))

    inter_col = np.zeros((15, 4))
    for i in range(4):
        inter_col[:, i] = np.linspace(color_b[0, i], color_b[1, i], 15)

    # 1.
    inter_col = np.zeros((15, 4))
    for i in range(4):
        inter_col[:, i] = np.linspace(color_b[0, i], color_b[1, i], 15)

    color_d[0:15, :] = inter_col

    # 2.
    inter_col = np.zeros((15, 4))
    for i in range(4):
        inter_col[:, i] = np.linspace(color_b[1, i], color_b[2, i], 15)

    color_d[15:30, :] = inter_col

    # 3.
    inter_col = np.zeros((70, 4))
    for i in range(4):
        inter_col[:, i] = np.linspace(color_b[2, i], color_b[3, i], 70)

    color_d[30:100, :] = inter_col

    color_dist = np.array([color_d[0], color_d[20], color_d[70]])
    color_group = np.zeros((3, 3))
    color_group[0, :] = np.array([241, 93, 93]) / 255
    color_group[1, :] = np.array([157, 191, 217]) / 255
    color_group[2, :] = np.array([162, 209, 164]) / 255

    color_elab = np.zeros((4, 3))
    color_elab[0, :] = np.array([31, 78, 121]) / 255
    color_elab[1, :] = np.array([189, 215, 28]) / 255
    color_elab[2, :] = np.array([0.256, 0.574, 0.431])
    color_elab[3, :] = np.array([1, 0.574, 0])

    return color_d, color_dist, color_group, color_elab


def group_connections(df):
    """
    Group connections into 'local', 'direct', or 'indirect' based on certain conditions.

    - 'local': Connections in same community and hemisphere
    - 'direct': Non-local connections with peak latency <= 65 ms.
    - 'indirect': Non-local connections with peak latency > 65 ms.

    Parameters:
    df (pd.DataFrame): DataFrame containing the connections with columns 'd', 'StimR', 'ChanR', 'H', and 'peak_latency'.

    Returns:
    pd.DataFrame: DataFrame with a new column 'Group' indicating the group of each connection.
    """

    # Initialize all connections as 'direct' first

    conditions = [(df['StimR'] == df['ChanR']) & (df['Hemi'] != 'B')& (df['d_inf'] <= 25), df['peak_latency'] <= 0.065]
    choices = ['local', 'direct']
    df['Group'] = np.select(conditions, choices, default='indirect')
    if "Sig" in df:
        df.loc[(df.Sig == 0)&(df.Group != 'local'), 'Group'] = np.nan

    return df


def group_connections_0(df):
    """
    Group connections into 'local', 'direct', or 'indirect' based on certain conditions.

    - 'local': Connections with d <= 20, h == 0, and not part of the forbidden pairs.
    - 'direct': Non-local connections with onset <= 20 ms.
    - 'indirect': Non-local connections with onset > 20 ms.

    Parameters:
    df (pd.DataFrame): DataFrame containing the connections with columns 'd', 'StimR', 'ChanR', 'H', and 'delay'.

    Returns:
    pd.DataFrame: DataFrame with a new column 'Group' indicating the group of each connection.
    """

    # Define impossible local pairs
    impossible_local_pairs = set([('Frontal', 'Temporal'), ('Frontal', 'Basotemporal')])

    # Initialize all connections as 'direct' first
    df['Group'] = 'direct'

    # Conditions for local connections
    is_local = (df['d'] <= 20) & (df['H'] == 0)
    for pair in impossible_local_pairs:
        is_local &= ~((df['StimR'] == pair[0]) & (df['ChanR'] == pair[1])) | ~(
                (df['ChanR'] == pair[0]) & (df['StimR'] == pair[1]))

    # Assign 'local' to the filtered connections
    df.loc[is_local, 'Group'] = 'local'

    # Assign 'indirect' to non-local connections with delay > 20
    df.loc[(df['Group'] == 'direct') & (df['delay'] > 0.02), 'Group'] = 'indirect'

    return df


def adding_hemisphere(df, lbls):
    # Determine hemisphere based on x coordinate
    if not "native_x"in lbls:
        lbls['native_x'] = lbls['x']
        lbls['native_y'] = lbls['y']
        lbls['native_z'] = lbls['z']

    lbls['Hemisphere'] = np.where(lbls['native_x'] > 0, 'R', 'L')

    # Create mapping dictionary
    if not "c_id" in lbls:
        lbls['c_id'] = np.arange(len(lbls))
    mapping_dict = lbls.set_index('c_id')['Hemisphere'].to_dict()
    if "sc_id" in df:
        # Map Stim and Chan indices to their corresponding hemispheres
        df['StimHemi'] = df['sc_id'].map(mapping_dict)
        df['ChanHemi'] = df['rc_id'].map(mapping_dict)
    else:
        # Map Stim and Chan indices to their corresponding hemispheres
        df['StimHemi'] = df['Stim'].map(mapping_dict)
        df['ChanHemi'] = df['Chan'].map(mapping_dict)

    # Determine if Stim and Chan are in the same hemisphere
    df['Hemi'] = np.where(df['StimHemi'] == df['ChanHemi'], df['StimHemi'], 'B')

    # Drop the intermediate columns
    df.drop(['StimHemi', 'ChanHemi'], axis=1, inplace=True)

    return df


def adding_anatomy(df, pair=1, area='Destrieux'):
    # area == 'Region' or 'Area'
    df = df.reset_index(drop=True)

    atlas = pd.read_excel(CIRC_AREAS_FILEPATH, sheet_name='atlas')
    if pair:
        for subregion in np.unique(df[['StimA', 'ChanA']]):
            region = atlas.loc[atlas.Abbreviation == subregion, area].values

            if len(region) > 0:
                if region[0] == "entorhinal_cortex":
                    region[0] = 'Entorhinal'
                elif region[0] == "AnteroHippocampus":
                    region[0] = 'Hippocampus'
                elif region[0] == "PosteroHippocampus":
                    region[0] = 'Hippocampus'
                df.loc[df.StimA == subregion, 'StimR'] = region[0]
                df.loc[df.ChanA == subregion, 'ChanR'] = region[0]
            else:
                region = atlas.loc[atlas.Subregion == subregion, area].values
                if len(region) > 0:
                    if region[0] == "entorhinal_cortex":
                        region[0] = 'Entorhinal'
                    elif region[0] == "AnteroHippocampus":
                        region[0] = 'Hippocampus'
                    elif region[0] == "PosteroHippocampus":
                        region[0] = 'Hippocampus'
                    df.loc[df.StimA == subregion, 'StimR'] = region[0]
                    df.loc[df.ChanA == subregion, 'ChanR'] = region[0]
                else:
                    df.loc[df.StimA == subregion, 'StimR'] = 'U'
                    df.loc[df.ChanA == subregion, 'ChanR'] = 'U'
    else:
        for subregion in np.unique(df[['ChanA']]):
            region = atlas.loc[atlas.Abbreviation == subregion, area].values
            if len(region) > 0:
                df.loc[df.ChanA == subregion, 'ChanR'] = region[0]
            else:
                df.loc[df.ChanA == subregion, 'ChanR'] = 'U'
    return df


def adding_distance_tracts(df, df_atlas, atlas=False):
    # Correct renaming of columns with a dictionary and using inplace or assignment
    df_atlas = df_atlas.rename(columns={'Count': 'tract_num', 'Length': 'tract_dist'})
    for col in ['tract_num', 'tract_dist']:
        if col in df:
            df = df.drop(columns=col)
    # Ensure that 'Stim' and 'Chan' are columns to be merged on both DataFrames
    # Correct the usage of selecting multiple columns from df_atlas
    if atlas:
        df = df.merge(df_atlas, on=['Stim', 'Chan'],
                      how='left').reset_index(drop=True)

    else:
        df = df.merge(df_atlas[['Stim', 'Chan', 'tract_num', 'tract_dist']], on=['Stim', 'Chan'],
                      how='left').reset_index(drop=True)
    return df


def adding_distance_tracts_matrix(df, M_distances, M_counts):
    df['tract_dist'] = 0
    df['tract_num'] = 0
    for s in np.unique(df.Stim).astype('int'):
        for c in np.unique(df.Chan).astype('int'):
            df.loc[(df.Stim == s) & (df.Chan == c), 'tract_dist'] = M_distances[s, c]
            df.loc[(df.Stim == s) & (df.Chan == c), 'tract_num'] = M_counts[s, c]
    return df


def adding_distance(df, coord_all):
    df['d'] = 0
    for s in np.unique(df.Stim):
        s = np.int64(s)
        for c in np.unique(df.Chan):
            c = np.int64(c)
            df.loc[(df.Stim == s) & (df.Chan == c), 'd'] = np.round(
                distance.euclidean(coord_all[s], coord_all[c]), 2)
    return df


def adding_mni_coord(lbls, subj):
    # Load Lookup from camille.
    path_gen = os.path.join(sub_path, 'Patients', subj, 'Electrodes', "Lookup.xlsx")
    lookup = pd.read_excel(path_gen, header=0, sheet_name='bipoles')

    if "type" in lookup:
        lookup = lookup[lookup['type'] == 'lead'].reset_index(drop=True)

    # Rounding coordinates
    for coord in ['native_x', 'native_y', 'native_z', 'mni_x', 'mni_y', 'mni_z']:
        lookup[coord] = np.round(lookup[coord], 2)

    for coord in ['x', 'y', 'z']:
        lbls[coord] = np.round(lbls[coord], 2)

    # Preparing for merge: Renaming lbls columns for matching with lookup
    lbls.rename(columns={'x': 'native_x', 'y': 'native_y', 'z': 'native_z'}, inplace=True)

    # Merging dataframes on rounded coordinates
    lbls = lbls.merge(lookup[['native_x', 'native_y', 'native_z', 'mni_x', 'mni_y', 'mni_z']],
                      on=['native_x', 'native_y', 'native_z'], how='left')

    return lbls


def adding_abbr(data_A, lbls):
    labels_all = lbls.label.values
    data_A['ChanA'] = 'Unknown'
    data_A['StimA'] = 'Unknown'
    for c in np.unique(data_A[['sc_id', 'rc_id']]).astype('int'):
        data_A.loc[data_A.rc_id == c, 'ChanA'] = " ".join(re.findall("[a-zA-Z_]+", labels_all[c]))
        data_A.loc[data_A.sc_id == c, 'StimA'] = " ".join(re.findall("[a-zA-Z_]+", labels_all[c]))
    return data_A


def adding_community(data_con):
    # area == 'Region' or 'Area'
    CIRC_AREAS_FILEPATH = 'X:\\4 e-Lab\e-Lab shared code\Softwares\Connectogram\circ_areas.xlsx'
    atlas = pd.read_excel(CIRC_AREAS_FILEPATH, sheet_name='atlas')
    # Convert all abbreviation and StimA/ChanA to lowercase for case insensitive comparison
    atlas['Abbreviation_lower'] = atlas['Abbreviation'].str.lower()
    data_con['StimA_lower'] = data_con['StimA'].str.lower()
    data_con['ChanA_lower'] = data_con['ChanA'].str.lower()

    # Create a mapping dictionary from abbreviation to community
    abbrev_to_community = atlas.set_index('Abbreviation_lower')['Community'].to_dict()

    # Map communities to StimA and ChanA
    data_con['StimR'] = data_con['StimA_lower'].map(abbrev_to_community).fillna('U')
    data_con['ChanR'] = data_con['ChanA_lower'].map(abbrev_to_community).fillna('U')

    # Drop the temporary lowercase columns
    data_con.drop(['StimA_lower', 'ChanA_lower'], axis=1, inplace=True)

    return data_con


def adding_destrieux(data_con, area='Destrieux'):
    # area == 'Region' or 'Area'
    CIRC_AREAS_FILEPATH = 'X:\\4 e-Lab\e-Lab shared code\Softwares\Connectogram\circ_areas.xlsx'
    atlas = pd.read_excel(CIRC_AREAS_FILEPATH, sheet_name='atlas')

    # Convert all abbreviation and StimA/ChanA to lowercase for case insensitive comparison
    atlas['Abbreviation_lower'] = atlas['Abbreviation'].str.lower()
    data_con['StimA_lower'] = data_con['StimA'].str.lower()
    data_con['ChanA_lower'] = data_con['ChanA'].str.lower()

    # Create a mapping dictionary from abbreviation to region
    abbrev_to_region = atlas.set_index('Abbreviation_lower')[area].to_dict()

    # Map regions to StimA and ChanA
    data_con['StimD'] = data_con['StimA_lower'].map(abbrev_to_region).fillna('U')
    data_con['ChanD'] = data_con['ChanA_lower'].map(abbrev_to_region).fillna('U')

    # Drop the temporary lowercase columns
    data_con.drop(['StimA_lower', 'ChanA_lower'], axis=1, inplace=True)

    return data_con


def adding_area(data_A, lbls, pair=1):
    labels_all = lbls.label.values
    data_A['ChanA'] = 'Unknown'
    data_A['StimA'] = 'Unknown'
    if pair:
        for c in np.unique(data_A[['Chan', 'Stim']]).astype('int'):
            data_A.loc[data_A.Chan == c, 'ChanA'] = " ".join(re.findall("[a-zA-Z_]+", labels_all[c]))
            data_A.loc[data_A.Stim == c, 'StimA'] = " ".join(re.findall("[a-zA-Z_]+", labels_all[c]))
            if "Hemisphere" in lbls:
                chans = data_A.loc[data_A.Stim == c, 'Chan'].values.astype('int')
                data_A.loc[data_A.Stim == c, 'H'] = np.array(lbls.Hemisphere[chans] != lbls.Hemisphere[c]) * 1
    else:
        for c in np.unique(data_A[['Chan']]).astype('int'):
            data_A.loc[data_A.Chan == c, 'ChanA'] = " ".join(re.findall("[a-zA-Z_]+", labels_all[c]))
    return data_A


def adding_subregion(data_con, pair=1, area='Subregion'):
    # area == 'Region' or 'Area'
    CIRC_AREAS_FILEPATH = 'X:\\4 e-Lab\e-Lab shared code\Softwares\Connectogram\circ_areas.xlsx'
    atlas = pd.read_excel(CIRC_AREAS_FILEPATH, sheet_name='atlas')
    if pair:
        for subregion in np.unique(data_con[['StimA', 'ChanA']]):
            region = atlas.loc[atlas.Abbreviation == subregion, area].values
            if len(region) > 0:
                data_con.loc[data_con.StimA == subregion, 'StimSR'] = region[0]
                data_con.loc[data_con.ChanA == subregion, 'ChanSR'] = region[0]
            else:
                print(subregion)
                data_con.loc[data_con.StimA == subregion, 'StimSR'] = 'U'
                data_con.loc[data_con.ChanA == subregion, 'ChanSR'] = 'U'
    else:
        for subregion in np.unique(data_con[['ChanA']]):
            region = atlas.loc[atlas.Abbreviation == subregion, area].values
            if len(region) > 0:
                data_con.loc[data_con.ChanA == subregion, 'ChanSR'] = region[0]
            else:
                data_con.loc[data_con.ChanA == subregion, 'ChanSR'] = 'U'
    return data_con


# todo: cleann all adding region /area

def adding_region_old(data_con, area='Region'):
    # area == 'Region' or 'Area'
    CIRC_AREAS_FILEPATH = 'X:\\4 e-Lab\e-Lab shared code\Softwares\Connectogram\circ_areas.xlsx'
    atlas = pd.read_excel(CIRC_AREAS_FILEPATH, sheet_name='atlas')
    for subregion in np.unique(data_con[['StimA', 'ChanA']]):
        region = atlas.loc[atlas.Abbreviation == subregion, area].values
        if len(region) > 0:
            data_con.loc[data_con.StimA == subregion, 'StimR'] = region[0]
            data_con.loc[data_con.ChanA == subregion, 'ChanR'] = region[0]
        else:
            print(subregion)
            data_con.loc[data_con.StimA == subregion, 'StimR'] = 'U'
            data_con.loc[data_con.ChanA == subregion, 'ChanR'] = 'U'

    return data_con


def adding_region(data_con, pair=1, area='Region'):
    # area == 'Region' or 'Area'
    CIRC_AREAS_FILEPATH = 'X:\\4 e-Lab\e-Lab shared code\Softwares\Connectogram\circ_areas.xlsx'
    atlas = pd.read_excel(CIRC_AREAS_FILEPATH, sheet_name='atlas')
    if area == 'Region':
        stim_col = 'StimR'
        resp_col = 'ChanR'
    elif area == 'Destrieux':
        stim_col = 'StimD'
        resp_col = 'ChanD'
    else:
        stim_col = 'StimL'
        resp_col = 'ChanL'
    if pair:
        for subregion in np.unique(data_con[['StimA', 'ChanA']]):
            region = atlas.loc[atlas.Abbreviation == subregion, area].values
            if len(region) > 0:
                data_con.loc[data_con.StimA == subregion, stim_col] = region[0]
                data_con.loc[data_con.ChanA == subregion, resp_col] = region[0]
            else:
                print(subregion)
                data_con.loc[data_con.StimA == subregion, stim_col] = 'U'
                data_con.loc[data_con.ChanA == subregion, resp_col] = 'U'
    else:
        for subregion in np.unique(data_con[['ChanA']]):
            region = atlas.loc[atlas.Abbreviation == subregion, area].values
            if len(region) > 0:
                data_con.loc[data_con.ChanA == subregion, resp_col] = region[0]
            else:
                data_con.loc[data_con.ChanA == subregion, resp_col] = 'U'
    return data_con



def get_connections_BL(subjs, sub_path):
    # path_export = os.path.join(sub_path,
    #                            'EvM\\Projects\\EL_experiment\Analysis\Patients\Across\BrainMapping\General\data\\')
    start = 1
    for subj in subjs:
        path_patient_analysis = os.path.join(sub_path, 'EvM', 'Projects', 'EL_experiment', 'Analysis', 'Patients', subj)
        summary_gen_path = path_patient_analysis + '\\BrainMapping\\' + 'CR' + '\\data\\summ_general_BL.csv'  # summary_general

        con_summary_all = pd.read_csv(summary_gen_path)
        con_summary_all = con_summary_all.drop_duplicates()
        con_summary_all['Subj'] = subj
        if start:
            con_all = con_summary_all.reset_index(drop=True)
            start = 0
        else:
            con_all = pd.concat([con_all, con_summary_all]).reset_index(drop=True)
    return con_all


def get_connections(subjs, sub_path):
    # path_export = os.path.join(sub_path,
    #                            'EvM\\Projects\\EL_experiment\Analysis\Patients\Across\BrainMapping\General\data\\')
    start = 1
    for subj in subjs:
        path_patient_analysis = os.path.join(sub_path, 'EvM', 'Projects', 'EL_experiment', 'Analysis', 'Patients', subj)
        summary_gen_path = path_patient_analysis + '\\BrainMapping\\' + 'CR' + '\\data\\summ_general.csv'  # summary_general

        con_summary_all = pd.read_csv(summary_gen_path)
        # Columns to be updated
        col_update = ['Sig', 'DI', 'Sig_diff']

        # Drop the columns to be updated if they exist in the original table
        con_summary_all = con_summary_all.drop(columns=[col for col in col_update if col in con_summary_all],
                                               errors='ignore')
        summary_wake_path = path_patient_analysis + '\\BrainMapping\\' + 'CR' + '\\data\\summ_wake.csv'  # summary_general

        # Read the new summary_wake table
        con_summary_w = pd.read_csv(summary_wake_path)

        # Merge the new data into the original table
        con_summary_all = con_summary_all.merge(con_summary_w[['Stim', 'Chan', 'Sig', 'DI', 'Sig_diff']],
                                                on=['Stim', 'Chan'], how='left')
        # Drop duplicates if necessary
        con_summary_all = con_summary_all.drop_duplicates()
        con_summary_all['Subj'] = subj
        if start:
            con_all = con_summary_all.reset_index(drop=True)
            start = 0
        else:
            con_all = pd.concat([con_all, con_summary_all]).reset_index(drop=True)
    return con_all



def get_AUC(subjs, path_analysis, ss='general'):
    # path_export = os.path.join(sub_path,
    #                            'EvM\\Projects\\EL_experiment\Analysis\Patients\Across\BrainMapping\General\data\\')

    start = 1
    for subj in subjs:
        # if platform.system() == "Linux":
        #     path_patient_analysis = os.path.join('/Volumes/vellen/PhD/EL_experiment/Analysis/', 'Patients', subj)
        # else:
        #     path_analysis =
        #     path_patient_analysis = os.path.join(sub_path, 'EvM', 'Projects', 'EL_experiment', 'Analysis', 'Patients',
        #                                          subj)
        path_patient_analysis = os.path.join(path_analysis, 'Patients',
                                             subj)
        if ss == 'general':
            file = os.path.join(path_patient_analysis, 'InputOutput', 'CR', 'data', 'AUC_mean.csv')
        else:
            file = os.path.join(path_patient_analysis, 'InputOutput', 'CR', 'data', 'AUC_mean_SS.csv')

        con_summary_all = pd.read_csv(file)
        con_summary_all = con_summary_all.drop_duplicates()
        con_summary_all['Subj'] = subj
        if start:
            con_all = con_summary_all
            start = 0
        else:
            con_all = pd.concat([con_all, con_summary_all]).reset_index(drop=True)
    return con_all



def get_connections_sleep(subjs, sub_path, ss_sel, DI_metric='ratio'):
    folder = 'BrainMapping'
    cond_folder = 'CR'
    start = 1
    for subj in subjs:
        for ss in ss_sel:
            path_patient_analysis = os.path.join(sub_path, 'EvM', 'Projects', 'EL_experiment', 'Analysis', 'Patients',
                                                 subj)
            summary_gen_path = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\summ_' + ss + '.csv'  # summary_general

            con_summary_all = pd.read_csv(summary_gen_path)
            con_summary_all = con_summary_all.drop_duplicates()
            con_summary_all['Subj'] = subj
            if start:
                con_sleep = con_summary_all
                start = 0
            else:
                con_sleep = pd.concat([con_sleep, con_summary_all]).reset_index(drop=True)
    return con_sleep


def save_labels_across(subjs):
    col_keep = ['label', 'native_x', 'native_y', 'native_z', 'mni_x', 'mni_y', 'mni_z']
    sub_path = 'X:\\4 e-Lab\\'

    data_all = pd.DataFrame()
    for subj in subjs:
        print(subj, end='\r')
        file = os.path.join(sub_path, 'Patients', subj, 'Electrodes',
                            f"{subj}_labels.xlsx")
        #
        if os.path.isfile(file):
            # tracts_distances = pd.read_csv(WM_dist_file).values[:, 1:]
            # tracts_counts = pd.read_csv(WM_count_file).values[:, 1:]
            lbls = pd.read_excel(file, header=0, sheet_name='BP')

            if "type" in lbls:
                lbls = lbls[lbls['type'] == 'SEEG'].reset_index(drop=True)
            lbls = adding_mni_coord(lbls, subj)

            lbls = lbls[col_keep]
            lbls.insert(0, 'c', np.arange(len(lbls)))
            lbls.insert(0, 'Subj', subj)
            # data_subj = adding_distance_tracts(con_all[con_all.Subj == subj].reset_index(drop=True), tract_matrix, True)
            data_all = pd.concat([data_all, lbls], ignore_index=True)
    file_save = os.path.join(sub_path, 'EvM', 'Projects', 'EL_experiment', 'Analysis', 'Patients', 'Across',
                             'Connecto',
                             'Data', 'labels_across.csv')
    data_all.to_csv(file_save, header=True, index=False)
    return data_all

def save_data_across(subjs, load_tract =True):
    # connection data:
    con_all = get_connections(subjs, sub_path)
    con_all = con_all[con_all.Sig > -1].reset_index(drop=True)
    con_all.loc[con_all.Sig == 0, 'delay'] = np.nan
    con_all.loc[con_all.Sig == 0, 'peak_latency'] = np.nan
    con_all.loc[con_all.Sig == 0, 'LL'] = np.nan

    # Adding tract data
    if load_tract:
        start_r = 20
        end_r = 20
        print(f"Processing for start_r={start_r} and end_r={end_r}")
        # Load and process data for each subject
        data_all = pd.DataFrame()
        for subj in subjs:
            print(subj, end='\r')
            file = os.path.join(sub_path, 'Patients', subj, 'Electrodes', 'Tracts',
                                f"{subj}_tracts_contacts_s{start_r}_e{end_r}_Region.csv")
            #
            if os.path.isfile(file):
                tract_matrix = pd.read_csv(file)
                data_subj = adding_distance_tracts(con_all[con_all.Subj == subj].reset_index(drop=True),
                                                      tract_matrix, True)
                data_all = pd.concat([data_all, data_subj], ignore_index=True)
    else:
        data_all = con_all.copy(True)

    # add sleep modulation
    df_sleep_mod = add_sleep_modulation(subjs)
    data_all = pd.merge(data_all, df_sleep_mod, on=['Subj', 'Stim', 'Chan'], how='left').reset_index(drop=True)
    if load_tract:
        col_keep = ['Subj', 'Stim', 'Chan', 'LL', 'Sig', 'DI', 'Sig_diff','delay', 'peak_latency', 'Hemi', 'd', 'd_inf', 'n_trials',
                    'NREM_effectsize', 'NREM_p', 'REM_effectsize', 'REM_p','NREM_Mag_r', 'NREM_Prob_r', 'NREM_AUC_r','NREM_Mag_diff', 'NREM_Prob_diff', 'NREM_AUC_diff','REM_Mag_r', 'REM_Prob_r', 'REM_AUC_r','REM_Mag_diff', 'REM_Prob_diff', 'REM_AUC_diff', 'AF_L', 'AF_R',
                    'C_FPH_L', 'C_FPH_R', 'C_FP_L', 'C_FP_R',
                    'C_PHP_L', 'C_PHP_R', 'C_PH_L', 'C_PH_R', 'C_PO_L', 'C_PO_R', 'EMC_L', 'EMC_R', 'FAT_L', 'FAT_R',
                    'IFOF_L',
                    'IFOF_R', 'ILF_L', 'ILF_R', 'MdLF_L', 'MdLF_R', 'PAT_L',
                    'PAT_R', 'SLF1_L', 'SLF1_R', 'SLF2_L', 'SLF2_R', 'SLF3_L',
                    'SLF3_R', 'VOF_L', 'VOF_R', 'tract_dist', 'tract_num']
        file_save = os.path.join(sub_path, 'EvM', 'Projects', 'EL_experiment', 'Analysis', 'Patients', 'Across',
                                 'Connecto',
                                 'Data', 'con_across_tracts.csv')
    else:
        col_keep = ['Subj', 'Stim', 'Chan', 'LL', 'Sig', 'DI','Sig_diff', 'delay', 'peak_latency', 'Hemi', 'd', 'd_inf', 'n_trials',
                    'NREM_effectsize', 'NREM_p', 'REM_effectsize', 'REM_p']
        file_save = os.path.join(sub_path, 'EvM', 'Projects', 'EL_experiment', 'Analysis', 'Patients', 'Across',
                                 'Connecto',
                                 'Data', 'con_across.csv')

    df_save = data_all[col_keep].reset_index(drop=True)
    df_save.to_csv(file_save, header=True, index=False)
    return df_save

def add_sleep_modulation(subjs):
    # adding sleepdata
    node_all_dfs = []  # A list to hold DataFrames
    folder = 'BrainMapping'
    cond_folder = 'CR'
    for subj in subjs:
        path_patient_analysis = os.path.join(sub_path, 'EvM', 'Projects', 'EL_experiment', 'Analysis', 'Patients',
                                             subj)
        exp_dir = os.path.join(path_patient_analysis, 'BrainMapping', 'CR', 'Graph', 'Connection')
        file = os.path.join(exp_dir, 'con_sleep_stats_LL.csv')

        if os.path.isfile(file):
            df = pd.read_csv(file)
            summary_gen_path = path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\summ_general.csv'  # summary_general

            con_summary = pd.read_csv(summary_gen_path)
            con_summary = con_summary[(con_summary.Sig > 0)].reset_index(drop=True)
            df['Subj'] = subj
            df = df.merge(con_summary[['Stim', 'Chan', 'd_inf', 'delay']], on=['Stim', 'Chan'],
                          how='right').reset_index(
                drop=True)
            node_all_dfs.append(df)  # Add DataFrame to the list
    # Concatenate all DataFrames in the list
    mag_all = pd.concat(node_all_dfs, ignore_index=True)
    mag_all = mag_all[mag_all['Subj'].notna()].reset_index(drop=True)
    # FDR correction
    mag_all['Sig_FDR'] = 0
    for ix_ss, ss_sel in enumerate(['NREM', 'REM']):
        for subj in np.unique(mag_all.Subj):
            req = ~np.isnan(mag_all.p_MW) & (mag_all.SleepState == ss_sel) & (mag_all.Subj == subj)
            mag_all.loc[req, 'Sig_FDR'] = 0
            p_values = mag_all.loc[req, 'p_MW'].values
            p_sig, p_corr = statsmodels.stats.multitest.fdrcorrection(p_values)
            mag_all.loc[req, 'Sig_FDR'] = np.array(p_sig * 1)

            mag_all['Sig_FDR'] = pd.to_numeric(mag_all['Sig_FDR'], errors='coerce')
    keep_col = ['Subj', 'Stim', 'Chan', 'effect_size', 'Sig_FDR', 'SleepState']
    # Split DataFrames
    df_nrem = mag_all[mag_all['SleepState'] == 'NREM'].copy()
    df_nrem = df_nrem[keep_col].reset_index(drop=True)
    df_nrem.rename(columns={'effect_size': 'NREM_effectsize', 'Sig_FDR': 'NREM_p'}, inplace=True)
    df_nrem.drop('SleepState', axis=1, inplace=True)

    df_rem = mag_all[mag_all['SleepState'] == 'REM'].copy()
    df_rem = df_rem[keep_col].reset_index(drop=True)
    df_rem.rename(columns={'effect_size': 'REM_effectsize', 'Sig_FDR': 'REM_p'}, inplace=True)
    df_rem.drop('SleepState', axis=1, inplace=True)

    # Merge DataFrames
    df_merged = pd.merge(df_nrem, df_rem, on=['Subj', 'Stim', 'Chan'])

    ## add Prob_ratio, LL_ratio
    ss = ['Wake', 'NREM', 'REM']
    con_sleep = get_connections_sleep(subjs, sub_path, ss)
    con_sleep_sig = con_sleep.loc[(con_sleep.Mean_sig_LL <50) &(con_sleep.Num_sig_trial >= 5) & ~np.isnan(con_sleep.Mean_sig_LL)].reset_index(
        drop=True)

    for state in ['NREM', 'REM']:
        for metric, metric_label in zip(['Mean_sig_LL', 'Sig'], ['Mag', 'Prob']):
            # Find connections that have both the specified sleep state and Wake values
            common_pairs = con_sleep_sig.groupby(['Subj', 'Stim', 'Chan']).filter(
                lambda x: all(s in x['SleepState'].values for s in [state, 'Wake'])
            )[['Subj', 'Stim', 'Chan']].drop_duplicates()

            data_sleep = pd.merge(common_pairs, con_sleep_sig[con_sleep_sig['SleepState'] == state],
                                  on=['Subj', 'Stim', 'Chan'])
            data_wake = pd.merge(common_pairs, con_sleep_sig[con_sleep_sig['SleepState'] == 'Wake'],
                                 on=['Subj', 'Stim', 'Chan'])

            # Merge sleep and Wake data on Subj, Stim, and Chan to ensure paired samples
            merged_data = pd.merge(data_sleep, data_wake, on=['Subj', 'Stim', 'Chan'], suffixes=('_sleep', '_wake'))

            # Remove samples where the metric is 0 in both sleep and Wake
            merged_data = merged_data[(merged_data[metric + '_sleep'] != 0) | (merged_data[metric + '_wake'] != 0)]

            # Extract the metric values for sleep and Wake
            data_sleep = merged_data[metric + '_sleep'].values
            data_wake = merged_data[metric + '_wake'].values

            # Calculate the mean ratio
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where((data_sleep != 0) | (data_wake != 0),
                                 (data_sleep - data_wake) / np.maximum(data_sleep, data_wake), 0)
                diff = np.where((data_sleep != 0) | (data_wake != 0),
                                 (data_sleep - data_wake), 0)
            merged_data[state + '_' + metric_label + '_r'] = ratio
            merged_data[state + '_' + metric_label + '_diff'] = diff

            # Add to large DataFrame
            df_merged = pd.merge(df_merged, merged_data[['Subj', 'Stim', 'Chan', state + '_' + metric_label + '_r', state + '_' + metric_label + '_diff']],
                                     on=['Subj', 'Stim', 'Chan'], how='outer')

            # Drop duplicates and reset index
            df_merged = df_merged.drop_duplicates().reset_index(drop=True)

    filename = os.path.join(sub_path, 'EvM', 'Projects', 'EL_experiment', 'Analysis', 'Patients', 'Across',
                            'Connecto', 'Data', 'con_across_AUC.csv')

    con_sleep_sig = pd.read_csv(filename)
    for state in ['NREM', 'REM']:
        for metric, metric_label in zip(['AUC'], ['AUC']):
            # Find connections that have both the specified sleep state and Wake values
            common_pairs = con_sleep_sig.groupby(['Subj', 'Stim', 'Chan']).filter(
                lambda x: all(s in x['SleepState'].values for s in [state, 'Wake'])
            )[['Subj', 'Stim', 'Chan']].drop_duplicates()

            data_sleep = pd.merge(common_pairs, con_sleep_sig[con_sleep_sig['SleepState'] == state],
                                  on=['Subj', 'Stim', 'Chan'])
            data_wake = pd.merge(common_pairs, con_sleep_sig[con_sleep_sig['SleepState'] == 'Wake'],
                                 on=['Subj', 'Stim', 'Chan'])

            # Merge sleep and Wake data on Subj, Stim, and Chan to ensure paired samples
            merged_data = pd.merge(data_sleep, data_wake, on=['Subj', 'Stim', 'Chan'], suffixes=('_sleep', '_wake'))

            # Remove samples where the metric is 0 in both sleep and Wake
            merged_data = merged_data[(merged_data[metric + '_sleep'] != 0) | (merged_data[metric + '_wake'] != 0)]

            # Extract the metric values for sleep and Wake
            data_sleep = merged_data[metric + '_sleep'].values
            data_wake = merged_data[metric + '_wake'].values

            # Calculate the mean ratio
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where((data_sleep != 0) | (data_wake != 0),
                                 (data_sleep - data_wake) / np.maximum(data_sleep, data_wake), 0)
                diff = np.where((data_sleep != 0) | (data_wake != 0),
                                 (data_sleep - data_wake), 0)
            merged_data[state + '_' + metric_label + '_r'] = ratio
            merged_data[state + '_' + metric_label + '_diff'] = diff

            # Add to large DataFrame
            df_merged = pd.merge(df_merged, merged_data[
                ['Subj', 'Stim', 'Chan', state + '_' + metric_label + '_r', state + '_' + metric_label + '_diff']],
                                 on=['Subj', 'Stim', 'Chan'], how='outer')

            # Drop duplicates and reset index
            df_merged = df_merged.drop_duplicates().reset_index(drop=True)

    return df_merged

def prepare_data_across(labels, data_con):
    labels.insert(0, 'c_id', -1)
    labels['c_id'] = labels.groupby(['Subj', 'c']).ngroup()

    # Step 1: Merge for sc_id
    data_con.rename(columns={'Stim': 'c'}, inplace=True)
    data_con = data_con.merge(labels[['Subj', 'c', 'c_id']], on=['Subj', 'c'], how='left',
                                        suffixes=('', '_sc'))
    data_con.rename(columns={'c': 'Stim'}, inplace=True)
    # Step 2: Rename c_id to sc_id
    data_con.rename(columns={'c_id': 'sc_id'}, inplace=True)

    # Step 3: Merge for rc_id
    data_con.rename(columns={'Chan': 'c'}, inplace=True)
    data_con = data_con.merge(labels[['Subj', 'c', 'c_id']], on=['Subj', 'c'], how='left',
                                        suffixes=('', '_rc'))
    data_con.rename(columns={'c': 'Chan'}, inplace=True)
    # Step 4: Rename c_id to rc_id
    data_con.rename(columns={'c_id': 'rc_id'}, inplace=True)

    # adding destreiux (_D) and region (R)
    data_con = adding_abbr(data_con, labels)
    data_con = adding_hemisphere(data_con, labels)
    data_con = adding_destrieux(data_con,
                                        area='Destrieux')  # ls.adding_region(data_con, pair = 1, area = 'Destrieux')
    data_con = adding_community(data_con)  # ls.adding_region(data_con, pair=1, area='Region')
    data_con = group_connections(data_con)
    return data_con, labels