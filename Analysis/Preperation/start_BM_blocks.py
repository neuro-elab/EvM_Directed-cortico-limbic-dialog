import os
import numpy as np
import h5py
import pandas as pd

from general_funcs import basic_func as bf
from BM_funcs import BM_func as BMf
import platform
from glob import glob
from datetime import datetime, timedelta

import tqdm
from pathlib import Path
import warnings

# I expect to see RuntimeWarnings in this block
warnings.simplefilter("ignore", category=RuntimeWarning)
pd.options.mode.chained_assignment = None  # default='warn'


dist_groups = np.array([[0, 30], [30, 60], [60, 120]])
dist_labels = ['local (<30 mm)', 'short (<60mm)', 'long']
Fs = 500
dur = np.zeros((1, 2), dtype=np.int32)
t0 = 1
dur[0, 0] = -t0
dur[0, 1] = 3

folder = 'BrainMapping'
# dur[0,:]       = np.int32(np.sum(abs(dur)))
x_ax = np.arange(dur[0, 0], dur[0, 1], (1 / Fs))
color_elab = np.zeros((3, 3))
color_elab[0, :] = np.array([31, 78, 121]) / 255
color_elab[1, :] = np.array([189, 215, 238]) / 255
color_elab[2, :] = np.array([0.256, 0.574, 0.431])

sub_path = 'X:\\4 e-Lab\\'  # y:\\eLab


def cal_con_trial(subj, cond_folder='CR', skip_single=True):
    """
        Compute the connectivity trial table (con_trial) for a given subject based on EEG stimulation data.

        For each epoched data block, this function:
        - Loads stimulation lists and corresponding EEG responses.
        - Computes connectivity metrics using a block-wise function.
        - Skips computation if a processed file already exists (unless skip_single=False).
        - Merges all blocks into a single con_trial table.
        - Annotates trials with seizure proximity info.
        - Filters out trials involving bad regions.
        - Saves the final con_trial table as a CSV file.
        """

    print(f'{subj} ---- START ------')

    # Initialize paths
    paths = get_patient_paths(subj)
    files_list = glob(os.path.join(paths["analysis"], f'*\\data\\Stim_list_*{cond_folder}*'))

    if not files_list:
        print("No stimulation files found.")
        return

    # Load electrode labels and define bad channels/regions
    labels, bad_chans, bad_region, coord_all, stim_data = load_labels_and_stim_info(paths, subj, files_list)

    con_trial_all = []
    stim_num_offset = 0

    for file_path in files_list:
        print(f'Loading {file_path[-11:-4]}', end='\r')
        stimlist = load_and_clean_stimlist(file_path)

        block_id = file_path[-11:-4]
        trial_file = os.path.join(paths["analysis"], cond_folder, 'data', f'con_trial_{block_id}.csv')

        if os.path.isfile(trial_file) and skip_single:
            con_trial_block = pd.read_csv(trial_file)
            if "Time" not in con_trial_block:
                con_trial_block = con_trial_block.merge(stimlist[['Num_block', 'Time']], on='Num_block')
        else:
            eeg_file = os.path.join(paths["analysis"], 'data', f'ALL_resps_{block_id}.npy')
            EEG_resp = np.load(eeg_file)

            if EEG_resp.shape[1] != len(stimlist):
                print('ERROR: Stimulation count mismatch.')
                break

            con_trial_block = BMf.LL_BM_connection(
                EEG_resp, stimlist, bad_chans, coord_all,
                labels["clinic"], stim_data["StimChanSM"], stim_data["StimChanIx"]
            )

            if cond_folder == 'CR':
                con_trial_block = con_trial_block.drop(columns=['Condition'], errors='ignore')
            else:
                con_trial_block = con_trial_block.drop(columns=['Sleep'], errors='ignore')

            con_trial_block = con_trial_block.merge(stimlist[['Num_block', 'Time']], on='Num_block')
            con_trial_block = con_trial_block.astype({'Chan': int, 'Stim': int, 'Num': int, 'Num_block': int})
            con_trial_block.to_csv(trial_file, index=False)

        con_trial_block['Num'] = con_trial_block['Num_block'] + stim_num_offset
        stim_num_offset += con_trial_block['Num_block'].max() + 1

        con_trial_all.append(con_trial_block)

    # Concatenate all blocks
    con_trial = pd.concat(con_trial_all, ignore_index=True)

    # Add seizure annotations
    con_trial = sz_time(subj, con_trial)

    # Remove trials with stim or rec in bad regions
    con_trial = con_trial[~np.isin(con_trial.Chan, bad_region)]
    con_trial = con_trial[~np.isin(con_trial.Stim, bad_region)]

    # Save final con_trial table
    final_file = os.path.join(paths["analysis"], cond_folder, 'data', 'con_trial_all.csv')
    con_trial.to_csv(final_file, index=False)

    print(f'{subj} ---- DONE ------')


# ----------------- Helper Functions -----------------

def get_patient_paths(subj):
    base_path = os.path.join(sub_path, 'EvM', 'Projects', 'EL_experiment', 'Analysis', 'Patients', subj)
    raw_data_path = os.path.join(sub_path, 'Patients', subj)
    if not os.path.exists(raw_data_path):
        raw_data_path = os.path.join('T:\\EL_experiment\\Patients', subj)

    info_path = os.path.join(raw_data_path, 'Electrodes')
    if not os.path.exists(os.path.join(info_path, f"{subj}_labels.xlsx")):
        info_path = os.path.join(raw_data_path, 'infos')

    return {
        "analysis": base_path,
        "raw_data": raw_data_path,
        "experiment": os.path.join(raw_data_path, 'Data', 'EL_experiment'),
        "info": info_path
    }


def load_labels_and_stim_info(paths, subj, files_list):
    label_file = os.path.join(paths["info"], f"{subj}_labels.xlsx")
    labels = pd.read_excel(label_file, sheet_name='BP')
    if "type" in labels:
        labels = labels[labels.type == 'SEEG'].reset_index(drop=True)

    stimlist = pd.read_csv(files_list[-1])  # Use last stim file
    labels_all, labels_region, labels_clinic, coords, stim_chans, stim_sm, stim_clin, stim_ix, stimlist = bf.get_Stim_chans(stimlist, labels)

    badchans = pd.read_csv(os.path.join(paths["analysis"], 'BrainMapping', 'data', 'badchan.csv'))
    bad_chans = np.unique(np.where(badchans.values[:, 1:] == 1)[0])

    bad_region = np.where(np.isin(labels_region, ['WM', 'OUT', 'Putamen']))[0]

    return (
        {"all": labels_all, "clinic": labels_clinic},
        bad_chans,
        bad_region,
        coords,
        {"StimChanSM": stim_sm, "StimChanIx": stim_ix}
    )


def load_and_clean_stimlist(file_path):
    stimlist = pd.read_csv(file_path)

    # Remove stimulations with zero channel values
    stimlist = stimlist[(stimlist.ChanP > 0) & (stimlist.ChanN > 0)].reset_index(drop=True)

    # Ensure 'noise' column exists
    if 'noise' not in stimlist:
        stimlist.insert(9, 'noise', 0)

    # Create datetime timestamp
    stimlist['Time'] = pd.to_datetime(stimlist['date'].astype(str), format='%Y%m%d') + \
                       pd.to_timedelta(stimlist['h'], unit='h') + \
                       pd.to_timedelta(stimlist['min'], unit='m') + \
                       pd.to_timedelta(stimlist['s'], unit='s')

    # Recreate StimNum and Num_block
    stimlist['StimNum'] = np.arange(len(stimlist))
    stimlist['Num_block'] = stimlist['StimNum']

    return stimlist


def sz_time(subj, con_trial):
    """
        Annotates the trial DataFrame with seizure proximity labels.

        Adds a new column 'Ictal' to `con_trial`:
            -  1 if within 10 minutes after a seizure (post-ictal)
            - -1 if within 10 minutes before a seizure (pre-ictal)
            -  0 otherwise

        Parameters:
            subj (str): Subject identifier
            con_trial (pd.DataFrame): DataFrame containing a 'Time' column (datetime)

        Returns:
            pd.DataFrame: Updated con_trial with 'Ictal' column
        """

    con_trial.insert(5, 'Ictal', 0)

    file = os.path.join(sub_path, 'Patients', subj, 'Data', 'EL_experiment', 'SZ_log.xlsx')

    if not os.path.isfile(file):
        return con_trial

    try:
        sz_log = pd.read_excel(file)
        sz_log = sz_log[~sz_log['SZ'].isna()].reset_index(drop=True)

        trial_times = pd.to_datetime(con_trial['Time'])

        for _, row in sz_log.iterrows():
            sz_datetime = datetime.combine(row['Date'].to_pydatetime().date(), row['Time'])
            delta = trial_times - sz_datetime

            # Mark post-ictal (0 to +10 min)
            con_trial.loc[delta.between(timedelta(seconds=0), timedelta(minutes=10)), 'Ictal'] = 1

            # Mark pre-ictal (-10 min to 0)
            con_trial.loc[delta.between(timedelta(minutes=-10), timedelta(seconds=0)), 'Ictal'] = -1

    except Exception as e:
        print(f"Error processing seizure log for {subj}: {e}")

    return con_trial
