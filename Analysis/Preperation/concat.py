import os
import numpy as np
import ntpath
import pandas as pd
from glob import glob
import h5py

sub_path = 'X:\\4 e-Lab\\'  # y:\\eLab


def concat_resp_condition(subj, folder='InputOutput', cond_folder='CR', EEG=True, skip=True):
    """
    Concatenates epoched EEG response data and stimulation metadata across all blocks
    into a single HDF5 file and a CSV file respectively.

    Parameters:
        subj (str): Subject identifier (e.g., 'EL010').
        folder (str): Main folder of the protocol (e.g., 'BrainMapping', 'InputOutput').
        cond_folder (str): Condition folder, typically 'CR'.
        EEG (bool): Whether to include and concatenate EEG response data.
        skip (bool): Whether to skip if the output file already exists.
    """

    path_patient_analysis = os.path.join(sub_path, 'EvM', 'Projects', 'EL_experiment', 'Analysis', 'Patients', subj)
    data_dir = os.path.join(path_patient_analysis, folder, 'data')
    save_dir = os.path.join(path_patient_analysis, folder, cond_folder, 'data')

    os.makedirs(save_dir, exist_ok=True)

    # Target HDF5 output file
    h5_file = os.path.join(save_dir, f'EEG_{cond_folder}.h5')
    if os.path.isfile(h5_file) and skip:
        print(f'Skipping {subj}: concatenated file already exists.')
        return

    # Locate stimlist files
    stimlist_files = sorted(glob(os.path.join(data_dir, f'Stim_list_*{cond_folder}*')))

    EEG_resp = []
    stimlist = []
    conds = np.empty(len(stimlist_files), dtype=object)

    for i, file in enumerate(stimlist_files):
        base_name = ntpath.basename(file)

        # Extract condition label (e.g. 'CR') from filename
        digits = [idx for idx, ch in enumerate(base_name) if ch.isdigit()]
        cond = base_name[digits[-2] - 2:digits[-2]] if len(digits) >= 2 else 'NA'
        conds[i] = cond

        if EEG:
            block_id = file[-11:-4]  # Assumes file ends with "_<block>.csv"
            eeg_file = os.path.join(data_dir, f'All_resps_{block_id}.npy')
            EEG_block = np.load(eeg_file)

        print(f"{i + 1}/{len(stimlist_files)} -- {base_name}", end='\r')

        # Load stimlist and annotate with condition
        stim_df = pd.read_csv(file)
        stim_df['type'] = cond

        if i == 0:
            stimlist = stim_df
            if EEG:
                EEG_resp = EEG_block
        else:
            stimlist = pd.concat([stimlist, stim_df], ignore_index=True)
            if EEG:
                EEG_resp = np.concatenate([EEG_resp, EEG_block], axis=1)

    # Clean and enrich stimlist
    stimlist = stimlist.drop(columns="StimNum", errors='ignore').fillna(0).reset_index(drop=True)

    if 'us' not in stimlist.columns:
        stimlist.insert(5, 'us', 0)

    try:
        stimlist['Time'] = pd.to_datetime(stimlist['date'].astype(str), format='%Y%m%d') + \
                           pd.to_timedelta(stimlist['h'], unit='H') + \
                           pd.to_timedelta(stimlist['min'], unit='min') + \
                           pd.to_timedelta(stimlist['s'], unit='s') + \
                           pd.to_timedelta(stimlist['us'], unit='us')
    except Exception as e:
        print(f"⚠️ Failed to create 'Time' column: {e}")

    # Drop unnecessary columns
    cols_to_drop = ["StimNum", 'StimNum.1', 'date', 'h', 'min', 's', 'us',
                    'ISI_s', 'TTL', 'TTL_PP', 'TTL_DS', 'TTL_PP_DS', 'currentflow']
    stimlist = stimlist.drop(columns=[col for col in cols_to_drop if col in stimlist.columns])

    # Re-insert StimNum index
    stimlist.insert(0, "StimNum", np.arange(len(stimlist)))

    # Save EEG and stimlist data
    if EEG:
        with h5py.File(h5_file, 'w') as hf:
            hf.create_dataset("EEG_resp", data=EEG_resp)
    stimlist_file = os.path.join(save_dir, f'stimlist_{cond_folder}.csv')
    stimlist.to_csv(stimlist_file, index=False)

    print(f"✅ Data for subject {subj} stored.")
