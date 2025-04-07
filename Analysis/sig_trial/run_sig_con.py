import os
import numpy as np
import sys
import statsmodels

main_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Analysis')
# Add "Analysis" directory to sys.path if it's not already there
if main_path not in sys.path:
    sys.path.insert(0, main_path)

import pandas as pd
from glob import glob
from general_funcs import basic_func as bf
from general_funcs import CCEP_func
import tqdm
import significant_connections as SCF
import significance_funcs
import h5py
from pathlib import Path

base_path  ='/Volumes/vellen/PhD/EL_experiment' #'X:\\4 e-Lab\\' # y:\\eLab
path_analysis = '/Volumes/vellen/PhD/EL_experiment/Analysis' # y:\\eLab # os.path.join(sub_path, 'EvM', 'Projects', 'EL_experiment', 'Analysis')


def rem_artifact_trials(con_trial, EEG_resp, labels_clinic, BM_prot=True):
    # recovery channel
    con_trial.loc[con_trial.Artefact == 2, 'Artefact'] = 0
    con_trial = significance_funcs.recovery_channel_artefact(con_trial, EEG_resp, labels_clinic, Fs=500)
    if BM_prot:
        # Calculate the median LL value for each Stim x Chan combination
        median_values = con_trial.groupby(["Stim", "Chan"])["LL"].median().reset_index()
        median_values.rename(columns={"LL": "Median_LL"}, inplace=True)

        # Merge median values back into the original DataFrame
        con_trial = con_trial.merge(median_values, on=["Stim", "Chan"], how="left")

        # Detect unexpected high LL values and update the 'Art' column
        con_trial.loc[(con_trial["LL"] > 4 * con_trial["Median_LL"])&(con_trial.Artefact == 0), "Artefact"] = 2
        con_trial = con_trial.drop(columns="Median_LL")
    else:
        # Calculate the median LL value for each Stim x Chan x Intensity combination
        median_values = con_trial.groupby(["Stim", "Chan", "Int"])["LL"].median().reset_index()
        median_values.rename(columns={"LL": "Median_LL"}, inplace=True)

        # Merge median values back into the original DataFrame
        con_trial = con_trial.merge(median_values, on=["Stim", "Chan", "Int"], how="left")

        # Detect unexpected high LL values and update the 'Art' column
        con_trial.loc[(con_trial["LL"] > 4 * con_trial["Median_LL"]) & (con_trial.Artefact == 0), "Artefact"] = 2
        con_trial = con_trial.drop(columns="Median_LL")

    #bad baseline
    # Calculate the median LL value for each Stim x Chan combination
    median_values = con_trial.groupby(["Stim", "Chan"])["LL_pre"].median().reset_index()
    median_values.rename(columns={"LL_pre": "Median_LL"}, inplace=True)

    # Merge median values back into the original DataFrame
    con_trial = con_trial.merge(median_values, on=["Stim", "Chan"], how="left")

    # Detect unexpected high LL values and update the 'Art' column
    con_trial.loc[(con_trial["LL_pre"] > 4 * con_trial["Median_LL"])&(con_trial.Artefact == 0), "Artefact"] = 2
    con_trial = con_trial.drop(columns="Median_LL")

    return con_trial


def trial_significance(subj, folder='BrainMapping', cond_folder='CR', fdr=False, p=0.05):
    """
    Update significance levels for trial data based on p-values and optionally apply FDR correction.

    Parameters:
    - subj (str): Subject identifier.
    - base_path (str): Base path to the patient's data directory.
    - folder (str): Protocol folder name for specific analysis (default 'BrainMapping').
    - cond_folder (str): Condition folder name within the top-level folder (default 'CR').
    - fdr (bool): Whether to apply False Discovery Rate (FDR) correction (default False).
    - p (float): Significance level threshold (default 0.05).
    """
    print(subj + ' ---- START ------ ')

    # Construct the file path using os.path.join to ensure correct path formation
    path_patient_analysis  = os.path.join(path_analysis, 'Patients', subj)
    file_con = os.path.join(path_patient_analysis, folder, cond_folder, 'data', 'con_trial_all.csv')

    # Load the control trial data
    con_trial = pd.read_csv(file_con)

    # Remove trials with artefacts
    req = (con_trial['p_value_LL'] >= 0) & (con_trial['Artefact'] < 1)
    valid_p_values = con_trial.loc[req, 'p_value_LL']

    # Apply FDR correction or simple thresholding
    if fdr:
        # p-values need to be less than the threshold after FDR correction to be considered significant
        passed, p_corr = statsmodels.stats.multitest.fdrcorrection(abs(valid_p_values - 1), alpha=p)
        con_trial.loc[req, 'Sig'] = passed.astype(int)
    else:
        # Set significance based on the uncorrected p-value threshold
        con_trial.loc[req, 'Sig'] = (valid_p_values >= 1 - p).astype(int)

    # Ensure the 'Sig' column is numeric
    con_trial['Sig'] = pd.to_numeric(con_trial['Sig'], errors='coerce')
    if "p_value_pre" in con_trial:
        valid_p_values = con_trial.loc[req, 'p_value_pre']

        # Apply FDR correction or simple thresholding
        if fdr:
            # p-values need to be less than the threshold after FDR correction to be considered significant
            passed, p_corr = statsmodels.stats.multitest.fdrcorrection(abs(valid_p_values - 1), alpha=p)
            con_trial.loc[req, 'Sig_pre'] = passed.astype(int)
        else:
            # Set significance based on the uncorrected p-value threshold
            con_trial.loc[req, 'Sig_pre'] = (valid_p_values >= 1 - p).astype(int)

        # Ensure the 'Sig' column is numeric
        con_trial['Sig_pre'] = pd.to_numeric(con_trial['Sig_pre'], errors='coerce')

    # Save updated data back to CSV
    con_trial.to_csv(file_con, header=True, index=False)
    print(subj + ' ---- Sig Value Updated ------ ')


def load_patient_data(subj, base_path, path_patient_analysis, folder, cond_folder):
    """Loads various patient data files required for processing."""
    paths = {
        'gen': os.path.join(base_path, 'Patients', subj),
        'patient': os.path.join(base_path, 'Patients', subj, 'Data', 'EL_experiment'),
        'infos': os.path.join(base_path, 'Patients', subj, 'Electrodes'),
        'con_trial': os.path.join(path_patient_analysis, folder, cond_folder, 'data', 'con_trial_all.csv'),
        'stimlist': os.path.join(path_patient_analysis, folder, cond_folder, 'data', f'stimlist_{cond_folder}.csv')
    }

    # Check if the general path exists, if not use an alternative path
    if not os.path.exists(paths['gen']):
        paths['gen'] = os.path.join('T:', 'EL_experiment', 'Patients', subj)

    # Load data
    stimlist = pd.read_csv(paths['stimlist'])
    lbls = pd.read_excel(os.path.join(paths['infos'], f"{subj}_labels.xlsx"), header=0, sheet_name='BP')
    if "type" in lbls:
        lbls = lbls[lbls.type == 'SEEG']
        lbls = lbls.reset_index(drop=True)

    con_trial = pd.read_csv(paths['con_trial'])

    return stimlist, lbls, con_trial, paths


def start_subj_GT(subj, folder='BrainMapping', cond_folder='CR', cluster_method='kmeans',
                  skip_GT=True, skip_surr=True, skip_summary=True, trial_sig_labeling=True):
    """
    Main function to process EEG data for a given subject. Steps include:
    1. Loading patient data
    2. Removing artifact trials
    3. Calculating ground truth connectivity (GT)
    4. Generating surrogate data for significance testing
    5. Summarizing connectivity data
    6. Labeling single trials with significant connections
    """
    print(f"{subj} ---- START ------ ")

    path_patient_analysis = os.path.join(path_analysis, 'Patients', subj)

    # Load patient and stimulation data
    stimlist, lbls, con_trial, paths = load_patient_data(subj, base_path, path_patient_analysis, folder, cond_folder)
    labels_all, labels_region, labels_clinic, coord_all, StimChans, StimChanSM, StimChansC, StimChanIx, stimlist = bf.get_Stim_chans(stimlist, lbls)

    # Define file paths
    data_path = os.path.join(path_patient_analysis, folder, 'data')
    file_t_resp = os.path.join(data_path, 'M_tresp.npy')
    file_CC_surr = os.path.join(data_path, f'M_CC_surr_{cluster_method}.csv')
    file_CC_LL_surr = os.path.join(data_path, f'LL_CC_surr_{cluster_method}.h5')
    file_GT = os.path.join(data_path, f'M_CC_{cluster_method}.h5')
    file_con = os.path.join(path_patient_analysis, folder, cond_folder, 'data', 'con_trial_all.csv')
    file_CC_summ = os.path.join(data_path, f'CC_summ_{cluster_method}.csv')
    EEG_CR_file = os.path.join(path_patient_analysis, folder, cond_folder, 'data', f'EEG_{cond_folder}.h5')

    # Load and clean con_trial
    con_trial = pd.read_csv(file_con)
    con_trial.loc[con_trial.LL == 0, 'Artefact'] = 1

    EEG_resp = h5py.File(EEG_CR_file)['EEG_resp']
    con_trial = rem_artifact_trials(con_trial, EEG_resp, labels_clinic)
    con_trial.to_csv(file_con, index=False)

    #### Step 1: Ground Truth (GT) connectivity estimation ####
    M_GT_all, CC_LL_surr = [], []
    if os.path.isfile(file_GT) and skip_GT:
        M_t_resp = np.load(file_t_resp)
    else:
        print('Calculating GT connectivity for each pair...')
        chan_all = np.unique(con_trial.Chan)
        n_chan = np.max(chan_all).astype(int) + 1

        M_GT_all = np.zeros((n_chan, n_chan, 3, 2000))
        M_t_resp = np.full((n_chan, n_chan, 6), -1.0)

        for sc in tqdm.tqdm(np.unique(con_trial.Stim)):
            sc = int(sc)
            resp_chans = np.unique(con_trial.loc[(con_trial.Ictal == 0) & (con_trial.Artefact < 1) & (con_trial.Stim == sc), 'Chan']).astype(int)
            for rc in resp_chans:
                M_GT_all[sc, rc, :, :], M_t_resp[sc, rc, :3], M_t_resp[sc, rc, 3:5], M_t_resp[sc, rc, 5], _ = SCF.get_GT(sc, rc, con_trial, EEG_resp)

        with h5py.File(file_GT, 'w') as hf:
            hf.create_dataset("M_GT_all", data=M_GT_all)
        np.save(file_t_resp, M_t_resp)
        print(f"{subj} -- GT connectivity calculation DONE")

    #### Step 2: Surrogate significance calculation ####
    if os.path.isfile(file_CC_surr) and skip_surr:
        M_t_resp = np.load(file_t_resp)
    else:
        print('Calculating surrogate distributions...')
        n_surr = 200
        M_CC_LL_surr = np.zeros((len(labels_all), 6))
        CC_LL_surr = np.zeros((len(labels_all), n_surr, 2, 2000))
        CC_WOI = np.zeros((len(labels_all), n_surr))

        con_trial_n = con_trial[(con_trial.Artefact < 1) & (con_trial.LL > 0)].reset_index(drop=True)
        summ = con_trial_n.groupby(['Stim', 'Chan'], as_index=False)['LL'].count()
        resp_chans = np.unique(con_trial.loc[con_trial.Artefact == 0, 'Chan']).astype(int)

        for rc in tqdm.tqdm(resp_chans):
            n_trials = int(np.median(summ.loc[summ.Chan == rc, 'LL']))
            LL_surr, CC_LL_surr[rc], CC_WOI[rc], LL_mean_surr = SCF.get_CC_surr(rc, con_trial, EEG_resp, n_trials)
            M_CC_LL_surr[rc, :] = [
                np.nanpercentile(LL_surr, q) for q in [50, 95, 99]
            ] + [
                np.nanpercentile(LL_mean_surr, q) for q in [50, 95, 99]
            ]

        with h5py.File(file_CC_LL_surr, 'w') as hf:
            hf.create_dataset("CC_LL_surr", data=CC_LL_surr)
            hf.create_dataset("CC_WOI", data=CC_WOI)

        surr_thr = pd.DataFrame(np.column_stack([np.arange(len(labels_all)), M_CC_LL_surr]),
                                columns=['Chan', 'CC_LL50', 'CC_LL95', 'CC_LL99', 'mean_LL50', 'mean_LL95', 'mean_LL99'])
        surr_thr.to_csv(file_CC_surr, index=False)
        print(f"{subj} -- Surrogate calculation DONE")

    #### Step 3: Connectivity summary ####
    if os.path.isfile(file_CC_summ) and skip_GT and skip_surr and skip_summary:
        CC_summ = pd.read_csv(file_CC_summ)
    else:
        if not M_GT_all:
            M_GT_all = h5py.File(file_GT)['M_GT_all']
        if not CC_LL_surr:
            with h5py.File(file_CC_LL_surr, 'r') as hf:
                CC_LL_surr = hf['CC_LL_surr'][:]
                CC_WOI = hf['CC_WOI'][:]

        CC_summ = SCF.get_CC_summ(M_GT_all, M_t_resp, CC_LL_surr, CC_WOI, coord_all, t_0=1, w=0.25, Fs=500)
        CC_summ.insert(0, 'Subj', subj)
        CC_summ.to_csv(file_CC_summ, index=False)
        print(f"{subj} -- Connectivity summary saved")

    #### Step 4: Label significant trials ####
    if trial_sig_labeling:
        if not M_GT_all:
            M_GT_all = h5py.File(file_GT)['M_GT_all']

        con_trial.drop(columns=[col for col in ['Sig', 'LL_WOI', 'LL_pre'] if col in con_trial], inplace=True)
        con_trial[['Sig', 'LL_WOI', 'LL_pre']] = -1 #will be re-calculated in the next step
        con_trial.drop(columns=[col for col in ['t_N2', 't_N1', 'sN2', 'sN1'] if col in con_trial], inplace=True)

        print('Labeling significant trials...')
        CC_summ['sig'] = statsmodels.stats.multitest.fdrcorrection(abs(CC_summ.p_val - 1))[0].astype(int)

        for sc in tqdm.tqdm(np.unique(con_trial.Stim), desc='Stimulation Channel'):
            sc = int(sc)
            resp_chans = np.unique(con_trial.loc[(con_trial.Artefact < 1) & (con_trial.Stim == sc), 'Chan']).astype(int)
            for rc in resp_chans:
                dat = CC_summ[(CC_summ.Stim == sc) & (CC_summ.Chan == rc) & (CC_summ.sig == 1) & (CC_summ.sig_w == 1)]
                if len(dat) > 0:
                    ix_cc = np.concatenate([[0], dat.CC.values.astype(int)])
                    M_GT = M_GT_all[sc, rc, ix_cc, :]
                    t_WOI = dat.t_WOI.values[0]
                    con_trial = SCF.get_sig_trial(sc, rc, con_trial, M_GT, t_WOI, EEG_resp, test=1, exp=2, w_cluster=0.25)
                else:
                    con_trial = SCF.get_sig_trial(sc, rc, con_trial, EEG_resp, 0, EEG_resp, test=0, exp=2, w_cluster=0.25)
                    con_trial.loc[(con_trial.Chan == rc) & (con_trial.Stim == sc), 'Sig'] = 0

        con_trial.to_csv(file_con, index=False)
        print(f"{subj} -- Trial labeling DONE")
