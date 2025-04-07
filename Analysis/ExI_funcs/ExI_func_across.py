import os
import numpy as np
import sklearn
import scipy
from sklearn.metrics import auc
import pandas as pd
from ..general_funcs import CCEP_func, LL_funcs as LLf, freq_funcs as ff
import statsmodels


cond_vals = np.arange(4)
SleepStates_val = ['Wake', 'NREM', 'REM']
Fs = 500
dur = np.zeros((1, 2), dtype=np.int32)
t0 = 1
dur[0, 0] = -t0
dur[0, 1] = 3

# dur[0,:]       = np.int32(np.sum(abs(dur)))
x_ax = np.arange(dur[0, 0], dur[0, 1], (1 / Fs))
color_elab = np.zeros((3, 3))
color_elab[0, :] = np.array([31, 78, 121]) / 255
color_elab[1, :] = np.array([189, 215, 238]) / 255
color_elab[2, :] = np.array([0.256, 0.574, 0.431])


def get_AUC_MAX_Pearson(Int_values, LL_values):
    AUC = auc(Int_values, LL_values)
    MAX = np.mean(
        np.sort(LL_values)[-3:])
    rho = scipy.stats.pearsonr(Int_values, LL_values)[0]
    return AUC, MAX, rho


def get_AUC_surr(rc, con_trial, EEG_resp, mx_true, Int_selc, n_trial=200, n=10, w=0.25):
    AUC_surr = np.zeros((n * 3,))
    max_surr = np.zeros((n * 3,))
    p_surr = np.zeros((n * 3,))
    stim_trials = np.unique(
        con_trial.loc[(con_trial.Stim == rc) | (con_trial.Stim == rc - 1) | (con_trial.Stim == rc + 1), 'Num'])
    trials_all = np.unique(con_trial.loc[(con_trial.Chan == rc) & (con_trial.Artefact < 1), 'Num'])
    trials_all = np.array([i for i in trials_all if i not in stim_trials])
    trials_all = trials_all.astype('int')
    Int_norm = (Int_selc - np.min(Int_selc)) / (np.max(Int_selc) - np.min(Int_selc))
    for rep in range(n):
        mx_all = np.zeros((len(Int_selc), 3))
        for i, intensity in enumerate(Int_selc):
            num_sel = np.unique(np.random.choice(trials_all, n_trial, replace=False))
            resp = ff.lp_filter(np.nanmean(EEG_resp[rc, num_sel, :], 0), 45, Fs)
            LL_resp = LLf.get_LL_all(np.expand_dims(resp, [0, 1]), Fs, w)[0][0]
            mx_all[i, 0] = np.max(LL_resp[int((t0 - 0.5) * Fs):int((t0 - w / 2) * Fs)])
            mx_all[i, 1] = np.max(LL_resp[int((w / 2) * Fs):int((t0 - 0.5) * Fs)])
            mx_all[i, 2] = np.max(LL_resp[int((t0 + 2) * Fs):int((t0 + 25) * Fs)])
        for i in range(3):
            normalized_mx_BL = (mx_all[:, i] - np.min(mx_all[:, i])) / (np.max(mx_true) - np.min(mx_all[:, i]))

            AUC_surr[int((i * n) + rep)], max_surr[int((i * n) + rep)], p_surr[
                int((i * n) + rep)] = get_AUC_MAX_Pearson(Int_norm, normalized_mx_BL)

    return AUC_surr, max_surr, p_surr


def get_pvalue(real_array, surr_array):
    count = np.sum(surr_array <= real_array[:, np.newaxis], axis=1)

    # Calculate the p-values
    p_values = np.array(count) / len(surr_array)

    return p_values


def get_AUC_real(sc, rc, con_trial, EEG_resp, get_surr = True, n=100, w=0.25):
    dat = con_trial[(con_trial['Stim'] == sc) & (con_trial['Chan'] == rc) & (con_trial['Artefact'] < 1)].reset_index(
        drop=True)
    Int_selc = np.unique(dat['Int'])
    n_trials = 200
    mx_all = np.zeros((len(Int_selc), 2))
    for i, intensity in enumerate(Int_selc):
        dati = dat[(dat['Int'] == intensity) & (dat['Artefact'] < 1)].reset_index(drop=True)
        resp = ff.lp_filter(np.nanmean(EEG_resp[rc, dati.Num.values.astype('int'), :], 0), 45, Fs)
        LL_resp = LLf.get_LL_all(np.expand_dims(resp, [0, 1]), Fs, w)[0][0]
        mx = np.max(LL_resp[int((t0 + w / 2) * Fs):int((t0 + 0.5 + w / 2) * Fs)])
        mx_all[i, 0] = mx
        n_trials = np.min([n_trials, len(dati.Num.values.astype('int'))])
    mx_all = np.array(mx_all[:, 0])
    mx_norm = (mx_all - np.min(mx_all)) / (np.max(mx_all) - np.min(mx_all))
    Int_norm = (Int_selc - np.min(Int_selc)) / (np.max(Int_selc) - np.min(Int_selc))
    AUC, MAX, rho = get_AUC_MAX_Pearson(Int_norm, mx_norm)
    # get surrogate data
    if get_surr:
        AUC_surr, max_surr, rho_surr = get_AUC_surr(rc, con_trial, EEG_resp, mx_all, Int_selc, n_trial=n_trials, n=n,
                                                    w=0.25)
        AUC_p = get_pvalue(np.array([AUC]), AUC_surr)[0]
        MAX_p = get_pvalue(np.array([MAX]), max_surr)[0]
        rho_p = get_pvalue(np.array([rho]), rho_surr)[0]
    else:
        AUC_p = np.nan
        MAX_p = np.nan
        rho_p = np.nan

    return AUC, MAX, rho, AUC_p, MAX_p, rho_p


def get_AUC_real_mean(sc, rc, con_trial, EEG_resp, ss='Wake', w=0.25):
    # Filter the trials based on conditions
    dat = con_trial[(con_trial['Stim'] == sc) &
                    (con_trial['Chan'] == rc) &
                    (con_trial['Artefact'] == 0) &
                    (con_trial['SleepState'] == ss)].reset_index(drop=True)

    Int_selc = np.unique(dat['Int'])
    mx_all = np.zeros((len(Int_selc), 2))
    n_trials = np.zeros((len(Int_selc), 2))
    for i, intensity in enumerate(Int_selc):
        dati = dat[dat['Int'] == intensity].reset_index(drop=True)

        # Process the EEG response
        resp = ff.lp_filter(np.nanmean(EEG_resp[rc, dati.Num.values.astype('int'), :], axis=0), 45, Fs)
        LL_resp = LLf.get_LL_all(np.expand_dims(resp, axis=[0, 1]), Fs, w)[0][0]

        # Calculate the maximum LL response in the specified time window
        mx = np.max(LL_resp[int((t0 - w / 2) * Fs):int((t0 + 0.5 + w / 2) * Fs)])
        mx_all[i, 0] = mx

        # Store the number of trials for the current intensity
        n_trials[i, 0] = len(dati)

    # Store the results in a DataFrame
    mx_all = np.array(mx_all[:, 0])
    n_trials = np.array(n_trials[:, 0])
    df = pd.DataFrame({
        'Int': Int_selc,
        'LL': mx_all,
        'N_trial': n_trials
    })

    # Add metadata to the DataFrame
    df['Stim'] = sc
    df['Chan'] = rc
    df['SleepState'] = ss

    return df



def save_AUC_connection(con_trial, EEG_resp):
    import matplotlib.pyplot as plt
    stim_all = np.unique(con_trial['Stim'])
    chan_all = np.unique(con_trial['Chan'])
    data_rows = []  # Use for collecting rows of data
    for sc in stim_all.astype('int'):
        for rc in chan_all.astype('int'):
            dat = con_trial[
                (con_trial['Stim'] == sc) & (con_trial['Chan'] == rc) & (con_trial['Artefact'] < 1)].reset_index(
                drop=True)
            if len(dat) > 0:
                AUC, MAX, rho, AUC_p, MAX_p, rho_p = get_AUC_real(sc, rc, con_trial, EEG_resp, n=100, w=0.25)
                # get delay
                num = np.unique(
                    con_trial.loc[(con_trial.Stim == sc) & (con_trial.Chan == rc) & (con_trial.Artefact < 1), 'Num'])
                trials = EEG_resp[rc, num, :]
                WOI = 0.1
                peak_lat, _, _ = CCEP_func.peak_latency(trials, WOI, t0=1, Fs=500, w_LL=0.25)
                # Append each set of results as a new row in the data_rows list
                data_rows.append([sc, rc, AUC, MAX, rho, 1 - AUC_p, 1 - MAX_p, 1 - rho_p])

    # Create the DataFrame after the loop, using the collected rows
    df = pd.DataFrame(data_rows, columns=["Stim", "Chan", "AUC", "MAX", "rho", "AUC_p", "MAX_p", "rho_p"])
    return df


def get_delay(con_trial, EEG_resp, auc_summary, plot=0):
    import matplotlib.pyplot as plt
    Fs = 500
    dur = np.zeros((1, 2), dtype=np.int32)
    t0 = 1
    dur[0, 0] = -t0
    dur[0, 1] = 3

    auc_summary['peak_latency'] = np.nan
    auc_summary['onset'] = np.nan

    auc_summary['sig_con'] = 0
    p_values = auc_summary['AUC_p'].values
    p_sig, p_corr = statsmodels.stats.multitest.fdrcorrection(p_values)
    auc_summary['sig_con'] = np.array(p_sig * 1)

    auc_summary.loc[
        (auc_summary.sig_con == 1) & (auc_summary.MAX_p < 0.05), 'sig_con'] = 1

    df_auc_sig = auc_summary.loc[
        (auc_summary.sig_con ==1)].reset_index(drop=True)
    stim_all = np.unique(df_auc_sig['Stim'])

    for sc in stim_all.astype('int'):
        chan_all = np.unique(df_auc_sig.loc[(df_auc_sig.Stim == sc), 'Chan'])
        for rc in chan_all.astype('int'):
            dat = con_trial[
                (con_trial['Int'] > 2) & (con_trial['Stim'] == sc) & (con_trial['Chan'] == rc) & (
                            con_trial['Artefact'] < 1)].reset_index(
                drop=True)
            if (len(dat) > 0):
                num = np.unique(dat['Num'])
                trials = EEG_resp[rc, num, :]
                WOI = 0.1
                t_onset, peak_lat, polarity, peak_detected = CCEP_func.CCEP_onset(trials, WOI=WOI, t0=1, Fs=500,
                                                                                  w_LL=0.25, plot=False,
                                                                                  skip_nonpeak=True)
                # peak_lat, _, _ = CCEP_func.peak_latency(trials, WOI, t0=1, Fs=500, w_LL=0.25)
                # Append each set of results as a new row in the data_rows list
                if peak_detected:
                    auc_summary.loc[
                        (auc_summary['Stim'] == sc) & (auc_summary['Chan'] == rc), 'peak_latency'] = peak_lat
                    auc_summary.loc[(auc_summary['Stim'] == sc) & (auc_summary['Chan'] == rc), 'onset'] = t_onset
                if plot:
                    plt.plot(x_ax, np.mean(trials, 0))
                    plt.xlim([-0.5, 1])
                    plt.axvline(0, color='k')
                    plt.axvline(peak_lat, color='r')
                    plt.show()
    return auc_summary

def save_AUC_connection_SS_mean(con_trial, EEG_resp,auc_summary):
    stim_all = np.unique(con_trial['Stim'])
    chan_all = np.unique(con_trial['Chan'])
    data_rows = []  # Use for collecting rows of data
    sleep_states = ['Wake', 'NREM', 'REM']
    for sc in stim_all.astype('int'):
        respchans = np.unique(
            auc_summary.loc[(auc_summary.Stim == sc) & (auc_summary.sig_con == 1), 'Chan'])
        # only significant response channels
        for rc in respchans:
            for ss in sleep_states:
                dat = con_trial[
                    (con_trial['SleepState'] == ss) & (con_trial['Stim'] == sc) & (con_trial['Chan'] == rc) & (
                            con_trial['Artefact'] < 1)].reset_index(
                    drop=True)
                if len(dat) > 0:
                    AUC, MAX, rho, _, _, _ = get_AUC_real(sc, rc, con_trial, EEG_resp,get_surr=False, n=40, w=0.25)
                    # Append each set of results as a new row in the data_rows list
                    data_rows.append([sc, rc, ss, AUC, MAX, rho])

    # Create the DataFrame after the loop, using the collected rows
    df = pd.DataFrame(data_rows, columns=["Stim", "Chan", "SleepState", "AUC", "MAX", "rho"])
    return df

def save_AUC_connection_SS(con_trial, EEG_resp,auc_summary):
    # FDR correction
    auc_summary['sig_con'] = 0
    p_values = auc_summary['AUC_p'].values
    p_sig, p_corr = statsmodels.stats.multitest.fdrcorrection(p_values)
    auc_summary['AUC_p_FDR'] = np.array(p_corr)
    auc_summary.loc[
        (auc_summary.AUC_p_FDR <0.05) & (auc_summary.MAX_p < 0.05), 'sig_con'] = 1

    # all stim channels
    stim_all = np.unique(con_trial['Stim'])

    # Initialize an empty DataFrame to store all results
    df_all = pd.DataFrame()

    sleep_states = ['Wake', 'NREM', 'REM']
    for sc in stim_all.astype('int'):
        respchans = np.unique(
            auc_summary.loc[(auc_summary.Stim == sc) & (auc_summary.sig_con == 1), 'Chan'])
        # Only significant response channels
        for rc in respchans:
            for ss in sleep_states:
                dat = con_trial[
                    (con_trial['SleepState'] == ss) & (con_trial['Stim'] == sc) & (con_trial['Chan'] == rc) & (
                                con_trial['Artefact'] < 1)
                    ].reset_index(drop=True)
                if len(dat) > 0:
                    df = get_AUC_real_mean(sc, rc, con_trial, EEG_resp, ss=ss, w=0.25)
                    df_all = pd.concat([df_all, df], ignore_index=True)

                    #AUC, MAX, rho, _, _, _ = get_AUC_real(sc, rc, con_trial, EEG_resp,get_surr=False, n=40, w=0.25)
                    # Append each set of results as a new row in the data_rows list
                    # data_rows.append([sc, rc, ss, AUC, MAX, rho])



    # Create the DataFrame after the loop, using the collected rows
    # df = pd.DataFrame(data_rows, columns=["Stim", "Chan", "SleepState", "AUC", "MAX", "rho"])
    return df_all


def get_AUC_trials(sc, rc, con_trial, EEG_resp, cond_val, n_trial=3, n_shuffle=10, w=0.25):
    data_pd = []
    Int_all = np.unique(con_trial.Int)
    for j in range(len(cond_val)):
        for i in range(len(Int_all)):
            count_run = 0
            Int = Int_all[i]
            dat = con_trial[
                (con_trial.Artefact < 1) & (con_trial.Stim == sc) & (con_trial.Chan == rc) & (con_trial.Int == Int) & (
                        con_trial.SleepState == cond_val[j])]
            stimNum = dat.Num.values.astype('int')
            n_epoch = np.floor(len(stimNum) / n_trial).astype('int')
            for n in range(n_shuffle):
                if n > 0:  # shuffle
                    np.random.shuffle(stimNum)
                for ne in range(n_epoch):
                    num_sel = np.sort(stimNum[int(ne * n_trial):int((ne + 1) * n_trial)])
                    # n_trial_spec = np.min([n_trial, len(stimNum)])
                    # num_sel = np.unique(np.random.choice(stimNum, n_trial_spec, replace = False))
                    mn = ff.lp_filter(np.mean(EEG_resp[rc, num_sel, int(t0 * Fs):int((t0 + 0.5) * Fs)], 0), 45, Fs)
                    LL_mn = LLf.get_LL_all(np.expand_dims(mn, [0, 1]), Fs, w)[0][0]
                    # plt.scatter(Int, np.max(LL_mn), color = color[j])
                    data_pd.append([Int, count_run, SleepStates_val[j], np.max(LL_mn)])
                    count_run += 1
    AUC_SS = pd.DataFrame(data_pd, columns=['Int', 'Run', 'SleepState', 'LL'])
    return AUC_SS


# Define a function to calculate normalized AUC for a group
def calculate_normalized_auc(group, max_overall, max_Int):
    if len(group) < 10:
        # Not enough data points to calculate AUC
        return np.nan
    else:
        # Normalize LL
        LL_norm = (group['LL'] - np.min(group['LL'])) / (max_overall - np.min(group['LL']))
        # Normalize Int
        Int_norm = group['Int'] / max_Int
        # Calculate AUC
        return sklearn.metrics.auc(Int_norm, LL_norm)

