# Standard library imports
import os
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance

# Custom module imports from sibling subfolders
from ..general_funcs import basic_func as bf, freq_funcs as ff, LL_funcs as LLf, CCEP_func as CCEPf
from ..sig_trial import Cluster_func as Cf, significance_funcs as sf
sub_path = 'X:\\4 e-Lab\\'  # y:\\eLab


def get_t_WOI(resp_mean, t_0=1, Fs=500, w=0.25):
    """
    Estimate the window of interest (WOI) time based on the peak of the line-length (LL) transform.

    Parameters:
        resp_mean (np.ndarray): 1D array of the mean response signal.
        t_0 (float): Start time (in seconds) for WOI search.
        Fs (int): Sampling frequency in Hz.
        w (float): Window length (in seconds) for LL calculation.

    Returns:
        float: Time (in seconds) of the peak LL value within the WOI.
    """
    # Compute LL transform for the response signal (assuming LLf.get_LL_all returns an array)
    LL_mean = LLf.get_LL_all(resp_mean[None, None, :], Fs, w)[0, 0]

    # Calculate indices for the window of interest (WOI)
    start_idx = int((t_0 + w / 2) * Fs)
    end_idx = int((t_0 + 0.5 - w / 2) * Fs)

    # Ensure that the end index does not exceed the length of the signal
    end_idx = min(end_idx, len(LL_mean))

    # Find peak within WOI using argmax, directly on the sliced array
    peak_relative_idx = np.argmax(LL_mean[start_idx:end_idx])
    peak_time = (start_idx + peak_relative_idx) / Fs

    return peak_time


def get_GT_trials(trials, t_0=1, Fs=500, w=0.25, n_cluster=2, cluster_type='Kmeans', t_WOI=-1):
    """
        This function processes EEG trials and clusters them based on the time-window of interest (WOI).
        It calculates several features such as the mean line length (LL), clustering results, and correlation
        between clusters. The function also computes the correlation coefficient between two clusters.

        Parameters:
            trials (np.ndarray): EEG data in the form of trials (shape: n_trials x n_samples).
            t_0 (float): The time of stimulation onset (in seconds) in the epoched data.
            Fs (int): Sampling frequency in Hz (default is 500 Hz).
            w (float): The window length for analysis in seconds (default is 0.25 s).
            n_cluster (int): Number of clusters for trial grouping (default is 2).
            cluster_type (str): Type of clustering to perform ('Kmeans' or 'similarity').
            t_WOI (float): Window of interest (WOI) time (default is -1, meaning it will be calculated).

        Returns:
            list: Contains various outputs, including:
                - r, t_onset, t_WOI: Parameters related to clustering (not computed here).
                - LL_mean: Mean line length transform.
                - p_CC: Pearson correlation coefficient between clusters.
                - M_GT: Mean data for all trials and for specific clusters.
                - y: Cluster assignments for each trial.
                - thr: Threshold (not computed here).
                - LL_CC: Line length values for the specific clusters.
                - LL_mean_max: Maximum line length value within the mean window.
        """
    # 1. Filter the trials using a band-pass filter (1-45 Hz).
    EEG_trial = ff.bp_filter(trials, 1, 45, Fs)

    # 2. Compute the mean response across all trials.
    resp_mean = np.nanmean(EEG_trial, 0)

    # 3. Compute the line length (LL) transform for the mean response.
    LL_mean = LLf.get_LL_all(np.expand_dims(np.expand_dims(resp_mean, 0), 0), Fs, w)

    # Initialize dummy parameters for backward compatibility with older versions.
    r = -1
    t_onset = -1
    thr = -1

    # 4. Calculate the window of interest (WOI) based on LL peak if not provided.
    if t_WOI == -1:
        t_WOI = np.argmax(LL_mean[0, 0, int((t_0 + w / 2) * Fs):int((t_0 + 0.5 - w / 2) * Fs)]) / Fs

    # 5. Cluster the trials based on the window of interest (WOI) and line length.
    if cluster_type == 'Kmeans':
        # Z-score the trials before clustering.
        EEG_trial = bf.zscore_CCEP(EEG_trial, t_0, Fs)
        # Perform KMeans clustering on the trials within the WOI.
        cc, y = Cf.ts_cluster(
            EEG_trial[:, int((t_0 + t_WOI) * Fs):int((t_0 + t_WOI + w) * Fs)], n=n_cluster, method='euclidean')
    else:  # For other similarity-based clustering methods.
        cc, y = Cf.ts_cluster(
            EEG_trial[:, int((t_0 + t_WOI) * Fs):int((t_0 + t_WOI + w) * Fs)], n=n_cluster, method='similarity')

    # 6. Store the mean across all trials and the mean for each specific cluster.
    M_GT = np.zeros((3, trials.shape[-1]))  # Assuming n_cluster = 2, hence 3 rows for the means.
    M_GT[0, :] = np.nanmean(EEG_trial, 0)  # Mean for all trials.

    # Mean for each cluster.
    for c in range(n_cluster):
        M_GT[c + 1, :] = np.nanmean(EEG_trial[y == c, :], 0)

    # 7. Calculate the Pearson correlation between the two clusters.
    p_CC = np.corrcoef(
        M_GT[1, int((t_0 + t_WOI) * Fs):int((t_0 + t_WOI + 2 * w) * Fs)],
        M_GT[2, int((t_0 + t_WOI) * Fs):int((t_0 + t_WOI + 2 * w) * Fs)])[0, 1]

    # 8. Compute the line length (LL) transform for the clusters.
    LL_CC = LLf.get_LL_all(np.expand_dims(M_GT[1:, int((t_0 + t_WOI) * Fs):int((t_0 + t_WOI + w) * Fs)], 0), Fs, w)
    LL_CC = np.max(LL_CC[0, :, :], 1)

    # 9. Get the maximum LL value from the mean response.
    LL_mean_max = np.max(LL_mean[0, 0, int((t_0 + w / 2) * Fs):int((t_0 + 0.5 - w / 2) * Fs)])
    return [r, t_onset, t_WOI], LL_mean[0, 0], p_CC, M_GT, y, thr, LL_CC, LL_mean_max


def get_GT(sc, rc, LL_CCEP, EEG_resp, Fs=500, t_0=1, w_cluster=0.25, n_cluster=2, int_sel=-1, t_WOI=-1):
    """
        Calculate ground truths for a specific connection (stimulus channel `sc` and recording channel `rc`).

        Parameters:
        sc (int): Stimulus channel.
        rc (int): Recording channel.
        LL_CCEP (DataFrame): DataFrame containing information about the CCEP data.
        EEG_resp (array): EEG response data with shape (channels, trials, timepoints).
        Fs (int): Sampling frequency in Hz (default is 500).
        t_0 (float): Time point of stimulation in seconds within epoched data (default is 1).
        w_cluster (float): Window size (in seconds) for clustering (default is 0.25).
        n_cluster (int): Number of clusters for trial grouping (default is 2).
        int_sel (int): Intensity selection for brain mapping protocol (default is -1 for all intensities).
        t_WOI (float): Time window of interest (default is -1 to automatically compute based on peak LL transform).

        Returns:
        tuple:
            - M_GT (array): Ground truth matrix of shape (n_cluster + 1, 2000).
            - [r, t_onset, t_WOI] (list): Stimulus-related metrics.
            - LL_CC (array): Local likelihood (LL) of the clustered trials.
            - LL_mean_max (float): Maximum value of LL across all trials.
            - corr_CC (float): Pearson correlation between the two clustered trial means.
        """

    # Filter data based on intensity selection (if specified) and conditions (e.g., non-ictal, non-artifact)
    if int_sel > -1:
        lists = LL_CCEP[
            (LL_CCEP['Int'] == int_sel) & (LL_CCEP['Ictal'] == 0) & (LL_CCEP['Artefact'] < 1) & (
                    LL_CCEP['Chan'] == rc) & (
                    LL_CCEP['Stim'] == sc)].reset_index(drop=True)
    else:
        lists = LL_CCEP[(LL_CCEP['Ictal'] == 0) & (LL_CCEP['Artefact'] < 1) & (
                LL_CCEP['Chan'] == rc) & (
                                LL_CCEP['Stim'] == sc)].reset_index(drop=True)

    stimNum_all = lists.Num.values.astype('int')
    M_GT = np.zeros((n_cluster + 1, 2000))
    r = -1
    t_onset = -1
    corr_CC = -1
    LL_mean_max = -1
    LL_CC = np.zeros((1, 2))

    # Only proceed if there are valid trials
    if len(stimNum_all) > 0:
        # Remove trials with NaN values
        t_nan = np.where(np.isnan(np.mean(EEG_resp[rc, stimNum_all, :], axis=1)))[0]
        if len(t_nan) > 0:
            stimNum_all = np.delete(stimNum_all, t_nan)  # Remove NaN trials

        # Proceed with trial processing if enough valid trials remain
        if len(stimNum_all) > 15:
            trials = EEG_resp[rc, stimNum_all, :]
            # Call the function to get ground truths for the trials
            [r, t_onset, t_WOI], _, corr_CC, M_GT, y, _, LL_CC, LL_mean_max = get_GT_trials(
                trials, t_0=t_0, Fs=Fs, w=w_cluster, n_cluster=n_cluster, cluster_type='similarity', t_WOI=t_WOI
            )
    return M_GT, [r, t_onset, t_WOI], LL_CC, LL_mean_max, corr_CC


def get_CC_surr(rc, LL_CCEP, EEG_resp, n_trials, Fs=500, w_cluster=0.25, n_cluster=2, n_surr=200):
    """
        Generate surrogate trials by selecting non-stimulated trials and calculate cluster centers CC.
        Surrogate trials are based on dummy stimulation onset after true stimulation onset.

        Parameters:
        rc (int): Recording channel.
        LL_CCEP (DataFrame): DataFrame containing the CCEP data.
        EEG_resp (array): EEG response data of shape (channels, trials, timepoints).
        n_trials (int): Number of trials to use in each surrogate iteration.
        Fs (int): Sampling frequency in Hz (default is 500).
        w_cluster (float): Window size (in seconds) for clustering (default is 0.25).
        n_cluster (int): Number of clusters for trial grouping (default is 2).
        n_surr (int): Number of surrogate iterations (default is 200).

        Returns:
        tuple:
            - LL_CC_surr (array): Surrogate trial LL values.
            - LL_surr_data (array): Ground truth data from surrogate trials.
            - WOI_surr (array): Window of interest for surrogate trials.
            - LL_mean_surr (array): Mean LL value for each surrogate iteration.
        """

    # Define surrogate trials (those where the stimulation channel is not stimulating)
    stim_trials = np.unique(
        LL_CCEP.loc[(LL_CCEP.Stim >= rc - 1) & (LL_CCEP.Stim <= rc + 1), 'Num'].values.astype('int'))

    # Stimulus numbers where no stimulation occurred in the surrogate trials
    StimNum = np.unique(LL_CCEP.loc[(LL_CCEP.d > 15) & (LL_CCEP['Ictal'] == 0) & (LL_CCEP.Artefact == 0) &
                                    (LL_CCEP.Chan == rc), 'Num'])

    # Remove trials that are nearby the stimulation trials (stim_trials)
    StimNum = [i for i in StimNum if i not in stim_trials and i not in stim_trials + 1 and i not in stim_trials - 1]

    # Initialize variables for surrogate results
    LL_mean_surr = np.zeros((n_surr, 1))  # Stores the mean LL values for surrogate trials
    LL_CC_surr = np.zeros((n_surr, 2))  # Stores the LL values for surrogate trial clusters
    WOI_surr = np.zeros((n_surr,))  # Stores the window of interest for surrogate trials
    LL_surr_data = np.zeros((n_surr, 2, 2000))  # Stores ground truth data for surrogate trials

    # Run the surrogate trial iterations
    for i in range(n_surr):
        # Randomly select surrogate trials
        StimNum_surr = np.unique(np.random.choice(StimNum, size=n_trials).astype('int'))

        # Remove trials containing NaN values
        t_nan = np.where(np.isnan(np.mean(EEG_resp[rc, StimNum_surr, :], axis=1)))[0]
        if len(t_nan) > 0:
            StimNum_surr = np.delete(StimNum_surr, t_nan)  # Remove NaN trials

        # Get EEG trials for the selected surrogate stimulus numbers
        trials = EEG_resp[rc, StimNum_surr, :]

        for t_0_surr in [3.2]:  # Timepoint for surrogate (do not change this)
            # Get clustering results for the surrogate trials
            [_, _, t_WOI], _, p_CC, M_GT, y, _, LL_CC, LL_mean_max = get_GT_trials(
                trials, t_0=t_0_surr, Fs=Fs, w=w_cluster, n_cluster=n_cluster, cluster_type='similarity')

            # Store the results of this surrogate iteration
            LL_mean_surr[i] = LL_mean_max
            LL_CC_surr[i] = LL_CC
            WOI_surr[i] = t_WOI
            LL_surr_data[i] = M_GT[1:, :]  # Store the mean data for the clusters

    # Return the results for all surrogate trials
    return LL_CC_surr, LL_surr_data, WOI_surr, LL_mean_surr


def get_corr_CC(signal, t0_surr, t_WOI_surr, w=0.25, Fs=500):
    # Array to store correlation coefficients
    correlations = np.zeros(signal.shape[0])

    # Compute the correlation for each pair
    for i in range(signal.shape[0]):
        start_idx = int((t0_surr + t_WOI_surr[i]) * Fs)
        end_idx = int((t0_surr + t_WOI_surr[i] + w) * Fs)

        # Extract slices for the correlation computation
        data1 = signal[i, 0, start_idx:end_idx]
        data2 = signal[i, 1, start_idx:end_idx]

        # Check if the window is within the bounds of the array
        if end_idx <= signal.shape[2] and start_idx < end_idx:
            # Compute correlation coefficient
            correlation_matrix = np.corrcoef(data1, data2)
            correlations[i] = correlation_matrix[0, 1]
        else:
            # Handle cases where the window is out of bounds
            correlations[i] = np.nan  # Assign NaN or handle as appropriate

    # Now `correlations` contains the correlation coefficient for each pair
    return correlations


def get_CC_summ(M_GT_all, M_t_resp, CC_LL_surr, WOI_surr, coord_all, t_0=1, w=0.25, w_LL_onset=0.1, smooth_win=0.1,
                Fs=500):
    # creates a table for each stim-chan pair with the two CC found indicating the WOI, LL and whether it's signficant
    start = 1
    t0_surr = 3.2
    for rc in range(M_GT_all.shape[0]):
        # get correlation CC1, CC2
        if ~np.isnan(np.max(WOI_surr[rc])):
            t_WOI_surr = WOI_surr[rc]
            corr_surr = get_corr_CC(CC_LL_surr[rc], t0_surr, WOI_surr[rc], w=w, Fs=Fs)
            corr_thr = np.percentile(corr_surr, 95)

            LL_resp_chan = LLf.get_LL_all(CC_LL_surr[rc], win=w, Fs=Fs)
            LL_resp_chan = LL_resp_chan.reshape(-1, LL_resp_chan.shape[2])
            LL_surr = np.max(LL_resp_chan[:, int((t0_surr + w / 2) * Fs):int((t0_surr + 0.5 - w / 2) * Fs)], 1)
            for sc in range(M_GT_all.shape[0]):
                d = np.round(distance.euclidean(coord_all[sc], coord_all[rc]), 2)
                LL_CC = LLf.get_LL_all(np.expand_dims(M_GT_all[sc, rc, :], 0), Fs=Fs, win=0.25)[0]
                WOI = M_t_resp[sc, rc, 2]
                if M_t_resp[sc, rc, 0] > -1:
                    corr_real = get_corr_CC(np.expand_dims(M_GT_all[sc, rc, 1:], 0), t_0, np.expand_dims(WOI, [0, 1]),
                                            w=w,
                                            Fs=Fs)
                    for i in range(1, 3):
                        LL_peak = np.max(LL_CC[i, int((t_0 + w / 2) * Fs):int((t_0 + 0.5 - w / 2) * Fs)])
                        LL_WOI = LL_CC[i, int((t_0 + WOI + w / 2) * Fs)]
                        p_value = get_pvalue_trial(np.expand_dims(LL_peak, 0), LL_surr)[
                            0]  # sig = np.array(LL_WOI > thr) * 1
                        thr_BL = np.percentile(LL_CC[i, int(t0_surr * Fs):int((4 - w / 2) * Fs)], 95)
                        LL_t_pk = np.array(LL_CC[i] >= thr_BL) * 1
                        LL_t_pk[:int((t_0 + w / 2) * Fs)] = 0
                        LL_t_pk[int((t_0 + 0.5 - w / 2) * Fs):] = 0
                        t_pk = sf.search_sequence_numpy(LL_t_pk, np.ones((int(0.125 * Fs),)))
                        sig_w = np.array((len(t_pk) > 0)) * 1

                        arr = np.array(
                            [[sc, rc, i, LL_WOI, WOI, LL_peak, p_value, sig_w, d, corr_real[0],
                              corr_real[0] > corr_thr]])
                        arr = pd.DataFrame(arr, columns=['Stim', 'Chan', 'CC', 'LL_WOI', 't_WOI', 'LL_pk', 'p_val',
                                                         'sig_w', 'd', 'corr_CC', 'corr_CC_sig'])
                        if start:
                            CC_summ = arr
                            start = 0
                        else:
                            CC_summ = pd.concat([CC_summ, arr])
                            CC_summ = CC_summ.reset_index(drop=True)
    return CC_summ


def zscore_peak(data, t0=1, Fs=500, thr=2.5):
    from scipy.signal import find_peaks
    sig = 0
    # Calculate the indices for the time window 0 to 0.5 seconds post trigger
    start_index = int(t0 * Fs)
    end_index = int((t0 + 0.5) * Fs)

    # Extract the segment of interest
    segment = data[start_index:end_index]

    # Find positive and negative peaks
    positive_peaks, _ = find_peaks(segment)
    negative_peaks, _ = find_peaks(-segment)
    # Check if any peaks exceed the thresholds
    positive_peak_exceeds_threshold = np.any(segment[positive_peaks] > thr)
    negative_peak_exceeds_threshold = np.any(segment[negative_peaks] < -thr)
    if positive_peak_exceeds_threshold:
        sig = 1
    elif negative_peak_exceeds_threshold:
        sig = 1

    return sig


def get_sig_trial(sc, rc, con_trial, M_GT, t_resp, EEG_CR, test=1, exp=2, w_cluster=0.25, t_0=1, t_0_BL=0.5,
                  Fs=500, int_sel=-1):
    if int_sel > -1:
        req = (con_trial.Stim == sc) & (con_trial.Chan == rc) & (con_trial.Artefact < 1) & (con_trial.Ictal == 0) & (
                con_trial.Int == int_sel)
    else:
        req = (con_trial.Stim == sc) & (con_trial.Chan == rc) & (con_trial.Artefact < 1) & (con_trial.Ictal == 0)
    dat = con_trial[req].reset_index(drop=True)
    dt = 10
    EEG_trials = ff.bp_filter(np.expand_dims(EEG_CR[rc, dat.Num.values.astype('int'), :], 0), 1, 45, Fs)
    LL_trials = LLf.get_LL_all(EEG_trials, Fs, w_cluster)
    if test:
        # for each trial get significance level based on surrogate (Pearson^2 * LL)
        #### first get surrogate data
        pear_surr_all = []
        # pear_surr_all_P2P = []
        for t_test in [2.7, 3.2]:  # surrogates times, todo: in future blockwise
            pear = np.zeros((len(EEG_trials[0]),)) - 1  # pearson to each CC
            lags = np.zeros((len(EEG_trials[0]),)) + t_test * Fs
            for n_c in range(len(M_GT)):
                pear_run, lag_run = sf.get_shifted_pearson_correlation(M_GT[n_c, :], EEG_trials[0], tx=t_0 + t_resp,
                                                                       ty=t_test,
                                                                       win=w_cluster,
                                                                       Fs=500, max_shift_ms=dt)
                lags[pear_run > pear] = lag_run[pear_run > pear]
                pear = np.max(
                    [pear, pear_run], 0)

            lags = lags + Fs * (w_cluster / 2)
            LL = [LL_trials[0, i, lags.astype('int')[i]] for i in range(len(lags.astype('int')))]
            pear_surr = np.sign(pear) * abs(pear ** exp) * LL
            pear_surr_all = np.concatenate([pear_surr_all, pear_surr])

        # other surr trials
        real_trials = np.unique(
            con_trial.loc[req, 'Num'].values.astype('int'))
        # trials where RC is stimulating
        stim_trials = np.unique(
            con_trial.loc[(con_trial.Stim >= rc - 1) & (con_trial.Stim <= rc + 1), 'Num'].values.astype('int'))
        StimNum = np.random.choice(np.unique(con_trial.loc[~np.isin(con_trial.Stim, [sc - 1, sc, sc + 1]) & ~np.isin(
            con_trial.Stim, [rc - 1, rc, rc + 1]) & (con_trial.Artefact == 0), 'Num']), size=200)
        StimNum = [i for i in StimNum if i not in stim_trials]
        StimNum = [i for i in StimNum if i not in stim_trials + 1]
        StimNum = [i for i in StimNum if i not in real_trials]

        StimNum = np.unique(StimNum).astype('int')
        EEG_surr = ff.bp_filter(np.expand_dims(EEG_CR[rc, StimNum, :], 0), 1, 45, Fs)
        bad_StimNum = np.where(np.max(abs(EEG_surr[0]), 1) > 1000)
        if (len(bad_StimNum[0]) > 0):
            StimNum = np.delete(StimNum, bad_StimNum)
            EEG_surr = ff.bp_filter(np.expand_dims(EEG_CR[rc, StimNum, :], 0), 1, 45, Fs)
        if len(StimNum) > 0:
            LL_surr = LLf.get_LL_all(EEG_surr, Fs, w_cluster)
            for t_test in [2.5, 3]:  # surrogates times, todo: in future blockwise
                pear = np.zeros((len(EEG_surr[0]),)) - 1
                lags = np.zeros((len(EEG_surr[0]),)) + t_test * Fs
                for n_c in range(len(M_GT)):
                    pear_run, lag_run = sf.get_shifted_pearson_correlation(M_GT[n_c, :], EEG_surr[0], tx=t_0 + t_resp,
                                                                           ty=t_test,
                                                                           win=w_cluster,
                                                                           Fs=500, max_shift_ms=dt)
                    lags[pear_run > pear] = lag_run[pear_run > pear]
                    pear = np.max(
                        [pear, pear_run], 0)

                lags = lags + Fs * (w_cluster / 2)
                LL = [LL_surr[0, i, lags.astype('int')[i]] for i in
                      range(len(lags.astype('int')))]
                pear_surr = np.sign(pear) * abs(pear ** exp) * LL
                pear_surr_all = np.concatenate([pear_surr_all, pear_surr])

        ##### real trials
        t_test = t_0 + t_resp  # timepoint that i tested is identical to WOI
        pear = np.zeros((len(EEG_trials[0]),)) - 1
        lags = np.zeros((len(EEG_trials[0]),)) + t_test * Fs
        for n_c in range(len(M_GT)):
            pear_run, lag_run = sf.get_shifted_pearson_correlation(M_GT[n_c, :], EEG_trials[0], tx=t_0 + t_resp,
                                                                   ty=t_test,
                                                                   win=w_cluster,
                                                                   Fs=500, max_shift_ms=dt)
            lags[pear_run > pear] = lag_run[pear_run > pear]
            pear = np.max(
                [pear, pear_run], 0)
        lags = lags + Fs * (w_cluster / 2)
        LL = [LL_trials[0, i, lags.astype('int')[i]] for i in
              range(len(lags.astype('int')))]  # LL = LL_trials[0, :, lags.astype('int')]
        compound_metric = np.sign(pear) * abs(pear ** exp) * LL
        pv = get_pvalue_trial(compound_metric, pear_surr_all)
        con_trial.loc[
            req, 'p_value_LL'] = pv

    ##### real trials, LL pre stim
    t_test = t_0_BL + t_resp  #

    LL_pre = LL_trials[0, :, int((t_test + w_cluster / 2) * Fs)]
    con_trial.loc[req, 'LL_pre'] = LL_pre
    return con_trial


def get_pvalue_trial(real_array, surr_array):
    count = np.sum(surr_array <= real_array[:, np.newaxis], axis=1)

    # Calculate the p-values
    p_values = np.array(count) / len(surr_array)

    return p_values


def get_sig_trial_surr(sc, rc, con_trial, M_GT, t_resp, EEG_CR, exp=2, w_cluster=0.25, t_0=1,
                       Fs=500):
    req = (con_trial.Stim == sc) & (con_trial.Chan == rc) & (con_trial.Artefact < 1)
    dat = con_trial[req].reset_index(drop=True)
    EEG_trials = ff.lp_filter(np.expand_dims(EEG_CR[rc, dat.Num.values.astype('int'), :], 0), 45, Fs)

    LL_trials = LLf.get_LL_all(EEG_trials, Fs, w_cluster)

    # for each trial get significance level based on surrogate (Pearson^2 * LL)
    #### first get surrogate data
    pear_surr_all = []
    for t_test in [0.5, 2.1]:  # surrogates times, todo: in future blockwise
        pear = np.zeros((len(EEG_trials[0]),)) - 1  # pearson to each CC
        for n_c in range(len(M_GT)):
            pear = np.max([pear, sf.get_pearson2mean(M_GT[n_c, :], EEG_trials[0], tx=t_0 + t_resp, ty=t_test,
                                                     win=w_cluster,
                                                     Fs=500)], 0)
        print(pear.shape)
        LL = LL_trials[0, :, int((t_test + w_cluster / 2) * Fs)]
        pear_surr = np.sign(pear) * abs(pear ** exp) * LL  # square the correlation but keep the sign
        pear_surr_all = np.concatenate([pear_surr_all, pear_surr])

    # other surr trials
    real_trials = np.unique(
        con_trial.loc[req, 'Num'].values.astype('int'))
    # trials where RC is stimulating
    stim_trials = np.unique(
        con_trial.loc[(con_trial.Stim >= rc - 1) & (con_trial.Stim <= rc + 1), 'Num'].values.astype('int'))
    StimNum = np.random.choice(np.unique(con_trial.loc[
                                             ~np.isin(con_trial.Stim, [sc - 1, sc, sc + 1]) & ~np.isin(con_trial.Stim,
                                                                                                       [rc - 1, rc,
                                                                                                        rc + 1]) & (
                                                     con_trial.Artefact == 0), 'Num']), size=200)
    StimNum = [i for i in StimNum if i not in stim_trials]
    StimNum = [i for i in StimNum if i not in stim_trials + 1]
    StimNum = [i for i in StimNum if i not in real_trials]

    StimNum = np.unique(StimNum).astype('int')
    EEG_surr = ff.lp_filter(np.expand_dims(EEG_CR[rc, StimNum, :], 0), 45, Fs)
    bad_StimNum = np.where(np.max(abs(EEG_surr[0]), 1) > 1000)
    if (len(bad_StimNum[0]) > 0):
        StimNum = np.delete(StimNum, bad_StimNum)
        EEG_surr = ff.lp_filter(np.expand_dims(EEG_CR[rc, StimNum, :], 0), 45, Fs)
    if len(StimNum) > 0:
        LL_surr = LLf.get_LL_all(EEG_surr, Fs, w_cluster)
        f = 1
        for t_test in [0.5, 2.1]:  # surrogates times, todo: in future blockwise
            pear = np.zeros((len(EEG_surr[0]),)) - 1
            for n_c in range(len(M_GT)):
                pear = np.max([pear, sf.get_pearson2mean(M_GT[n_c, :], EEG_surr[0], tx=t_0 + t_resp, ty=t_test,
                                                         win=w_cluster,
                                                         Fs=500)], 0)

            LL = LL_surr[0, :, int((t_test + w_cluster / 2) * Fs)]
            # pear_surr = np.arctanh(np.max([pear,pear2],0))*LL
            pear_surr = np.sign(pear) * abs(pear ** exp) * LL
            pear_surr_all = np.concatenate([pear_surr_all, pear_surr])

    return pear_surr_all
