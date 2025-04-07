import numpy as np
from tkinter import *

#root = Tk()
#root.withdraw()
#from general_funcs import freq_funcs as ff
from general_funcs import basic_func as bf
#from general_funcs import LL_funcs as LLf
#
#from scipy.signal import hilbert, butter, filtfilt
# from scipy.signal import find_peaks

method_labels = ['LL', 'Pearson', 'Compound (LL*Pearson)']


def recovery_channel_artefact(con_trial, EEG_resp, labels_clinic, Fs=500):
    """
    Identify and label artefactual trials in EEG data based on the AUC comparison
    to a baseline distribution. This function examines the recovery channels for
    each stimulated channel. It compares the AUC of EEG responses in the recovery
    trials to the AUC of non-recovery trials. If the AUC of a recovery trial
    exceeds the 99th percentile of the AUC distribution of the non-recovery trials,
    it is labeled as a bad trial (artefact = 2).

    Parameters:
        con_trial (DataFrame): DataFrame containing trial information, including stimulation
                               and channel data.
        EEG_resp (array): Multidimensional array containing EEG response data for each trial.
        labels_clinic (list): List of clinical labels associated with the channels.

    Returns:
        DataFrame: The modified DataFrame with artefact trials marked.
    """

    # for each channel that was stimulated
    for rc in np.unique(con_trial.Stim):
        stim_channels = bf.check_stim_labels(rc, labels_clinic)  # Validate and get stimulated channels
        stim_num = np.unique(
            con_trial.loc[np.isin(con_trial.Stim, stim_channels), 'Num'])  # Find trial numbers for stimulated trials

        if len(stim_num) > 0:
            # Identify next trial numbers following stimulated trials (recovery trials)
            rec_trials_num = np.unique(con_trial.loc[(con_trial.Chan == rc) & np.isin(con_trial.Num, stim_num + 1) & (
                    con_trial.Artefact < 1), 'Num'])
            rec_trials = EEG_resp[rc, rec_trials_num,
                         int(0.4 * Fs):int(0.9 * Fs)]  # Extract EEG responses for recovery trials

            # Identify trial numbers that are not directly following stimulated trials
            non_rec_trials_num = np.unique(con_trial.loc[
                                               (con_trial.Chan == rc) & (~np.isin(con_trial.Num, stim_num + 1)) & (
                                                       con_trial.Artefact < 1), 'Num'])

            if len(non_rec_trials_num) > 1:
                non_rec_trials = EEG_resp[rc, non_rec_trials_num,
                                 int(0.4 * Fs):int(0.9 * Fs)]  # Extract EEG responses for non-recovery trials
                AUC_rec = np.trapz(abs(rec_trials), axis=-1)  # Compute AUC for recovery trials
                AUC_nonrec = np.trapz(non_rec_trials, axis=-1)  # Compute AUC for non-recovery trials

                thr = 2 * np.nanpercentile(AUC_nonrec,
                                           99)  # Determine the 99th percentile threshold from non-recovery trials
                bad_rec = np.where(AUC_rec > thr)  # Identify bad recovery trials

                if bad_rec[0].size > 0:
                    con_trial.loc[(con_trial.Chan == rc) & np.isin(con_trial.Num, rec_trials_num[
                        bad_rec[0]]), 'Artefact'] = 2  # Label bad recovery trials

    return con_trial


def mark_artefacts(con_trial, metric, thr=6):
    con_trial.insert(0, 'zP2P', con_trial.groupby(['Stim', 'Chan'])['P2P'].transform(
        lambda x: (x - x.mean()) / x.std()).values)

    for c in np.unique(con_trial.Chan):
        val_dist = con_trial.loc[(con_trial.Chan == c) & (con_trial.Artefact == 0), metric].values
        if len(val_dist) > 0:
            val_dist_z = (val_dist - np.nanmean(val_dist)) / np.nanstd(val_dist)
            con_trial.loc[(con_trial.Chan == c) & (con_trial.Artefact == 0), 'zscore'] = val_dist_z
    con_trial.loc[(con_trial.Artefact == 0) & (con_trial.zscore > thr) & (con_trial.zP2P > thr), 'Artefact'] = 3
    con_trial.loc[(con_trial.Artefact == 0) & (con_trial.P2P_BL > 3000), 'Artefact'] = 3
    con_trial.drop('zscore', axis=1, inplace=True)
    con_trial.drop('zP2P', axis=1, inplace=True)
    return con_trial


def search_sequence_numpy(arr, seq):
    """ Find sequence in an array using NumPy only.

    Parameters
    ----------
    arr    : input 1D array
    seq    : input 1D array

    Output
    ------
    Output : 1D Array of indices in the input array that satisfy the
    matching of input sequence in the input array.
    In case of no match, an empty list is returned.
    """

    # Store sizes of input array and sequence
    Na, Nseq = arr.size, seq.size

    # Range of sequence
    r_seq = np.arange(Nseq)

    # Create a 2D array of sliding indices across the entire length of input array.
    # Match up with the input sequence & get the matching starting indices.
    M = (arr[np.arange(Na - Nseq + 1)[:, None] + r_seq] == seq).all(1)

    # Get the range of those indices as final output
    if M.any() > 0:
        return np.where(np.convolve(M, np.ones((Nseq), dtype=int)) > 0)[0]
    else:
        return []  # No match found


def get_pearson2mean_windowed(x_gt, x_trials, tx=1, win=0.25, Fs=500):
    # get pearson coeff for all trials to ground truth for selected time window
    x0 = int(tx * Fs)
    x1 = int(x0 + win * Fs)
    wdp = np.int64(Fs * win)  # 100ms -> 50 sample points
    EEG_pad = np.pad(x_trials, [(0, 0), (np.int64(wdp / 2), np.int64(wdp / 2))], 'constant',
                     constant_values=(0, 0))  # 'reflect'(18, 3006)
    corr_all = np.zeros((x_trials.shape[0], x_trials.shape[1]))
    for i in range(x_trials.shape[1]):  # entire response
        corr_all[:, i] = np.corrcoef(x_gt[x0:x1], EEG_pad[:, i:int(i + (win * Fs))])[0, 1:]
    return corr_all


def calculate_max_cross_correlation(x_gt, x_trials, tx=1, ty=1, win=0.25, Fs=500, max_lag_ms=10):
    """
    Calculate the maximum cross-correlation for each trial with the ground truth time series.

    :param ground_truth: Ground truth time series (array-like).
    :param trials: Trials to be investigated (2D array-like, each row is a trial).
    :param fs: Sampling frequency in Hz.
    :param max_lag_ms: Maximum time lag allowed in milliseconds.
    :return: Array of maximum cross-correlations for each trial.
    """
    max_lag = int(Fs * max_lag_ms / 1000)  # Convert max lag from ms to samples
    x0 = int(tx * Fs)
    x1 = int(x0 + win * Fs)
    y0 = int(ty * Fs) - max_lag
    y1 = int(y0 + win * Fs) + max_lag
    ground_truth = x_gt[x0:x1]
    trials = x_trials[:, y0:y1]

    max_correlations = np.zeros(trials.shape[0])

    for i, trial in enumerate(trials):
        max_corr = 0

        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                shifted_trial = np.pad(trial[:lag], (abs(lag), 0), 'constant')
                shifted_gt = ground_truth
            else:
                shifted_trial = trial
                shifted_gt = np.pad(ground_truth[lag:], (0, lag), 'constant')

            corr = np.corrcoef(shifted_trial, shifted_gt)[0, 1]

            if abs(corr) > abs(max_corr):
                max_corr = corr

        max_correlations[i] = max_corr

    return max_correlations


def get_shifted_pearson_correlation(x_gt, x_trials, tx, ty, win, Fs, max_shift_ms=10):
    """
        Calculate the maximum cross-correlation for each trial with the ground truth time series.

        :param ground_truth: Ground truth time series (array-like).
        :param trials: Trials to be investigated (2D array-like, each row is a trial).
        :param fs: Sampling frequency in Hz.
        :param max_lag_ms: Maximum time lag allowed in milliseconds.
        :return: Array of maximum cross-correlations for each trial.
        """
    # Define the window for the ground truth based on WOI
    x0 = int(tx * Fs)
    x1 = int(x0 + win * Fs)

    # Calculate the maximum allowed shift in samples
    max_shift = int(max_shift_ms * Fs / 1000)

    # Initialize an array to store the maximum correlation for each trial
    if len(x_trials.shape) == 1:
        trials = [x_trials]
    else:
        trials = x_trials

    max_correlations = np.zeros(len(trials))
    max_correlations_lag = np.zeros(len(trials))

    # Iterate over each trial
    for i, trial in enumerate(trials):
        max_corr = -1
        max_corr_lag = 0
        # Iterate over the allowed shift range
        for shift in range(-max_shift, max_shift + 1):
            y0 = int((ty + shift / Fs) * Fs)
            y1 = int(y0 + win * Fs)

            # Ensure the window is within the bounds of the trial
            if y0 >= 0 and y1 <= len(trial):
                # Calculate Pearson correlation coefficient
                corr = np.corrcoef(x_gt[x0:x1], trial[y0:y1])[0, 1]

                # Check if this is the maximum correlation for this trial
                if corr > max_corr:
                    max_corr = corr
                    max_corr_lag = y0

        max_correlations[i] = max_corr
        max_correlations_lag[i] = max_corr_lag

    return max_correlations, max_correlations_lag


def get_pearson2mean(x_gt, x_trials, tx=1, ty=1, win=0.25, Fs=500):
    # get pearson coeff for all trials to ground truth for selected time window
    x0 = int(tx * Fs)
    x1 = int(x0 + win * Fs)
    y0 = int(ty * Fs)
    y1 = int(y0 + win * Fs)
    if len(x_trials.shape) == 1:
        corr = np.corrcoef(x_gt[x0:x1], x_trials[y0:y1])[0, 1]
        # corr, p = stats.pearsonr(mn_gt[x0:x1], x_trials[y0:y1],1) # spearmanr
        # corr = corr[0,1]
    else:
        corr = np.corrcoef(x_gt[x0:x1], x_trials[:, y0:y1])[0, 1:]
        # corrs, p = stats.spearmanr(np.expand_dims(x_gt[x0:x1], 0), x_trials[:, y0:y1], 1, alternative='greater')
        # # corr = corr[0,1:]
        # p = p[0, 1:]
        # corr[p > 0.01] = corr[p > 0.01] * 0.8

    return corr
