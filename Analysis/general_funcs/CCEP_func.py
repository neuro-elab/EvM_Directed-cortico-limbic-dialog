import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.fftpack
from scipy.signal import savgol_filter, find_peaks, peak_prominences
import scipy.io as sio
from general_funcs import freq_funcs as ff

cond_vals = np.arange(4)
cond_labels = ['BM', 'BL', 'Fuma', 'Benzo']
cond_colors = ['#494159', '#594157', "#F1BF98", "#8FB996"]


####GENERAL functions to analyze characteristics of a CCEP
def CCEP_metric(trials, t0=1, w_AUC=1, Fs=500):
    """
    Calculate CCEP metrics including N1, N2, and AUC.

    Parameters:
    trials (array): The trial data (trial x time).
    t0 (float): The time of stimulation onset in seconds.
    w_AUC (float): The width of the AUC window in seconds.
    Fs (int): The sampling frequency.

    Returns:
    tuple: N1, N2, AUC and P2P values.
    """
    # Mean response across trials
    mean = np.nanmean(trials, 0)
    mean = zscore_CCEP(mean)
    pol = -1 if abs(np.min(mean[int((t0 + 0.01) * Fs):int((t0 + 0.05) * Fs)])) > np.max(
        mean[int((t0 + 0.01) * Fs):int((t0 + 0.05) * Fs)]) else 1
    # if there is a neagitve peak within 50 ms -> pol = -1
    # if there is a postive peak within 50ms -> pol +
    # if there is a peak in both polarity, take the stronger one
    # if there is no peak,

    # Calculate N1 and N2
    N1 = np.max(pol * trials[:, int((t0 + 0.01) * Fs):int((t0 + 0.05) * Fs)], axis=1)
    N2 = np.max(pol * trials[:, int((t0 + 0.05) * Fs):int((t0 + 0.4) * Fs)], axis=1)
    # Get P2P within 1s
    peak_min = np.min(trials[:, int((t0 + 0.01) * Fs):int((t0 + w_AUC) * Fs)], axis=1)
    peak_max = np.max(trials[:, int((t0 + 0.01) * Fs):int((t0 + w_AUC) * Fs)], axis=1)
    P2P = abs(peak_max - peak_min)
    # Z-score the trials
    trials_z = zscore_CCEP(trials, t_0=t0, w0=0.5, Fs=Fs)

    # Calculate AUC
    auc_start = int(t0 * Fs)
    auc_end = int((t0 + w_AUC) * Fs)
    AUC = np.sum(np.abs(trials_z[:, auc_start:auc_end]), axis=1)

    return N1, N2, AUC, P2P


def zscore_CCEP(data, t_0=1, w0=0.5, w1=0.05, Fs=500):
    if len(data.shape) == 1:
        m = np.mean(data[int((t_0 - w0) * Fs):int((t_0 - w1) * Fs)])
        s = np.std(data[int((t_0 - w0) * Fs):int((t_0 - w1) * Fs)])
        data = (data - m) / s
    else:
        m = np.nanmean(data[:, int((t_0 - w0) * Fs):int((t_0 - 0.05) * Fs)], -1)
        s = np.nanstd(data[:, int((t_0 - w0) * Fs):int((t_0 - 0.05) * Fs)], -1)
        m = m[:, np.newaxis]
        s = s[:, np.newaxis]
        data = (data - m) / s
    return data

def get_peak(signal):
    """
       Find the location of the first peak among the two strongest peaks in both polarities.

       Parameters:
       - signal (array): Time signal.

       Returns:
       - float: Location of the first peak in seconds.
       """
    # Find positive peaks and their prominences
    positive_peaks, _ = find_peaks(signal)
    positive_prominences = peak_prominences(signal, positive_peaks)[0]

    # Find negative peaks and their prominences
    negative_peaks, _ = find_peaks(-signal)
    negative_prominences = peak_prominences(-signal, negative_peaks)[0]

    # Handle the case when no peaks are found
    if len(positive_prominences) == 0:
        sorted_positive_peaks = np.array([np.nan])
    else:
        # Sort positive peaks by prominence in descending order
        sorted_positive_peaks = positive_peaks[np.argsort(-positive_prominences)]

    if len(negative_prominences) == 0:
        sorted_negative_peaks = np.array([np.nan])
    else:
        # Sort negative peaks by prominence in descending order
        sorted_negative_peaks = negative_peaks[np.argsort(-negative_prominences)]

    # Get the two strongest peaks from both polarities
    first_peak = np.nanmin([sorted_positive_peaks[0], sorted_negative_peaks[0]])

    return first_peak

def peak_latency(trials, WOI, t0=1, Fs=500, w_LL=0.25):
    #Based on David et al. 2023
    BL_period = [int((t0 - 0.2) * Fs), int((t0 - 0.01) * Fs)]
    trials = ff.lp_filter(trials, 45, Fs)
    # 1. Baseline Correction
    bl_median = np.median(trials[:, BL_period[0]:BL_period[1]], axis=1)
    trials = trials - bl_median[:, None]

    # 2. Average signal
    mean_signal = np.mean(trials, axis=0)

    # 3. Baseline correction: subtract baseline mean and z-score with respect to baseline
    baseline_mean = np.mean(mean_signal[BL_period[0]:BL_period[1]])
    baseline_std = np.std(mean_signal[BL_period[0]:BL_period[1]])

    # 4. Z-scoring the mean signal
    z_scored_signal = (mean_signal - baseline_mean) / baseline_std

    threshold = 5
    peaks, _ = find_peaks(z_scored_signal, height=threshold)
    neg_peaks, _ = find_peaks(-z_scored_signal, height=threshold)
    peaks = peaks[(peaks > int((t0 + 0.015) * Fs)) & (peaks < int((t0 + WOI + 2 * w_LL / 3) * Fs))]
    neg_peaks = neg_peaks[(neg_peaks > int((t0 + 0.015) * Fs)) & (neg_peaks < int((t0 + WOI + 2 * w_LL / 3) * Fs))]

    peak_detected = 1
    # Finding the first peak crossing statistical threshold
    if peaks.size > 0 or neg_peaks.size > 0:
        all_peaks = np.sort(np.concatenate((peaks, neg_peaks)))
        first_peak = all_peaks[0]
    else:
        mean_signal = mean_signal - baseline_mean
        mean_signal[:int((t0 + 0.010) * Fs)] = 0
        mean_signal[int((t0 + WOI + 0.3) * Fs):] = 0
        first_peak = get_peak(mean_signal)
        peak_detected = 0
    if np.isnan(first_peak):
        polarity = np.nan
    else:
        polarity = np.sign(mean_signal[int(first_peak)])
    return first_peak / Fs - t0, polarity, peak_detected


def CCEP_onset(trials, WOI=0, t0=1, Fs=500, w_LL=0.25, plot=False, skip_nonpeak = 0):
    """
    Calculate the onset of a Cortico-Cortical Evoked Potential (CCEP) in a signal.

    Parameters:
    - trials (array): all trials for given connection .
    - WOI (float): Onset of Window Of Interest based on previous LL calculations (connection-specific).
    - t_0 (float): Time of stimulation in the signal (e.g., for epoch: [-1, 3] -> t_0 = 1).
    - Fs (int): Sampling frequency.
    - w_LL_onset (float): Window length for onset detection.

    Returns:
    - float: Time of response onset after stimulation, in seconds.
    """
    signal = np.mean(trials, 0)
    peak_lat, polarity, peak_detected = peak_latency(trials, WOI, t0=1, Fs=500, w_LL=0.25)

    return np.nan, peak_lat, polarity, peak_detected