import os
import numpy as np
import pywt
from scipy import signal

from scipy import fft
import sys
from scipy.signal import hilbert, chirp


def butter_highpass(cutoff, fs, order=5):
    nyq             = 0.5 * fs
    normal_cutoff   = cutoff / nyq
    [b, a]          = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_lowpass(cutoff, fs, order=5):
    nyq                 = 0.5 * fs
    normal_cutoff       = cutoff / nyq
    [b, a]              = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq         = 0.5 * fs
    low         = lowcut / nyq
    high        = highcut / nyq
    b, a        = signal.butter(order, [low, high], btype='band')
    return b, a

def hp_filter(data, cutoff, fs, order=5, dir=0):
    [b, a]      = butter_highpass(cutoff, fs, order=order)
    if dir == 'f':
        y = signal.lfilter(b, a, data)
    elif dir == 'b':
        y = signal.lfilter(b, a, np.flip(data, -1))
        y = np.flip(y, -1)
    else:
        y = signal.filtfilt(b, a, data)
    return y

def resample(data, fs_old, fs_new):
    data = signal.decimate(data, fs_old/fs_new)
    return data

def lp_filter(data, cutoff, fs, order=5, dir=0):
    [b, a] = butter_lowpass(cutoff, fs, order=order)
    if dir == 'f':
        y = signal.lfilter(b, a, data)
    elif dir == 'b':
        y = signal.lfilter(b, a, np.flip(data, -1))
        y = np.flip(y,-1)
    else:
        y = signal.filtfilt(b, a, data)
    return y

def bp_filter(data, lowcut, highcut, fs, order=5, dir=0):
    b, a        = butter_bandpass(lowcut, highcut, fs, order=order)
    if dir == 'f':
        y = signal.lfilter(b, a, data)
    elif dir == 'b':
        y = signal.lfilter(b, a, np.flip(data, -1))
        y = np.flip(y, -1)
    else:
        y = signal.filtfilt(b, a, data)
    return y

def get_WT(data, Fs, f0=0.5, f1=200, nf=100):
    ### Morlet WT
    # data = [chans, trials, time window], 3 dims
    T_samp          = 1 / Fs
    scale           = Fs/np.geomspace(f0, f1, nf)
    # freq          = pywt.scale2frequency('cmor1.5-1.0', scale)/ T_samp
    # calculate pwr for each trial
    coef, freqs     = pywt.cwt(data, scale, "cmor1.5-1.0", sampling_period=T_samp)
    pwr             = abs(coef) # absolute values
    phase           = np.angle(coef)
    # phase_mean      = np.nanmean(phase, axis=2) #mean for all trials
    # pwr_mean        = np.nanmean(pwr, axis=2)
    return pwr, phase, freqs

def get_HT(data, Fs, f0=0.5, f1=200, nf=100):
    # Employ filtering with logarithmic spacing of filters
    freqs       = np.geomspace(f0, f1, num=nf, endpoint=False)
    steps_v     = np.diff(freqs)

    # Initialize array:
    pwr       = np.zeros((len(freqs), data.shape[0], data.shape[1]))
    phase     = np.zeros((len(freqs), data.shape[0], data.shape[1]))
    i         = 0
    # For each sub-frequency band:
    for f, stepf in zip(freqs, steps_v):
        data_filt          = bp_filter(data, f, f + stepf, Fs)
        analytic_signal    = hilbert(data_filt)
        pwr[i]             = np.abs(analytic_signal)
        #phase[i]           = np.unwrap(np.angle(analytic_signal))
        i =i+1
    return pwr, phase, freqs

def get_FFT(data, Fs):
    ### Fast Fourier Transform
    y               = fft(data)
    n               = data.shape[2]
    y               = y[:,:,0:np.int(n / 2 )]
    f               = np.arange(0, n / 2) * Fs / n  # (0:n/2-1)*(Fs/n)
    power           = abs(y) ** 2 / n
    power_mean      = np.nanmean(power, axis=1)
    return f, power, power_mean
