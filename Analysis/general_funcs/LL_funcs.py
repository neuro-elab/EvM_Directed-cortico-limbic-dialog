
import numpy as np
import scipy.fftpack
import freq_funcs as ff
import basic_func as bf

def get_LL_bl(data, wdp_S, win_S, Fs):
    wdp = np.int(Fs * wdp_S)  # sliding window size in samples
    win = np.int(Fs * win_S)  # total window that is analyzed 1.5s
    EEG_pad = np.pad(data, [(0, 0), (0, 0), (np.int(wdp / 2), np.int(wdp / 2))], 'reflect')  # (18, 3006)
    LL_trial = np.zeros((data.shape[0], data.shape[1], win))
    for i in range(win):  # calculate LL for sliding window. take values of +- 1/2 wdp
        n = i + np.int(wdp / 2)
        LL_trial[:, :, i] = np.sum(abs(np.diff(EEG_pad[:, :, n - np.int(wdp / 2):n + np.int(wdp / 2)], axis=-1)),
                                   axis=-1)
    LL_mean = np.mean(LL_trial, axis=(1, 2))
    LL_std = np.std(LL_trial, axis=(1, 2))
    LL_thr = LL_mean + 2 * LL_std

    return LL_mean, LL_std, LL_thr, LL_trial


def get_P2P_resp(data, Fs, IPI, t_0):  # specific for Int and IPI
    # t_0 after how many seconds starts the stimulation --> -self.dur[0, 0]
    start = np.int((t_0 + IPI / 1000 + 0.01) * Fs)  # 50ms after second response
    end = np.int((t_0 + IPI / 1000 + 0.07) * Fs)  # 300ms after second response
    end2 = np.int((t_0 + IPI / 1000 + 0.04) * Fs)

    resp_pks = np.zeros((data.shape[0], data.shape[1], 3))
    resp_pks_loc = np.zeros((data.shape[0], data.shape[1], 2))
    resp_pks[:, :, 0] = np.min(data[:, :, start:end], axis=-1)
    resp_pks_loc[:, :, 0] = (np.argmin(data[:, :, start:end], axis=-1) + start - (t_0) * Fs) / Fs
    resp_pks[:, :, 1] = np.max(data[:, :, start:end2], axis=-1)
    resp_pks_loc[:, :, 1] = (np.argmax(data[:, :, start:end2], axis=-1) + start - (t_0) * Fs) / Fs
    resp_pks[:, :, 2] = resp_pks[:, :, 1] - resp_pks[:, :, 0]

    return resp_pks, resp_pks_loc


def get_RMS_resp(data, Fs, IPI, t_0, win):  # specific for Int and IPI
    # t_0 after how many seconds starts the stimulation --> -self.dur[0, 0]
    start = np.int((t_0 + IPI / 1000 + 0.012) * Fs)  # 15ms after second response
    end = np.int((t_0 + IPI / 1000 + 0.012 + win) * Fs)  # 515ms after second response

    resp_RMS = np.zeros((data.shape[0], data.shape[1], 1))
    resp_RMS[:, :, 0] = np.sqrt(np.mean(data[:, :, start:end] ** 2, axis=2))

    return resp_RMS


def get_LL_resp(data, Fs, IPI, t_0, win):  # specific for Int and IPI
    # t_0 after how many seconds starts the stimulation --> -self.dur[0, 0]
    # win in seconds
    start = np.int((t_0 + IPI / 1000 + 0.010) * Fs)  # 15 ms after second response
    end = np.int((t_0 + IPI / 1000 + 0.010 + win) * Fs)  # eg. win= 0.300 --> 315ms after second response

    resp_LL = np.zeros((data.shape[0], data.shape[1], 1))
    resp_LL[:, :, 0] = np.sum(abs(np.diff(data[:, :, start:end], axis=-1)), axis=-1) / (win * 1000)

    return resp_LL


def get_N1peaks_mean(sc, rc, LL_CCEP, EEG_resp):
    lists = LL_CCEP[
        (LL_CCEP['Artefact'] == 0) & (LL_CCEP['Int'] >= 6) & (LL_CCEP['Chan'] == rc) & (LL_CCEP['Stim'] == sc)]
    stimNum_all = lists.Num.values.astype('int')
    resp_all = bf.zscore_CCEP(ff.lp_filter(np.mean(EEG_resp[rc, stimNum_all, :], 0), 40, Fs))
    if np.max(abs(resp_all[Fs:int(1.5 * Fs)])) > 4:
        start_resp = 0
        peaks_all, properties_all = scipy.signal.find_peaks(abs(resp_all[int(t_0 * Fs):int((t_0 + 0.5) * Fs)]),
                                                            height=1.5, prominence=0.05, distance=0.03 * Fs, width=1)  #
        if len(peaks_all) > 0:
            peaks_all = peaks_all[0]
            w = \
                scipy.signal.peak_widths(abs(resp_all[int(t_0 * Fs):int((t_0 + 0.5) * Fs)]), [peaks_all],
                                         rel_height=0.5)[0]

            start_resp = (peaks_all - w) / Fs - 0.01

            if start_resp < 0.01:
                start_resp = 0

        pk, pk_s, p = get_peaks_all(resp_all, start_resp)

    else:
        pk_s = np.zeros((4, 1))
        p = 0

    return pk_s[0], p


def get_LL_both(data, Fs, IPI, t_0=1, win=0.25):  # specific for Int and IPI
    # get LL of first and second pulse. put first pulse LL to nan if window is larger than IPI
    # t_0 = 5
    # w = 0.1
    art = 0.01
    resp_LL = np.zeros((data.shape[0], data.shape[1], 2))
    # first pulse
    # IPI_start = np.round((t_0+art)*Fs) # start position at first trigger plus 20ms (art removal), time zero
    w_start = np.int64(np.round((t_0 + art) * Fs))  # start position at sencond trigger plus 20ms (art removal)
    w_end = np.int64(np.round((t_0 + win) * Fs) - 1)
    n = np.int64((w_end - w_start))
    inds = np.linspace(w_start, w_end, n).T.astype(int)
    inds = np.expand_dims(inds, axis=(0, 1))
    # set to nan if IPI is smaller than LL window
    nans_ind = np.where(IPI < (win) * 1000)[0]
    resp_LL[:, :, 0] = np.sum(abs(np.diff(np.take_along_axis(data, inds, axis=2), axis=-1)), axis=-1) / (win * 1000)
    resp_LL[:, nans_ind, 0] = np.nan

    # second pulse
    w_start = np.int64(
        np.round((IPI / 1000 + t_0 + art) * Fs))  # start position at sencond trigger plus 20ms (art removal)
    w_end = np.int64(np.round((IPI / 1000 + t_0 + win) * Fs) - 1)
    n = np.int64((w_end - w_start)[0, 0])
    inds = np.linspace(w_start, w_end, n).T.astype(int)

    resp_LL[:, :, 1] = np.sum(abs(np.diff(np.take_along_axis(data, inds, axis=2), axis=-1)), axis=-1) / (win * 1000)
    return resp_LL  # chns x stims x SP/PP LL



def pk2pk(data, Fs, t_0):
    start = np.int64((t_0 + 0.015) * Fs)  # 50ms after second response
    end = np.int64((t_0 + 0.10) * Fs)  # 50ms after second response
    resp_pks = np.zeros((data.shape[0], 3))
    resp_pks_loc = np.zeros((data.shape[0], 2))
    resp_pks[:, 0] = np.min(data[:, start:end], axis=-1)
    resp_pks[:, 1] = np.max(data[:, start:end], axis=-1)
    resp_pks[:, 2] = resp_pks[:, 1] - resp_pks[:, 0]
    resp_pks_loc[:, 1] = (np.argmax(data[:, start:end], axis=-1) + start - (t_0) * Fs) / Fs
    resp_pks_loc[:, 0] = (np.argmin(data[:, start:end], axis=-1) + start - (t_0) * Fs) / Fs
    return resp_pks, resp_pks_loc


def get_LL_full(data, Fs, IPI, t_0=5, win=0.1):
    art = 0.01
    resp_LL = np.zeros((data.shape[0], data.shape[1], 2))
    # first pulse
    # IPI_start = np.round((t_0+art)*Fs) # start position at first trigger plus 20ms (art removal), time zero
    w_start = np.int64(np.round((t_0 + art) * Fs))  # start position at sencond trigger plus 20ms (art removal)
    w_end = np.int64(np.round((t_0 + win) * Fs) - 1)
    n = np.int64((w_end - w_start))
    inds = np.linspace(w_start, w_end, n).T.astype(int)
    inds = np.expand_dims(inds, axis=(0, 1))
    # set to nan if IPI is smaller than LL window
    nans_ind = np.where(IPI < (win) * 1000)[0]
    resp_LL[:, :, 0] = np.sum(abs(np.diff(np.take_along_axis(data, inds, axis=2), axis=-1)), axis=-1) / (win * 1000)
    resp_LL[:, nans_ind, 0] = np.nan

    # second pulse
    w_start = np.int64(
        np.round((IPI / 1000 + t_0 + art) * Fs))  # start position at sencond trigger plus 20ms (art removal)
    w_end = np.int64(np.round((IPI / 1000 + t_0 + win) * Fs) - 1)
    n = np.int64((w_end - w_start)[0, 0])
    inds = np.linspace(w_start, w_end, n).T.astype(int)

    resp_LL[:, :, 1] = np.sum(abs(np.diff(np.take_along_axis(data, inds, axis=2), axis=-1)), axis=-1) / (win * 1000)
    return resp_LL  # chns x stims x SP/PP LL



def get_LL_all(data, Fs, win):  # specific for Int and IPI
    ## LL for entire signal
    wdp = np.int64(Fs * win)  # 100ms -> 50 sample points
    EEG_pad = np.pad(data, [(0, 0), (0, 0), (np.int64(wdp / 2), np.int64(wdp / 2))], 'constant',
                     constant_values=(0, 0))  # 'reflect'(18, 3006)
    LL_trial = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
    # LL_max = np.zeros((data.shape[0], data.shape[1], 2))
    for i in range(data.shape[2]):  # entire response
        n = i + np.int64(wdp / 2)
        LL_trial[:, :, i] = np.nansum(abs(np.diff(EEG_pad[:, :, n - np.int64(wdp / 2):n + np.int64(wdp / 2)], axis=-1)),
                                      axis=-1) / (win * 1000)
    # onset in ms after probing pulse, must be at least 10ms + half of the sliding window to not include fake data
    return LL_trial
