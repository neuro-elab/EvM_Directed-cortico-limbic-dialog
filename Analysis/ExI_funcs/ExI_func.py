import os
import numpy as np
import pandas as pd
from ..general_funcs import freq_funcs as ff
from ..general_funcs import basic_func as bf
from ..general_funcs import LL_funcs as LLf
import scipy.fftpack
import sklearn
import matplotlib.pyplot as plt
from scipy.spatial import distance
from numpy import trapz
from glob import glob
import ntpath

cond_vals = np.arange(4)
cond_labels = ['BM', 'BL', 'Fuma', 'Benzo']
cond_colors = ['#494159', '#594157', "#F1BF98", "#8FB996"]

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


def get_AUC(mag_val, int_val, mag_max = -1):
    if mag_max == -1:
        mag_max = np.max(mag_val)
    mag_norm = (mag_val - np.min(mag_val)) / (mag_max - np.min(mag_val))
    Int_norm = (int_val - np.min(int_val)) / (np.max(int_val) - np.min(int_val))
    auc_value = sklearn.metrics.auc(Int_norm, mag_norm)
    return auc_value, mag_max


def get_LL_all_block(EEG_resp, stimlist, lbls, bad_chans, Fs=500,t_0=1,w_LL=0.25):
    """
        Calculates response magnitude (LL) for each stimulation and response channel.
        Marks the channels close to stimulation as artefacts and stores other important information.

        Parameters:
        EEG_resp (array): EEG response data (channels, trials, timepoints).
        stimlist (DataFrame): Stimulus information.
        lbls (list): Channel labels.
        bad_chans (list): List of channels to mark as artefacts.
        Fs (int): Sampling frequency (default 500 Hz).
        t_0 (float): Time of interest for calculation (default 1s).
        w_LL (float): Window size for LL calculation (default 0.25s).

        Returns:
        DataFrame: Contains LL and other information related to stimulations and responses.
        """

    # Set baseline time point
    t_bl = t_0 - 0.5 - 0.01

    # Get stimulation channel data
    labels_all, labels_region, labels_clinic, coord_all, StimChans, StimChanSM, StimChansC, StimChanIx, stimlist = bf.get_Stim_chans(
        stimlist, lbls)

    # Add necessary columns to stimlist if not present
    for col, default_value in [("Num_block", stimlist.StimNum), ("condition", 0), ("sleep", 0)]:
        if col not in stimlist.columns:
            stimlist.insert(0, col, default_value, True)

    # Initialize result array
    data_LL = np.zeros((1, 12))

    # Select relevant stimulations
    stim_spec = stimlist[stimlist.IPI_ms == 0]  # Filter based on IPI
    stimNum = stim_spec.StimNum.values
    noise_val = stim_spec.noise.values

    if len(stimNum)>0:
        #resps = EEG_resp[:, stimNum, :]
        resps = ff.lp_filter(EEG_resp[:, stimNum, :], 45, Fs)
        ChanP1 = bf.SM2IX(stim_spec.ChanP.values, StimChanSM, np.array(StimChanIx))
        IPIs = np.expand_dims(np.array(stim_spec.IPI_ms.values), 1)
        #LL = LLf.get_LL_both(data=resps, Fs=Fs, IPI=IPIs, t_0=1, win=w)
        LL_trial = LLf.get_LL_all(resps[:, :, int(t_0 * Fs):int((t_0+0.5) * Fs)], Fs, w_LL)
        LL_peak = np.max(LL_trial, 2)
        t_peak = np.argmax(LL_trial, 2) + int((t_0 - w_LL / 2) * Fs)
        t_peak[t_peak < (t_0 * Fs)] = t_0 * Fs
        ## Baseline LL (control, no stim)
        LL_trial = LLf.get_LL_all(resps[:, :, int(t_bl * Fs):int((t_bl + 0.5) * Fs)], Fs, w_LL)
        LL_peak_bl = np.max(LL_trial, 2)

        for c in range(len(LL_peak)):
            val = np.zeros((LL_peak.shape[1], 12))
            val[:, 0] = c  # response channel
            val[:, 1] = ChanP1
            val[:, 4] = stim_spec.Int_prob.values  # Intensity
            val[:, 3] = noise_val
            val[:, 2] = LL_peak[c, :]  # LL
            val[:, 6] = stim_spec['h'].values
            val[:, 5] = stim_spec['condition'].values
            val[:, 7] = stimNum
            val[:, 8] = stim_spec.date.values
            val[:, 9] = stim_spec.sleep.values
            val[:, 10] = stim_spec.stim_block.values
            val[:, 11] =  LL_peak_bl[c, :]  # LL_peak_ratio[c, :]  # ratio
            # set stimulation channels to nan, artefact = 1
            val[np.where(bf.check_inStimChan(c, ChanP1, labels_clinic) == 1), 3] = 1
            val[np.where(bf.check_inStimChan(c, ChanP1, labels_clinic) == 1), 2] = np.nan

            # if its the recovery channel, check if strange peak is appearing
            pks = np.max(abs(resps[c, :, np.int64((t_0 - 0.05) * Fs):np.int64((t_0 + 0.5) * Fs)]), 1)
            pks_loc = np.argmax(abs(resps[c, :, np.int64((t_0 - 0.05) * Fs):np.int64((t_0 + 0.5) * Fs)]), 1) + np.int64(
                (t_0 - 0.05) * Fs)
            ix = np.where(
                (pks > 500) & (pks_loc > np.int64((t_0 - 0.005) * Fs)) & (pks_loc < np.int64((t_0 + 0.008) * Fs)))
            # original stim number:
            sn = stim_spec.StimNum.values[ix]
            rec_chan = stimlist.loc[np.isin(stimlist.StimNum, sn - 1), 'ChanP'].values
            rec_chan = bf.SM2IX(rec_chan, StimChanSM, np.array(StimChanIx))
            if np.isin(c, rec_chan):
                val[ix, 3] = 1

            data_LL = np.concatenate((data_LL, val), axis=0)

        data_LL = data_LL[1:-1, :]  # remove first row (dummy row)
        LL_all = pd.DataFrame(
            {"Chan": data_LL[:, 0], "Stim": data_LL[:, 1], "LL": data_LL[:, 2], "LL_BL": data_LL[:, 11], "Artefact": data_LL[:, 3], "Int": data_LL[:, 4],
             'Condition': data_LL[:, 5], "Block": data_LL[:, 10],
             "Num": data_LL[:, 7],"Num_block": data_LL[:, 7]})


        # remove bad channels
        LL_all.loc[(LL_all.Chan).isin(bad_chans), 'Artefact'] = 1
        LL_all.loc[(LL_all.Chan).isin(bad_chans), 'Artefact'] = 1
    else:
        data_LL = np.zeros((1, 12))
        data_LL[:,2:4] = np.nan
        LL_all = pd.DataFrame(
            {"Chan": data_LL[:, 0], "Stim": data_LL[:, 1], "LL": data_LL[:, 2], "LL_BL": data_LL[:, 2], "Artefact": data_LL[:, 3], "Int": data_LL[:, 4],
             'Condition': data_LL[:, 5], "Block": data_LL[:, 10],
             "Num": data_LL[:, 7],"Num_block": data_LL[:, 7]})
    return LL_all



def LL_mx(EEG_trial, Fs=500, w=0.25, t0=1.01):
    # calculate mean response and get LL (incl peak)
    resp = ff.lp_filter(np.mean(EEG_trial, 0), 20, Fs)
    LL_resp = LLf.get_LL_all(np.expand_dims(np.expand_dims(resp, axis=0), 0), Fs, w)
    LL_resp = LL_resp[0, 0]
    mx = np.max(LL_resp[np.int64((t0 + w / 2) * Fs):np.int64((t0 + w) * Fs)])
    mx_ix = np.argmax(LL_resp[np.int64((t0 + w / 2) * Fs):np.int64((t0 + w) * Fs)])
    return mx, mx_ix, LL_resp


def sig_resp(mean, thr, w=0.25, Fs=500):
    # check whether a mean response is a significant CCEP based on a pre-calculated threshold thr
    mean = ff.lp_filter(mean, 45, Fs)
    LL_resp = LLf.get_LL_all(np.expand_dims(np.expand_dims(mean, axis=0), 0), Fs, w)
    LL_resp = LL_resp[0, 0]
    mx = np.max(LL_resp[np.int64((1.01 + w / 2) * Fs):np.int64((1.01 + w) * Fs)])
    max_ix = np.argmax(LL_resp[np.int64((1.01 + w / 2) * Fs):np.int64((1.01 + w) * Fs)])
    if mx > thr:
        sig = 1
    else:
        sig = 0
    return LL_resp, mx, max_ix, sig


def concat_resp_condition(subj, cond_folder='CR'):
    folder = 'InputOutput'
    path_patient_analysis = 'T:\EL_experiment\Projects\EL_experiment\Analysis\Patients\\' + subj
    files = glob(path_patient_analysis + '\\' + folder + '\\data\\Stim_list_*'+cond_folder+'*')
    files = np.sort(files)
    # prots           = np.int64(np.arange(1, len(files) + 1))  # 43
    stimlist = []
    EEG_resp = []
    conds = np.empty((len(files),), dtype=object)
    for p in range(len(files)):
        file = files[p]
        #file = glob(self.path_patient + '/Analysis/'+folder+'/data/Stim_list_' + str(p) + '_*')[0]
        idxs       = [i for i in range(0, len(ntpath.basename(file))) if ntpath.basename(file)[i].isdigit()]
        cond       = ntpath.basename(file)[idxs[-2] - 2:idxs[-2] ]#ntpath.basename(file)[idxs[-2] + 2:-4]  #
        conds[p]   = cond
        EEG_block  = np.load(path_patient_analysis + '\\' + folder + '\\data\\All_resps_' + file[-11:-4]+ '.npy')
        print(str(p+1)+'/'+str(len(files))+' -- All_resps_' + file[-11:-4])
        stim_table = pd.read_csv(file)
        stim_table['type'] = cond
        if len(stimlist) == 0:
            EEG_resp = EEG_block
            stimlist = stim_table
        else:
            EEG_resp = np.concatenate([EEG_resp, EEG_block], axis=1)
            stimlist = pd.concat([stimlist, stim_table])
    stimlist  = stimlist.drop(columns="StimNum", errors='ignore')
    stimlist        = stimlist.fillna(0)
    stimlist = stimlist.reset_index(drop=True)
    col_drop = ["StimNum",'StimNum.1', 's', 'us', 'ISI_s', 'TTL', 'TTL_PP', 'TTL_DS', 'TTL_PP_DS', 'currentflow']
    for d in range(len(col_drop)):
        if (col_drop[d] in stimlist.columns):
            stimlist = stimlist.drop(columns=col_drop[d])
    stimlist.insert(0, "StimNum", np.arange(len(stimlist)), True)
    np.save(path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\EEG_'+cond_folder+'.npy', EEG_resp)
    stimlist.to_csv(path_patient_analysis + '\\' + folder + '\\' + cond_folder + '\\data\\stimlist_' + cond_folder + '.csv', index=False,
                    header=True)  # scat_plot
    print('data stored')
    return EEG_resp, stimlist


def get_IO_summary(LL_mean, cond_sel, labels_all, labels_region, path_patient):
    cond1 = 'Condition'  # 'condition', 'h'
    cond_folder = 'Ph'  # 'Ph', 'Sleep', 'CR'
    Condition = 'Condition'
    if cond_sel == 'Hour':
        Condition = 'Hour'
        cond_folder = 'CR'
    data_mean = np.zeros((1, 9))
    data_test = LL_mean[LL_mean.LL > 0]  # no artefacts
    stims = np.unique(data_test.Stim)
    Int_all = np.unique(data_test.Int)
    for sc in stims:  # repeat for each stimulation channel
        sc = np.int64(sc)
        resps = np.unique(data_test.loc[(data_test.Stim == sc), 'Chan'])
        for rc in resps:
            rc = np.int64(rc)
            LL0 = np.min((data_test.loc[(data_test.Stim == sc) & (data_test.Chan == rc), 'LL norm']).values)
            cond_val = np.unique(data_test[cond_sel])

            for day in np.unique(data_test.Date):
                for j in range(len(cond_val)):
                    dati = data_test[(data_test.Date == day) &
                                     (data_test.Stim == sc) & (data_test.Chan == rc) & (
                                                 data_test[Condition] == cond_val[j])]
                    if len(dati) > 3:
                        val = np.zeros((1, 9))
                        val[0, 0] = rc  # response channel
                        val[0, 1] = sc
                        val[0, 6] = cond_val[j]  # condition
                        val[0, 4] = np.nanmean(dati.d)  # distance
                        val[0, 8] = day  # date
                        val[0, 2] = trapz(dati['LL norm'].values - LL0, dati['Int'].values) / np.max(Int_all)  # AUC
                        Int_min = np.unique(dati.loc[dati.Sig == 1, 'Int'])  # all ints inducing CCEP
                        if len(np.unique(dati.loc[dati.Sig == 0, 'Int'])) > 0:  # if not all int inducing CCEPs
                            # only int with higher int also inducing CCEP
                            Int_min = np.unique(dati.loc[dati.Sig == 1, 'Int'])[
                                np.where(Int_min - np.unique(dati.loc[dati.Sig == 0, 'Int'])[-1] > 0)]
                        if (len(Int_min) > 0) and (np.mean(dati.loc[dati.Int > Int_all[-3], 'Sig']) == 1):

                            Int_min = Int_min[0]
                            val[0, 3] = Int_min
                            val[0, 5] = dati.loc[dati.Int == Int_min, 'LL norm'].values
                            val[0, 7] = 1
                            data_mean = np.concatenate((data_mean, val), axis=0)
                        else:
                            Int_min = 0
                            val[0, 3] = np.nan
                            val[0, 5] = 1
                            val[0, 7] = 0

    data_mean = data_mean[1:-1, :]  # remove first row (dummy row)
    IO_mean = pd.DataFrame(
        {"RespC": data_mean[:, 0], "StimC": data_mean[:, 1], "AUC": data_mean[:, 2], "MPI": data_mean[:, 3],
         "d": data_mean[:, 4], Condition: data_mean[:, 6], "Date": data_mean[:, 8], "Sig": data_mean[:, 7]})

    for c in range(len(labels_all)):
        IO_mean.loc[(IO_mean.RespC == c), "RespR"] = labels_region[c]
        IO_mean.loc[(IO_mean.RespC == c), "Resp"] = labels_all[c]
        IO_mean.loc[(IO_mean.StimC == c), "StimR"] = labels_region[c]
        IO_mean.loc[(IO_mean.StimC == c), "Stim"] = labels_all[c]

    IO_mean = IO_mean.drop(IO_mean[IO_mean['RespR'] == 'OUT'].index)
    # IO_mean=IO_mean.drop(IO_mean[IO_mean['RespR']=='WM'].index)
    file = path_patient + '/Analysis/InputOutput/' + cond_folder + '/data/IO_mean_' + Condition + '.csv'
    IO_mean.to_csv(file, index=False,
                   header=True)  # scat_plot = scat_plot.fillna(method='ffill')
    print('data stored -- ' + path_patient + '/Analysis/InputOutput/LL/data/IO_mean_' + Condition + '.csv')
    return IO_mean
