import numpy as np
import pandas as pd
from general_funcs import freq_funcs as ff
from general_funcs import basic_func as bf
from general_funcs import LL_funcs as LLf
from general_funcs import CCEP_func
from scipy.spatial import distance


def init_stimlist_columns(stimlist, StimChanSM):
    """Initialize required columns if they are not present."""
    column_defaults = {"Num_block": stimlist.StimNum, "condition": 0, "sleep": 0}
    for col, default_val in column_defaults.items():
        if col not in stimlist.columns:
            stimlist[col] = default_val
    # Filter stimlist based on conditions
    stim_spec = stimlist[(stimlist.IPI_ms == 0) & (np.isin(stimlist.ChanP, StimChanSM)) & (stimlist.noise == 0)]
    stim_spec.reset_index(drop=True, inplace=True)

    return stimlist, stim_spec


def calculate_artefact(resps, stimlist, stim_spec, t_0, Fs, c, ChanP1, StimChanSM, StimChanIx, labels_clinic):
    """Detect artefact if recording channel has high LL and was stimulating the trial before (still recovering)"""

    return bf.check_inStimChan(c, ChanP1, labels_clinic)  # stim channel is recording channel


def LL_BM_connection(EEG_resp, stimlist, bad_chans, coord_all, labels_clinic, StimChanSM, StimChanIx):
    """Caulculate for each possible conenction the LL and keeps important factors (Time of Stim, distance, etc.)"""
    Fs = 500
    w_LL = 0.25
    t_0 = 1  # time of stimulation in data

    # Init required columns in stimlist
    stimlist, stim_spec = init_stimlist_columns(stimlist, StimChanSM)

    # Analyze each channel
    data_CCEP = []
    stimNum = stim_spec.StimNum.values  # [:,0]
    noise_val = stim_spec.noise.values  # [:,0]
    stimNum_block = stim_spec.Num_block.values  # [:,0]
    resps = ff.lp_filter(EEG_resp[:, stimNum, :], 45, Fs)
    ChanP1 = bf.SM2IX(stim_spec.ChanP.values, StimChanSM, np.array(StimChanIx))

    ## ge tLL for each stimulation and channel
    LL_trial = LLf.get_LL_all(resps[:, :, int((t_0 - 0.55) * Fs):int((t_0 - 0.05) * Fs)], Fs, w_LL)
    LL_peak_BL = np.max(LL_trial, 2)
    LL_trial = LLf.get_LL_all(resps[:, :, int(t_0 * Fs):int((t_0 + 0.5) * Fs)], Fs, w_LL)
    LL_peak = np.max(LL_trial, 2)
    t_peak = np.argmax(LL_trial, 2) + int((t_0 - w_LL / 2) * Fs)
    t_peak[t_peak < (t_0 * Fs)] = t_0 * Fs
    inds = np.repeat(np.expand_dims(t_peak, 2), int(w_LL * Fs), 2)
    inds = inds + np.arange(int(w_LL * Fs))
    pN = np.min(np.take_along_axis(resps, inds, axis=2), 2)
    pP = np.max(np.take_along_axis(resps, inds, axis=2), 2)
    p2p = abs(pP - pN)
    pN = np.min(resps[:, :, 0:int(t_0 * Fs)], 2)
    pP = np.max(resps[:, :, 0:int(t_0 * Fs)], 2)
    p2p_BL = abs(pP - pN)
    for c in range(LL_peak.shape[0]):
        val = [
            [c, ChanP1[i], noise_val[i], stimNum_block[i], stim_spec.condition.values[i], stim_spec.date.values[i],
             stim_spec.sleep.values[i], stim_spec.stim_block.values[i], LL_peak[c, i], LL_peak_BL[c, i],
             stim_spec.h.values[i],
             stimNum[i], p2p[c, i], p2p_BL[c, i]]
            for i in range(LL_peak.shape[1])
        ]
        val = np.array(val)
        chan_stimulating = bf.check_inStimChan(c, ChanP1, labels_clinic)
        if len(chan_stimulating) > 0:
            indices = np.where(chan_stimulating == 1)[0]
            val[indices, 2] = 1

        # Convert the numpy array back to a list
        val = val.tolist()
        data_CCEP.extend(val)

    # Convert to DataFrame
    LL_CCEP = pd.DataFrame(data_CCEP, columns=["Chan", "Stim", "Artefact", "Num_block", "Condition", "Date", "Sleep",
                                               "Block", "LL", "LL_BL", "Hour", "Num", "P2P", "P2P_BL"])

    # Mark bad channels as artefacts
    LL_CCEP.loc[LL_CCEP['Chan'].isin(bad_chans), 'Artefact'] = 1

    # distance
    for s in np.unique(LL_CCEP.Stim):
        s = np.int64(s)
        for c in np.unique(LL_CCEP.Chan):
            c = np.int64(c)
            LL_CCEP.loc[(LL_CCEP.Stim == s) & (LL_CCEP.Chan == c), 'd'] = np.round(
                distance.euclidean(coord_all[s], coord_all[c]), 2)

    return LL_CCEP  # , trial_sig


def cal_correlation_condition(con_trial, con_summary, metric='LL', condition='Block'):
    """Create correlation matrix between the BM during different conditions (Block, Hour, Sleep) based on conenction strength metric (LL, P)"""
    con_summary = con_summary[
        (con_summary.Sig > 0) & (con_summary.Num_sig_trial > 10) & (con_summary.True_peak == 1)].reset_index(drop=True)
    con_summary['Con_ID'] = con_summary.groupby(['Stim', 'Chan']).ngroup()
    # Clean table
    con_trial_cleaned = con_trial[(con_trial.Sig > -1) & (con_trial.Artefact < 1)].copy()
    con_trial_cleaned.loc[con_trial_cleaned.Sig == 0, 'LL'] = 0
    con_trial_cleaned = con_trial_cleaned.merge(con_summary[['Stim', 'Chan', 'Con_ID']], on=['Stim', 'Chan'])
    stim_all = np.unique(con_trial_cleaned.Stim)
    for b in np.unique(con_trial_cleaned[condition]):
        stim_block = np.unique(con_trial_cleaned.loc[con_trial_cleaned[condition] == b, 'Stim'])
        stim_all = np.intersect1d(stim_all, stim_block)
    con_trial_cleaned = con_trial_cleaned[np.isin(con_trial_cleaned.Stim, stim_all)].reset_index(drop=True)

    # con_trial_cleaned.loc[con_trial_cleaned.Sig < 0, 'Sig'] = np.nan

    # # Calculate mean for significant trials
    # con_trial_cleaned['m_sig'] = np.nan
    # con_trial_cleaned.loc[con_trial_cleaned.Sig == 1, 'm_sig'] = con_trial_cleaned.loc[
    #     con_trial_cleaned.Sig == 1, metric]
    # con_trial_cleaned['Prob'] = con_trial_cleaned.Sig

    # Create a pivot table
    con_pivot = con_trial_cleaned.pivot_table(index=['Con_ID'], columns=condition, values=['LL', 'Sig'],
                                              aggfunc='mean')

    # Fill missing values with global mean
    # con_pivot = con_pivot.fillna(con_trial_cleaned[metric].mean())

    V = con_pivot[metric].values
    if np.isnan(np.max(V)):
        num_cols = V.shape[1]
        correlation_matrix = np.zeros((num_cols, num_cols))
        # Calculate correlations, excluding NaN values in pairwise comparisons
        for i in range(num_cols):
            for j in range(i, num_cols):  # Start from i to avoid recalculating mirrored values
                if i == j:
                    correlation_matrix[i, j] = 1  # Correlation of a column with itself is always 1
                    continue
                x, y = V[:, i], V[:, j]
                valid_mask = ~np.isnan(x) & ~np.isnan(y)  # Mask to exclude NaNs for both columns
                if np.any(valid_mask):  # Check if there are any valid (non-NaN) comparisons
                    corr = np.corrcoef(x[valid_mask], y[valid_mask])[0, 1]
                else:
                    corr = np.nan  # If no valid comparisons, result is undefined
                correlation_matrix[i, j] = correlation_matrix[j, i] = corr
    else:
        correlation_matrix = np.corrcoef(V, rowvar=False)

    return correlation_matrix, np.unique(con_trial_cleaned[condition])



def get_con_summary_SS(con_trial, con_summary_gen, CC_summ, EEG_resp, sleep='Wake', delay=0):
    """Create summary table of each conenction showing mean response strength, probability, DI, distance and delay"""
    sig_con = con_summary_gen[(con_summary_gen.Sig > 0.029465) & (con_summary_gen.True_peak == 1)].reset_index(
        drop=True)
    sig_con['Sig_true'] = 1
    # Clean table
    con_trial_cleaned = con_trial[
        (con_trial.SleepState == sleep) & (con_trial.Sig > -1) & (con_trial.Artefact < 1)
        ].copy()

    # Calculate weighted LL (LL when non-significant trials are set to 0)
    con_trial_cleaned['LL_w'] = con_trial_cleaned['Sig'] * con_trial_cleaned['LL']

    # Define a custom aggregation dictionary
    aggregation = {
        'LL': ['mean', 'var'],  # Mean and variance of LL
        'LL_w': 'mean',  # Mean of weighted LL (LL when non-significant trials are set to 0)
        'Sig': 'mean',  # Mean significance
        'd': 'mean'  # Mean significance
    }

    # Calculate metrics for all trials
    con_summary = con_trial_cleaned.groupby(['Stim', 'Chan']).agg(aggregation)

    # Flatten MultiIndex columns
    con_summary.columns = ['_'.join(col).strip() for col in con_summary.columns.values]

    # Calculate metrics for significant trials only
    sig_trials = con_trial_cleaned[con_trial_cleaned.Sig == 1].reset_index(drop=True)
    sig_metrics = sig_trials.groupby(['Stim', 'Chan'])['LL'].agg(['mean', 'var'])

    # Rename columns for clarity
    sig_metrics = sig_metrics.rename(columns={'mean': 'LL_sig_mean', 'var': 'LL_sig_var'})

    # Merge the two summaries
    final_summary = pd.merge(con_summary, sig_metrics, on=['Stim', 'Chan'], how='left')
    final_summary = final_summary.rename(columns={'d_mean': 'd', 'Sig_mean': 'Sig'})

    # con_summary = con_summary[(con_summary.Sig >0)]
    # CC_summ = CC_summ[(CC_summ.sig == 1)]
    # CC_summ = CC_summ.groupby(['Stim', 'Chan'], as_index=False)[['t_WOI']].mean()
    final_summary = pd.merge(final_summary, con_summary_gen[['Stim', 'Chan', 't_WOI']], on=['Stim', 'Chan'],
                             how='outer')

    final_summary = final_summary.merge(sig_con[['Stim', 'Chan', 'Sig_true']], on=['Stim', 'Chan'],
                                        how='left').reset_index(drop=True)
    final_summary.loc[np.isnan(final_summary.Sig_true), 'Sig'] = 0
    final_summary = final_summary.drop(columns=['Sig_true'])
    # con_summary.insert(4, 'DI', np.nan)  # asym[rc, sc, 1]
    ## adding DI value
    # Ensure that for every A->B, B->A also exists in the dataframe
    df = final_summary.copy()
    # Create a DataFrame with pairs in both directions
    all_pairs = df[['Stim', 'Chan', 'Sig']]
    all_pairs['real'] = 1
    reverse_pairs = df.rename(columns={'Stim': 'Chan', 'Chan': 'Stim'})[['Stim', 'Chan', 'Sig']]
    all_pairs = pd.concat([all_pairs, reverse_pairs]).reset_index(drop=True)

    # Group by Stim and Chan
    grouped = all_pairs.groupby(['Stim', 'Chan'])

    # Apply the calculate_di function to each group
    di_results = grouped.apply(calculate_di).reset_index()
    di_results.rename(columns={0: 'DI'}, inplace=True)

    # Drop duplicates from all_pairs and merge with di_results
    all_pairs = all_pairs.merge(di_results, on=['Stim', 'Chan'], how='left')
    # Merge the DI back to the original dataframe
    con_summary = df.merge(all_pairs[['Stim', 'Chan', 'DI', 'Sig']], on=['Stim', 'Chan', 'Sig'], how='left')
    con_summary.insert(2, 'SleepState', sleep)
    con_summary = con_summary.drop_duplicates().reset_index(drop=True)
    con_summary = get_averaged_measurements(con_summary, CC_summ, con_trial_cleaned, EEG_resp)
    if delay:
        con_summary.delay = np.nan
        con_summary = get_delay_finetuned(con_summary, con_summary_gen, CC_summ, con_trial_cleaned, EEG_resp)
    return con_summary


def get_averaged_measurements(con_summary, CC_summ, con_trial_cleaned, EEG_resp, t0=1, Fs=500):
    # add delay
    for i in range(len(con_summary[con_summary.Sig > 0])):
        sc = con_summary.loc[con_summary.Sig > 0, 'Stim'].values[i]
        rc = con_summary.loc[con_summary.Sig > 0, 'Chan'].values[i]

        # only significant trials
        num = con_trial_cleaned.loc[(con_trial_cleaned.Stim == sc) & (con_trial_cleaned.Chan == rc) & (
                con_trial_cleaned.Sig == 1), 'Num'].values
        # all trials (to compare to old method)
        num_z = con_trial_cleaned.loc[(con_trial_cleaned.Stim == sc) & (con_trial_cleaned.Chan == rc), 'Num'].values
        WOI = CC_summ.loc[(CC_summ.Stim == sc) & (CC_summ.Chan == rc), 't_WOI'].values
        if len(WOI) > 0:
            WOI = WOI[0]
        else:
            WOI = 0
        signal = np.nanmean(EEG_resp[rc, num, :], 0)
        P2P = np.max(signal[int(1 * Fs):int((t0 + WOI + 0.25) * Fs)]) - np.min(
            signal[int(t0 * Fs):int((t0 + WOI + 0.25) * Fs)])
        LL_transform = LLf.get_LL_all(np.expand_dims(signal, [0, 1]), Fs, 0.25)[0, 0]
        LL_pk = np.max(LL_transform[int((t0 - 0.125) * Fs):int((t0 + WOI + 0.150) * Fs)])
        _, peak_latency, _, peak_detected = CCEP_func.CCEP_onset(EEG_resp[rc, num_z, :], WOI=WOI)
        zscore = CCEP_func.zscore_CCEP(np.nanmean(EEG_resp[rc, num_z, :], 0), t_0=t0, w0=0.2, w1=0.02, Fs=500)
        con_summary.loc[(con_summary.Stim == sc) & (con_summary.Chan == rc), 'True_peak'] = peak_detected
        con_summary.loc[(con_summary.Stim == sc) & (con_summary.Chan == rc), 'peak_latency'] = peak_latency

        con_summary.loc[(con_summary.Stim == sc) & (con_summary.Chan == rc), 'Num_sig_trial'] = len(num)
        con_summary.loc[(con_summary.Stim == sc) & (con_summary.Chan == rc), 'Mean_sig_P2P'] = P2P
        con_summary.loc[(con_summary.Stim == sc) & (con_summary.Chan == rc), 'Mean_sig_LL'] = LL_pk
        con_summary.loc[(con_summary.Stim == sc) & (con_summary.Chan == rc), 'Zscore'] = np.max(
            abs(zscore[int(Fs * (t0 + 0.05)):int(Fs * (t0 + 0.5))]))

    return con_summary


# Define a function to calculate DI
def calculate_di(group, DI_metric='ratio'):
    if group.shape[0] == 2:
        sig_values = group['Sig'].values
        if sig_values[0] == 0 and sig_values[1] == 0:
            return np.nan
        elif sig_values[0] == sig_values[1]:
            return 0
        else:
            if DI_metric == 'ratio':
                return (sig_values[0] - sig_values[1]) / max(sig_values)
            elif DI_metric == 'diff':
                return sig_values[0] - sig_values[1]
            else:
                abs(sig_values[0] - sig_values[1]) * ((sig_values[0] - sig_values[1]) / max(sig_values))
    else:
        return np.nan


def add_ntrials(con_trial, con_summary):
    """Add the number of single trials for each connections"""
    if "Artefact" in con_summary:
        con_summary = con_summary.drop(columns=["Artefact"]).reset_index(drop=True)
    if "n_trials" in con_summary:
        con_summary = con_summary.drop(columns=["n_trials"]).reset_index(drop=True)
    # Clean table
    con_trial_cleaned = con_trial[(con_trial.Sig > -1) & (con_trial.Artefact < 1)].copy()
    # Group by columns 'A' and 'B' and count occurrences in 'C'
    result = con_trial_cleaned.groupby(['Stim', 'Chan'], as_index=False)['Artefact'].count()
    # Rename the count column for clarity
    result.rename(columns={'Artefact': 'n_trials'}, inplace=True)
    con_summary = con_summary.merge(result, on=['Stim', 'Chan'], how='left').reset_index(drop=True)
    return con_summary

def add_ntrials_ss(con_trial, con_summary, ss ='Wake'):
    """Add the number of single trials for each connections"""
    if "Artefact" in con_summary:
        con_summary = con_summary.drop(columns=["Artefact"]).reset_index(drop=True)
    if "Num_trial" in con_summary:
        con_summary = con_summary.drop(columns=["Num_trial"]).reset_index(drop=True)
    # Clean table
    con_trial_cleaned = con_trial[(con_trial.Sig > -1) & (con_trial.Artefact < 1)& (con_trial.SleepState == ss)].copy()
    # Group by columns 'A' and 'B' and count occurrences in 'C'
    result = con_trial_cleaned.groupby(['Stim', 'Chan'], as_index=False)['Artefact'].count()
    # Rename the count column for clarity
    result.rename(columns={'Artefact': 'Num_trial'}, inplace=True)
    con_summary = con_summary.merge(result, on=['Stim', 'Chan'], how='left').reset_index(drop=True)
    return con_summary


def calculate_metric_diff(group, metric='peak_latency'):
    if group.shape[0] == 2:
        sig_values = group[metric].values
        if np.isnan(np.min([sig_values[0], sig_values[1]])):
            return np.nan
        else:
            return sig_values[0] - sig_values[1]
    else:
        return np.nan


def add_metric_diff_direction(con_summary, metric='Sig'):
    new_metric = metric + '_diff'
    if new_metric in con_summary:
        con_summary = con_summary.drop(columns=[new_metric]).reset_index(drop=True)

    """Similar to DI, add the difference in delay, probability or whatever metric AB, BA"""
    # Ensure that for every A->B, B->A also exists in the dataframe
    if metric == 'Sig':
        df = con_summary[con_summary.Sig > -1].reset_index(drop=True)
    else:
        df = con_summary[con_summary.Sig > 0].reset_index(drop=True)
    # Create a DataFrame with pairs in both directions
    all_pairs = df[['Stim', 'Chan', metric]]
    all_pairs['real'] = 1
    reverse_pairs = df.rename(columns={'Stim': 'Chan', 'Chan': 'Stim'})[['Stim', 'Chan', metric]]
    all_pairs = pd.concat([all_pairs, reverse_pairs]).reset_index(drop=True)

    # Group by Stim and Chan
    grouped = all_pairs.groupby(['Stim', 'Chan'])

    # Apply the function to each group
    di_results = grouped.apply(lambda group: calculate_metric_diff(group, metric=metric)).reset_index()

    di_results.rename(columns={0: new_metric}, inplace=True)

    # Drop duplicates from all_pairs and merge with di_results
    all_pairs = all_pairs.merge(di_results, on=['Stim', 'Chan'], how='left')
    # Merge the DI back to the original dataframe
    con_summary = con_summary.merge(all_pairs[['Stim', 'Chan', new_metric, metric]], on=['Stim', 'Chan', metric],
                                    how='left')
    return con_summary


def add_delay_diff(con_summary, metric='peak_latency'):
    if "delay_diff" in con_summary:
        con_summary = con_summary.drop(columns=["delay_diff"]).reset_index(drop=True)

    """Similar to DI, add the difference in delay AB, BA"""
    # Ensure that for every A->B, B->A also exists in the dataframe
    df = con_summary[con_summary.Sig > 0].reset_index(drop=True)
    # Create a DataFrame with pairs in both directions
    all_pairs = df[['Stim', 'Chan', metric]]
    all_pairs['real'] = 1
    reverse_pairs = df.rename(columns={'Stim': 'Chan', 'Chan': 'Stim'})[['Stim', 'Chan', metric]]
    all_pairs = pd.concat([all_pairs, reverse_pairs]).reset_index(drop=True)

    # Group by Stim and Chan
    grouped = all_pairs.groupby(['Stim', 'Chan'])

    # Apply the calculate_di function to each group
    di_results = grouped.apply(calculate_delay_diff).reset_index()
    di_results.rename(columns={0: 'delay_diff'}, inplace=True)

    # Drop duplicates from all_pairs and merge with di_results
    all_pairs = all_pairs.merge(di_results, on=['Stim', 'Chan'], how='left')
    # Merge the DI back to the original dataframe
    con_summary = con_summary.merge(all_pairs[['Stim', 'Chan', 'delay_diff', metric]], on=['Stim', 'Chan', metric],
                                    how='left')
    return con_summary


def get_con_summary(con_trial, CC_summ, EEG_resp, run_BL=False):
    """Create summary table of each conenction showing mean response strength, probability, DI, distance and delay"""
    # Clean table
    if run_BL:
        # calculate response probability run on prestimulation period (baseline) as a control
        # no need to run DI, delay etc.
        # FDR
        con_trial['Sig_BL'] = 0
        if 'p_value_BL' in con_trial:
            req = (con_trial['Artefact'] < 1) & ~np.isnan(con_trial['p_value_BL'])
            valid_p_values = con_trial.loc[req, 'p_value_BL'].values

            con_trial.loc[req, 'Sig_BL'] = (valid_p_values >= 1 - 0.05).astype(int)

            # Ensure the 'Sig' column is numeric
            con_trial.loc['Sig_BL'] = pd.to_numeric(con_trial['Sig_BL'], errors='coerce')
        else:
            req = (con_trial['Artefact'] < 1)
            con_trial.loc[req, 'Sig_BL'] = 0

        con_trial_cleaned = con_trial[(con_trial.Sig_BL > -1) & (con_trial.Artefact < 1)].copy()
        con_summary = con_trial_cleaned.groupby(['Stim', 'Chan'], as_index=False)[[
            'Sig_BL']].mean()  # con_summary = con_trial_cleaned.groupby(['Stim', 'Chan'], as_index=False)['LL', 'LL_sig', 'd', 'Sig'].mean()
        CC_sig = CC_summ[(CC_summ.corr_CC > -0.9) & (CC_summ.p_val > 0.97)].reset_index(drop=True)
        CC_sig['Sig1'] = 1
        con_summary = con_summary.merge(CC_sig[['Stim', 'Chan', 'Sig1']], on=['Stim', 'Chan'], how='left').reset_index(
            drop=True)
        con_summary.loc[np.isnan(con_summary.Sig1), 'Sig_BL'] = 0
        con_summary = con_summary.drop(columns=['Sig1']).reset_index(
            drop=True)

    else:
        # 1. get mean Sig (response probability)
        # 2. get corresponding DI value for each connection
        # 3. calculate peak latency
        con_trial_cleaned = con_trial[(con_trial.Sig > -1) & (con_trial.Artefact < 1)].copy()
        con_trial_cleaned['LL_sig'] = con_trial_cleaned['Sig'] * con_trial_cleaned['LL']
        con_summary = con_trial_cleaned.groupby(['Stim', 'Chan'], as_index=False)[['LL', 'LL_sig', 'd',
                                                                                   'Sig']].mean()  # con_summary = con_trial_cleaned.groupby(['Stim', 'Chan'], as_index=False)['LL', 'LL_sig', 'd', 'Sig'].mean()
        CC_summ = CC_summ.groupby(['Stim', 'Chan'], as_index=False)[['t_WOI']].mean()
        con_summary = pd.merge(con_summary, CC_summ, on=['Stim', 'Chan'], how='outer')
        ## adding DI value
        # Ensure that for every A->B, B->A also exists in the dataframe
        df = con_summary.copy()
        # Create a DataFrame with pairs in both directions
        all_pairs = df[['Stim', 'Chan', 'Sig']]
        all_pairs['real'] = 1
        reverse_pairs = df.rename(columns={'Stim': 'Chan', 'Chan': 'Stim'})[['Stim', 'Chan', 'Sig']]
        all_pairs = pd.concat([all_pairs, reverse_pairs]).reset_index(drop=True)

        # Group by Stim and Chan
        grouped = all_pairs.groupby(['Stim', 'Chan'])

        # Apply the calculate_di function to each group
        di_results = grouped.apply(calculate_di).reset_index()
        di_results.rename(columns={0: 'DI'}, inplace=True)

        # Drop duplicates from all_pairs and merge with di_results
        all_pairs = all_pairs.merge(di_results, on=['Stim', 'Chan'], how='left')
        # Merge the DI back to the original dataframe
        con_summary = df.merge(all_pairs[['Stim', 'Chan', 'DI', 'Sig']], on=['Stim', 'Chan', 'Sig'], how='left')
        con_summary = con_summary.drop_duplicates().reset_index(drop=True)
        con_summary = get_delay(con_summary, CC_summ, con_trial_cleaned, EEG_resp)

    return con_summary


def get_con_summary_update(con_trial, CC_summ, EEG_resp, con_summary):
    """Create summary table of each conenction showing mean response strength, probability, DI, distance and delay"""
    con_summary = con_summary.drop(columns=['Sig', 'DI']).reset_index(drop=True)
    con_summary = con_summary.drop_duplicates().reset_index(drop=True)
    con_trial_cleaned = con_trial[(con_trial.Sig > -1) & (con_trial.Artefact < 1)].copy()
    con_trial_cleaned['LL_sig'] = con_trial_cleaned['Sig'] * con_trial_cleaned['LL']
    df = con_trial_cleaned.groupby(['Stim', 'Chan'], as_index=False)[['LL', 'LL_sig', 'd',
                                                                      'Sig']].mean()  # con_summary = con_trial_cleaned.groupby(['Stim', 'Chan'], as_index=False)['LL', 'LL_sig', 'd', 'Sig'].mean()
    CC_summ = CC_summ.groupby(['Stim', 'Chan'], as_index=False)[['t_WOI']].mean()
    df = pd.merge(df, CC_summ, on=['Stim', 'Chan'], how='outer')
    ## adding DI value
    # Create a DataFrame with pairs in both directions
    all_pairs = df[['Stim', 'Chan', 'Sig']]
    all_pairs['real'] = 1
    reverse_pairs = df.rename(columns={'Stim': 'Chan', 'Chan': 'Stim'})[['Stim', 'Chan', 'Sig']]
    all_pairs = pd.concat([all_pairs, reverse_pairs]).reset_index(drop=True)

    # Group by Stim and Chan
    grouped = all_pairs.groupby(['Stim', 'Chan'])

    # Apply the calculate_di function to each group
    di_results = grouped.apply(calculate_di).reset_index()
    di_results.rename(columns={0: 'DI'}, inplace=True)

    # Drop duplicates from all_pairs and merge with di_results
    all_pairs = all_pairs.merge(di_results, on=['Stim', 'Chan'], how='left')
    # Merge the DI back to the original dataframe
    df = df.merge(all_pairs[['Stim', 'Chan', 'DI', 'Sig']], on=['Stim', 'Chan', 'Sig'], how='left')
    con_summary = con_summary.merge(df[['Stim', 'Chan', 'DI', 'Sig']], on=['Stim', 'Chan'], how='left')
    con_summary = con_summary.drop_duplicates().reset_index(drop=True)

    return con_summary


def get_delay_finetuned(con_summary, con_summary_general, CC_summ, con_trial_cleaned, EEG_resp, t0=1, Fs=500):
    # add delay
    for i in range(len(con_summary[con_summary.Sig > 0])):
        sc = con_summary.loc[con_summary.Sig > 0, 'Stim'].values[i]
        rc = con_summary.loc[con_summary.Sig > 0, 'Chan'].values[i]

        # only significant trials
        num = con_trial_cleaned.loc[(con_trial_cleaned.Stim == sc) & (con_trial_cleaned.Chan == rc) & (
                con_trial_cleaned.Sig == 1), 'Num'].values
        # all trials (to compare to old method)
        num_z = con_trial_cleaned.loc[(con_trial_cleaned.Stim == sc) & (con_trial_cleaned.Chan == rc), 'Num'].values
        WOI = CC_summ.loc[(CC_summ.Stim == sc) & (CC_summ.Chan == rc), 't_WOI'].values
        if len(WOI) > 0:
            WOI = WOI[0]
        else:
            WOI = 0
        signal = np.nanmean(EEG_resp[rc, num, :], 0)
        # P2P = np.max(signal[int(1 * Fs):int((t0 + WOI + 0.25) * Fs)]) - np.min(
        #    signal[int(t0 * Fs):int((t0 + WOI + 0.25) * Fs)])
        #LL_transform = LLf.get_LL_all(np.expand_dims(signal, [0, 1]), Fs, 0.25)[0, 0]
        # LL_pk = np.max(LL_transform[int((t0 - 0.125) * Fs):int((t0 + WOI + 0.25) * Fs)])
        # signal = np.nanmean(EEG_resp[rc, num, :], 0)
        # delay, _, _, _ = CCEP_func.cal_delay(signal, WOI=WOI)
        # delay, peak_latency, polarity, peak_detected = CCEP_func.CCEP_onset(EEG_resp[rc, num, :], WOI=WOI)
        onset = \
            con_summary_general.loc[
                (con_summary_general.Stim == sc) & (con_summary_general.Chan == rc), 'delay'].values[0]
        peak_lat_general = \
            con_summary_general.loc[
                (con_summary_general.Stim == sc) & (con_summary_general.Chan == rc), 'peak_latency'].values[0]

        polarity = \
            con_summary_general.loc[
                (con_summary_general.Stim == sc) & (con_summary_general.Chan == rc), 'peak_polarity'].values[0]

        delay, peak_latency, peak_detected, onset_detected = CCEP_func.CCEP_onset_finetuning(
            ff.lp_filter(EEG_resp[rc, num_z, :], 45, Fs),
            onset,
            peak_lat_general, polarity, WOI=WOI, t0=1,
            Fs=500, w_LL=0.25)

        N1 = signal[int((t0 + peak_latency) * Fs)]
        # zscore = CCEP_func.zscore_CCEP(np.nanmean(EEG_resp[rc, num_z, :], 0), t_0=t0, w0=0.2, w1=0.05, Fs=500)
        con_summary.loc[(con_summary.Stim == sc) & (con_summary.Chan == rc), 'delay'] = delay
        con_summary.loc[(con_summary.Stim == sc) & (con_summary.Chan == rc), 'peak_latency'] = peak_latency
        # con_summary.loc[(con_summary.Stim == sc) & (con_summary.Chan == rc), 'Mean_P2P'] = P2P
        # con_summary.loc[(con_summary.Stim == sc) & (con_summary.Chan == rc), 'Mean_LL'] = LL_pk
        con_summary.loc[(con_summary.Stim == sc) & (con_summary.Chan == rc), 'N1'] = N1
        #con_summary.loc[(con_summary.Stim == sc) & (con_summary.Chan == rc), 'Zscore'] = np.max(
        #     abs(zscore[int(Fs * (t0 + 0.05)):int(Fs * (t0 + 0.5))]))
        con_summary.loc[(con_summary.Stim == sc) & (con_summary.Chan == rc), 'True_peak'] = peak_detected

    return con_summary


def get_delay(con_summary, CC_summ, con_trial_cleaned, EEG_resp, t0=1, Fs=500):
    # add delay
    for i in range(len(con_summary[con_summary.Sig > 0])):
        sc = con_summary.loc[con_summary.Sig > 0, 'Stim'].values[i]
        rc = con_summary.loc[con_summary.Sig > 0, 'Chan'].values[i]
        # only significant trials
        num = con_trial_cleaned.loc[(con_trial_cleaned.Stim == sc) & (con_trial_cleaned.Chan == rc) & (
                con_trial_cleaned.Sig == 1), 'Num'].values
        # all trials (to compare to old method)
        num_z = con_trial_cleaned.loc[(con_trial_cleaned.Stim == sc) & (con_trial_cleaned.Chan == rc), 'Num'].values
        WOI = CC_summ.loc[(CC_summ.Stim == sc) & (CC_summ.Chan == rc), 't_WOI'].values
        if len(WOI) > 0:
            WOI = WOI[0]
        else:
            WOI = 0
        signal = np.nanmean(EEG_resp[rc, num, :], 0)
        P2P = np.max(signal[int(1 * Fs):int((t0 + WOI + 0.25) * Fs)]) - np.min(
            signal[int(t0 * Fs):int((t0 + WOI + 0.25) * Fs)])
        LL_transform = LLf.get_LL_all(np.expand_dims(signal, [0, 1]), Fs, 0.25)[0, 0]
        LL_pk = np.max(LL_transform[int((t0 - 0.125) * Fs):int((t0 + WOI + 0.25) * Fs)])
        # signal = np.nanmean(EEG_resp[rc, num, :], 0)
        # delay, _, _, _ = CCEP_func.cal_delay(signal, WOI=WOI)
        _, peak_latency, polarity, peak_detected = CCEP_func.CCEP_onset(EEG_resp[rc, num_z, :], WOI=WOI)
        N1 = signal[int((t0 + peak_latency) * Fs)]
        zscore = CCEP_func.zscore_CCEP(np.nanmean(EEG_resp[rc, num_z, :], 0), t_0=t0, w0=0.2, w1=0.01, Fs=500)
        # zscore = np.nanmean(CCEP_func.zscore_CCEP(EEG_resp[rc, num_z, :], t_0=t0, w0=0.2, w1=0.02, Fs=500), 0)
        # con_summary.loc[(con_summary.Stim == sc) & (con_summary.Chan == rc), 'delay'] = delay
        # Peak summary
        con_summary.loc[(con_summary.Stim == sc) & (con_summary.Chan == rc), 'True_peak'] = peak_detected
        con_summary.loc[(con_summary.Stim == sc) & (con_summary.Chan == rc), 'peak_latency'] = peak_latency
        con_summary.loc[(con_summary.Stim == sc) & (con_summary.Chan == rc), 'peak_polarity'] = polarity
        con_summary.loc[(con_summary.Stim == sc) & (con_summary.Chan == rc), 'N1'] = N1
        # Average of significant trials
        con_summary.loc[(con_summary.Stim == sc) & (con_summary.Chan == rc), 'Num_sig_trial'] = len(num)
        con_summary.loc[(con_summary.Stim == sc) & (con_summary.Chan == rc), 'Num_all_trial'] = len(num_z)
        con_summary.loc[(con_summary.Stim == sc) & (con_summary.Chan == rc), 'Mean_P2P'] = P2P
        con_summary.loc[(con_summary.Stim == sc) & (con_summary.Chan == rc), 'Mean_LL'] = LL_pk


        con_summary.loc[(con_summary.Stim == sc) & (con_summary.Chan == rc), 'Zscore'] = np.max(
            abs(zscore[int(Fs * (t0 + 0.01)):int(Fs * (t0 + 0.5))]))
    return con_summary


def get_peaks_AUC(data, EEG_resp, t0=1, Fs=500):
    new_lab = ['N1', 'N2', 'AUC', 'P2P_1s']
    for l in new_lab:
        if l not in data:
            data.insert(6, l, np.nan)
    for sc in np.unique(data.Stim).astype('int'):
        for rc in np.unique(data.loc[(data.Artefact < 1) & (data.Stim == sc), 'Chan']).astype('int'):
            req = (data['Chan'] == rc) & (data['Stim'] == sc) & (data.Artefact < 1)
            lists = data[req].reset_index(drop=True)
            stimNum_all = lists.Num.values.astype('int')
            N1, N2, AUC, P2P = CCEP_func.CCEP_metric(EEG_resp[rc, stimNum_all, :], t0=t0, w_AUC=1, Fs=Fs)
            data.loc[req, 'N1'] = N1
            data.loc[req, 'N2'] = N2
            data.loc[req, 'AUC'] = AUC
            data.loc[req, 'P2P_1s'] = P2P

    return data
