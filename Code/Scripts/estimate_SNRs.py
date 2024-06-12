'''
This script estimates the sensor and source-space signal-to-noise ratios of
each evoked response based on the eLORETA inverse operators. 
'''

import mne
import numpy as np

basedir = '/m/nbe/scratch/megci/MFinverse'

exclude = [5, 8, 13, 15]
subjects = [f'MEGCI_S{id}' for id in list(range(1, 25)) if id not in exclude]

evokeds = [basedir + '/Classic/Data/Evoked/' + subject + '_f-ave.fif' for
           subject in subjects]

invs = [basedir + '/Classic/Data/inv/' + subject + '-ico4-rest1-inv.fif' for
        subject in subjects]

fwds = [basedir + '/Classic/Data/fwd/' + subject + '-ico4-fwd.fif' for
        subject in subjects]

covs = [basedir + '/Classic/Data/cov/' + subject + '-rest1-cov.fif' for
        subject in subjects]

stcs = [[f'{basedir}/Classic/Data/stc/{sub}-ico4-eLORETA-f-sector{stim}-lh.stc'
         for stim in range(1, 25)] for sub in subjects]

rest_raws = ['/m/nbe/scratch/megci/data/MEG/megci_rawdata_mc_ic/' + s.lower()
             + '_mc/rest1_raw_tsss_mc_transOHP_blinkICremoved.fif' for s in subjects]

timing_fpath = '/m/nbe/scratch/megci/MFinverse/Classic/Data/plot/V1_medians_evoked.csv'

timing = np.loadtxt(timing_fpath).tolist()

PeakSNRs_sensor = []
PeakSNRs_source = []
SNRs = []

for i in range(len(subjects)):
    s_snrs = []
    s_PeakSNRs_sensor = []
    s_PeakSNRs_source = []
    
    subject_evokeds = mne.read_evokeds(evokeds[i], verbose = 50)
    subject_inv = mne.minimum_norm.read_inverse_operator(invs[i], verbose = 50)
    
    t = timing[i]
    print(f"Computing SNRs for {subjects[i]} at {t} s")

    for e in subject_evokeds:
        s_snrs.append(mne.minimum_norm.estimate_snr(e.copy().crop(0., 0.3),
                                                    subject_inv,
                                                    verbose = 50)[0])
        s_PeakSNRs_sensor.append(mne.minimum_norm.estimate_snr(e.copy().crop(t, t),
                                                               subject_inv,
                                                               verbose = 50)[0])
    
    fwd = mne.read_forward_solution(fwds[i], verbose = 50)
    cov = mne.read_cov(covs[i], verbose = 50)
    raw = mne.io.read_raw(rest_raws[i], verbose = 50)
    for stc_path in stcs[i]:
        stc = mne.read_source_estimate(stc_path)
        s_PeakSNRs_source.append(stc.estimate_snr(raw.info, fwd, cov, verbose = 50).data)
    
    #print(f"{subjects[i]} SNRs: {subject_snrs}")
    print(f"Average SNR for {subjects[i]}: {np.mean(s_snrs)}")
    print(f"Peak SNR for {subjects[i]}: {np.mean(s_PeakSNRs_sensor)}")
    print(f"Source space SNR for {subjects[i]}: "
          + f"Mean: {10**(0.1*np.mean(s_PeakSNRs_source))}, "
          + f"peak: {10**(0.1*np.max(s_PeakSNRs_source))}\n")

    SNRs.append(s_snrs)
    PeakSNRs_sensor.append(s_PeakSNRs_sensor)
    PeakSNRs_source.append(s_PeakSNRs_source)

peakSourceSNRs = [10**(0.1*np.max(s_SNRs)) for s_SNRs in PeakSNRs_source]

print(f"Mean SNR: {np.mean(SNRs)}\nMedian: {np.median(SNRs)}\nSTD: {np.std(SNRs)}\n")
print(f"Mean Peak SNR: {np.mean(PeakSNRs_sensor)}\nMedian: "
      + f"{np.median(PeakSNRs_sensor)}\nSTD: {np.std(PeakSNRs_sensor)}")
