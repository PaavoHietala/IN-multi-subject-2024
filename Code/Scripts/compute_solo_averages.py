'''
Script for computing the mean, median and std peak-target distances for solo
eLORETA subjects to check how representative a given subject is.
'''

import sys
import os.path as op
import mne
import numpy as np

sys.path.append(op.dirname(op.dirname(op.realpath(__file__))))
from Core.utils import tabulate_geodesics

src_spacing = 'ico4'
subjects_dir = mne.get_config('SUBJECTS_DIR')
project_dir = '/m/nbe/scratch/megci/MFinverse/Classic/'

exclude = [5, 8, 13, 15]
subjects = [f'MEGCI_S{id}' for id in list(range(1, 25)) if id not in exclude]

stimuli = [f'sector{num}' for num in range(1, 25)]

bilaterals = ['sector3', 'sector7', 'sector11',
              'sector15', 'sector19', 'sector23']

# Load V1 peak timing for all subjects
timing_fpath = project_dir + 'Data/plot/V1_medians_evoked.csv'
timing = np.loadtxt(timing_fpath).tolist()

# Restrict timings only to selected subjects
[timing.insert(i - 1, 0) for i in exclude]
timing = [t for i, t in enumerate(timing) if str(i + 1)
            in [sub[7:] for sub in subjects]]
print('Loaded timings: ', timing)

for s_idx, subject in enumerate(subjects):
    tabulate_geodesics(project_dir, src_spacing, 'eLORETA', 'f', stimuli,
                       bilaterals, '', counts = [1], mode = 'stc_m',
                       subject = subject, overwrite = True, time = timing[s_idx])

fnames = [f'distances_{s}_stc_m__24targets.csv' for s in subjects]
data = np.zeros([20, 30])
for idx, f in enumerate(fnames):
    fpath = op.join(project_dir, 'Data', 'plot', f)
    data[idx, :] = np.genfromtxt(fpath, delimiter=',')

print(f'Mean: {np.mean(data[np.isfinite(data)])} '
      + f'Median: {np.median(data[np.isfinite(data)])} '
      + f'std: {np.std(data[np.isfinite(data)])}')

rowavg = np.zeros(20)
rowmed = np.zeros(20)
rowstd = np.zeros(20)

for row_idx in range(20):
    row = data[row_idx, :]
    rowavg[row_idx] = np.mean(row[np.isfinite(row)])
    rowmed[row_idx] = np.median(row[np.isfinite(row)])
    rowstd[row_idx] = np.std(row[np.isfinite(row)])

avgidx = np.argmin(np.abs(rowavg - np.mean(data[np.isfinite(data)])))
medidx = np.argmin(np.abs(rowmed - np.median(data[np.isfinite(data)])))
stdidx = np.argmin(np.abs(rowstd - np.std(data[np.isfinite(data)])))

print(f'Closest mean is for subject: {subjects[avgidx]}\n'
      + f'Closest median is for subject: {subjects[medidx]}\n'
      + f'Closest std is for subject: {subjects[stdidx]}')
