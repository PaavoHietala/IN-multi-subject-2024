import sys
import os.path as op
import mne
import numpy as np

sys.path.append(op.dirname(op.dirname(op.realpath(__file__))))
from Core.utils import get_fname, dense_to_sparse_idx

# A script for computing the average distance between the target vertex in the
# middle of a hemisphere (sector 9 and 13) and other vertices on the same hemi.

subject = 'fsaverage'
src_spacing = 'ico4'
subjects_dir = mne.get_config('SUBJECTS_DIR')
project_dir = '/m/nbe/scratch/megci/MFinverse/reMTW/'

#-------------------------------------------------------------------------------

fname_src = get_fname(subject, 'src', src_spacing = src_spacing)
src = mne.read_source_spaces(op.join(project_dir, 'Data', 'src', fname_src),
                             verbose = False)

dense = [mne.surface.read_surface(op.join(subjects_dir, 'fsaverage',
                                          'surf', 'lh.white'),
                                  return_dict = True),
         mne.surface.read_surface(op.join(subjects_dir, 'fsaverage',
                                          'surf', 'rh.white'),
                                  return_dict = True)]

distances = []
for hemi_idx, hemi in enumerate(['lh', 'rh']):
    targets = np.genfromtxt(op.join(project_dir,
                                    'MFMEG_stimorder_vertices_rh_lh.txt'),
                            dtype='int')
    
    # Right column in target txt file is for left hemisphere and vice versa
    targets = [t for t in targets[:, 0 if hemi_idx == 1 else 1] if t != -1]

    distances.append(np.zeros(len(targets)))

    # Pick the centermost target vertex (sector 9 for rh and and 13 for lh)
    center = 53176 if hemi_idx == 1 else 53425

    center_sparse = dense_to_sparse_idx(src[hemi_idx],
                                        dense[hemi_idx], 
                                        center)

    # Compute average geodesic distance between target vertices
    for target_idx, target in enumerate(targets):
        target = dense_to_sparse_idx(src[hemi_idx],
                                     dense[hemi_idx], 
                                     targets[target_idx])
        dist = src[hemi_idx]['dist'][center_sparse, target] * 1000

        distances[hemi_idx][target_idx] = dist

agg = np.concatenate(([d for d in distances[0] if d != 0], 
                      [d for d in distances[1] if d != 0]))
print(agg)
print(f'Grand mean: {np.mean(agg):.1f}, median: {np.median(agg):.1f},'
      + f'std: {np.std(agg):.1f}, min: {np.min(agg):.1f},'
      + f'max: {np.max(agg):.1f}')
