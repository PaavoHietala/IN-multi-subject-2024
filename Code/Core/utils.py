#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Common functions used in multiple pipelines.

Created on Tue March 30 15:22:35 2021

@author: hietalp2
'''

import os
import mne
import numpy as np

def get_fname(subject, ftype, stc_method = None, src_spacing = None,
              fname_raw = None, task = None, stim = None, layers = 1,
              suffix = None):
    '''
    Create a project-standard filename from given parameters.
    
    Parameters
    ----------
    subject : str
        Subject name/identifier as in filenames.
    ftype : str
        File type for which a fname is generate, e.g. 'src', 'fwd', 'cov'.
    stc_method : str, optional
        Inversion method used in this file, e.g. 'dSPM', by default None.
    src_spacing : str, optional
        Source space scheme used in this file, e.g. 'oct6', by default None.
    fname_raw : str, optional
        File name of raw recording from which the info is extracted.
        By default None.
    task : str, optional
        Task in the evoked response, e.g. 'f', by default None.
    stim: str, optional
        Stimulus name, e.g. 'sector1', by default None.
    layers: str, optional
        How many layers the BEM model has, by default 1.
    suffix : str, optional
        A string to append to the end of the filename before file extension.
        By default None.

    Raises
    ------
    ValueError
        Error given if invalid ftype is given.

    Returns
    -------
    str
        File name built from given parameters.
    '''
    
    if ftype == 'src':
        return '-'.join(filter(None, [subject, src_spacing, suffix, 'src.fif']))
    elif ftype == 'fwd':
        return '-'.join(filter(None, [subject, src_spacing, suffix, 'fwd.fif']))
    elif ftype == 'cov':
        run = os.path.split(fname_raw)[-1].split('_')[0]
        return '-'.join(filter(None, [subject, run, suffix, 'cov.fif']))
    elif ftype == 'inv':
        run = os.path.split(fname_raw)[-1].split('_')[0]
        return '-'.join(filter(None, [subject, src_spacing, run, suffix, 'inv.fif']))
    elif ftype == 'stc':
        return '-'.join(filter(None, [subject, src_spacing, stc_method, task,
                                      stim, suffix]))
    elif ftype == 'stc_m':
        return '-'.join(filter(None, [subject, src_spacing, stc_method,
                                      'fsaverage', task, stim, suffix]))
    elif ftype == 'bem':
        return '-'.join(filter(None, [subject, layers, suffix, 'shell-bem-sol.fif']))
    elif ftype == 'label':
        return '-'.join(filter(None, [subject, src_spacing, stc_method, task,
                                      stim, suffix]))
    else:
        raise ValueError('Invalid file type ' + ftype)

def prepare_directories(project_dir):
    '''
    Prepare directories used in the pipelines in <project_dir>/*

    Parameters
    ----------
    project_dir : str
        Base directory to create the project folders in.
    
    Returns
    -------
    None.
    '''

    subdirs = ['fwd', 'src', 'inv', 'stc', 'stc_m', 'avg', 'cov', 'plot',
               'slurm_out']
    for dirname in ['Data'] + [os.path.join('Data', dir) for dir in subdirs]:
        try:
            os.makedirs(project_dir + dirname)
            print("Created dir", dirname)
        except FileExistsError:
            print("Directory", dirname, "already exists")

def average_stcs_source_space(subjects, project_dir, src_spacing, stc_method,
                              task, stimuli, suffix = None, timing = None,
                              overwrite = False):
    '''
    Average the source time courses that have been morphed to fsaverage and
    save them per-stimuli to <project_dir>/Data/avg/

    Parameters
    ----------
    subject : str
        Subject name/identifier as in filenames.
    project_dir : str
        Base directory of the project with Data subfolder.
    src_spacing : str
        Source space scheme used in this file, e.g. 'oct6'.
    stc_method : str
        Inversion method used, e.g. 'eLORETA'.
    task: str
        Task in the estimated stcs, e.g. 'f'.
    stimuli: list of str
        List of stimuli for whcih the stcs are estimated.
    suffix : str, optional
        Suffix to append to the end of the output filename before -ave.fif.
    timing : None or list, optional
        List of single timepoints per subject to which the estimate is confined
        to, by default False (average each timepoint)
    overwrite : bool, optional
        Overwrite existing files switch, by default False.

    Returns
    -------
    None.
    '''
        
    # Average each stimulus and save the avg stc to disk
    for stim in stimuli:
        fname = get_fname('fsaverage', 'stc', stc_method = stc_method, 
                          src_spacing = src_spacing, task = task, stim = stim,
                          suffix = suffix)
        fpath = os.path.join(project_dir, 'Data', 'avg', fname)
        
        if overwrite or not os.path.isfile(fpath + '-lh.stc'):
            # Load stcs for all subjects with this stimulus
            stcs = []
            for subject in subjects:
                fname_m = get_fname(subject, 'stc_m', stc_method = stc_method,
                                    src_spacing = src_spacing, task = task,
                                    stim = stim,
                                    suffix = (suffix if stc_method == 'remtw'
                                                     else None))
                fpath_m = os.path.join(project_dir, 'Data', 'stc_m', fname_m)
                stcs.append(mne.read_source_estimate(fpath_m))
            
            # Set the first stc as base and add all others to it, divide by n
            if timing == None:
                avg = stcs[0].copy()
                avg.data = abs(avg.data)
                for i in range(1, len(subjects)):
                    avg.data += abs(stcs[i].data)
            else:
                avg = stcs[0].crop(tmin = timing[0], tmax = timing[0],
                                   include_tmax = True).copy()
                avg.data = abs(avg.data)
                for i in range(1, len(subjects)):
                    avg.data += abs(stcs[i].crop(tmin = timing[i],
                                                 tmax = timing[i],
                                                 include_tmax = True).data)
            avg.data = avg.data / len(subjects)
            avg.save(fpath)

def find_peaks(project_dir, src_spacing, stc_method, task, stimuli, bilaterals,
               suffix, return_index = False, subject = 'fsaverage',
               mode = 'avg', stc = None, time = None):
    '''
    Find peak activation vertices and hemispheres from mne.SourceEstimate files.

    Parameters
    ----------
    project_dir : str
        Base directory of the project with Data subfolder.
    src_spacing : str
        Source space scheme used in this file, e.g. 'oct4'.
    stc_method : str
        Inversion method used, e.g. 'eLORETA'.
    task : str
        Task in the estimated stcs, e.g. 'f'.
    stimuli : list of str
        List of stimuli for whcih the stcs are estimated.
    bilaterals: list of str
        List of bilateral stimuli (on midline); label peaks on both hemis.
    suffix : str
        Suffix to append to stc filename before the common suffix and extension
        (e.g. '-avg.fif').
    return_index : bool, optional
        Whether to return peak vertex index instead of vertex ID, by default
        False.
    subject : str, optional
        Name of the subject the stc is for, by default 'fsaverage'.
    mode : str, optional
        Type of stc, can be either 'stc', 'stc_m' or 'avg', by default 'avg'
    stc : mne.SourceEstimate, optional
        If source estimate is given, skip loading it again, by default None.
    time : float, optional
        A timepoint for which to get the peak. If None, overall peak is used.
        By default None.
    
    Returns
    -------
    peaks : list of int
        List of peak valued vertex indices for each stimulus
    peak_hemis : list of str
        List of hemispheres for each peak index in same order as peaks
    '''

    peaks = []
    peak_hemis = []

    for stim in stimuli:
        if stc == None:
            fname = get_fname(('fsaverage' if mode == 'avg' else subject),
                            ('stc_m' if mode == 'stc_m' else 'stc'),
                            stc_method = stc_method, src_spacing = src_spacing,
                            task = task, stim = stim, suffix = suffix)
            fpath = os.path.join(project_dir, 'Data', mode, fname)
            print('Reading ' + fpath)
            stc = mne.read_source_estimate(fpath)
        
        if time != None and stc.data.shape[1] > 1:
            stc.crop(tmin = time, tmax = time)
        
        if stim not in bilaterals:
            # Not bilateral; get global peak and store it in peaks and hemis
            if np.max(abs(stc.lh_data)) > np.max(abs(stc.rh_data)):
                hemi = 'lh'
            else:
                hemi = 'rh'
            peak = stc.get_peak(hemi = hemi, vert_as_index = return_index)[0]
            print(f'Got peak for stimulus {stim} on {hemi} at index {peak}')
            peaks.append(peak)
            peak_hemis.append(hemi)
        else:
            # Bilateral; get both lh and rh peaks and store them
            peak_lh = stc.get_peak(hemi = 'lh', vert_as_index = return_index)[0]
            peak_rh = stc.get_peak(hemi = 'rh', vert_as_index = return_index)[0]
            print(f'Got peak for stimulus {stim} on lh at index {peak_lh} and',
                  f'on rh at {peak_rh}')

            peaks.append(peak_lh)
            peaks.append(peak_rh)
            peak_hemis += ['lh', 'rh']
        
        stc = None
    
    return (peaks, peak_hemis)

def dense_to_sparse_idx(sparse, dense, dense_idx):
    '''
    Convert dense FreeSurfer mesh vertex indices to sparse MNE source space
    indices. The conversion is based on finding the nearest XYZ vertex
    coordinate in the sparse space, given the XYZ coordinate of a dense mesh
    vertex.

    Parameters
    ----------
    sparse : mne.SourceSpaces
        Sparse MNE source space from mne.read_source_space
    dense : dict
        Dense MNE surface from mne.read_surface with return_dict=True.
        The surface can be e.g. FreeSurfer lh.white.
    
    Returns
    -------
    int
        Index of the sparse vertex corresponding to given vertex on dense surf
    '''

    sparse_idx = mne.surface._compute_nearest(sparse['rr'][:sparse['nuse']],
                                              np.array([dense[2]['rr'][dense_idx] / 1000]))
    
    print(f"Got sparse index {sparse_idx[0]} for dense index {dense_idx}")

    return sparse_idx[0]

def tabulate_geodesics(project_dir, src_spacing, stc_method, task, stimuli,
                       bilaterals, suffix, counts = [1, 5, 10, 15, 20],
                       subject = 'fsaverage', mode = 'avg', overwrite = False,
                       time = None):
    '''
    TODO fix docs (add time)
    Compute geodesic distances between peaks and the V1 label vertices and
    save them in a csv file. Each row is one subject count in <counts> and
    each column is a peak (two peaks for bilateral stimuli).

    Parameters
    ----------
    project_dir : str
        Base directory of the project with Data subfolder.
    src_spacing : str
        Source space scheme used in this file, e.g. 'oct6'.
    stc_method : str
        Inversion method used, e.g. 'dSPM'.
    task : str
        Task in the estimated stcs, e.g. 'f'.
    stimuli: list of str
        List of stimuli for whcih the stcs are estimated.
    bilaterals : list of str
        Names of bilateral stimuli
    suffix : str
        Suffix to append to the end of the output filename. Numbers are
        stripped from the beginning to allow suffix to be e.g. 10subjects.
    counts : list, optional
        List of subject counts to calculate the distances for,
        by default [1, 5, 10, 15, 20]
    subject : str, optional
        Subject for which the distances are calculated. If the mode is 'avg' or
        'stc_m', the subject is forced as 'fsaverage', by default 'fsaverage'
    mode : str, optional
        Stc type, either 'stc', 'stc_m' or 'avg', by default 'avg'
    overwrite : bool, optional
        Overwrite existing files switch, by default False.
    time : float, optional
        Time in seconds for which the geodesics are computed, if the source
        estimates have a timecourse. None if stcs have only one timepoint.

    Returns
    -------
    None.
    '''

    print(counts, stimuli, bilaterals)
    subjects_dir = mne.get_config('SUBJECTS_DIR')
    suffix = suffix.lstrip('0123456789')
    distances = np.zeros((len(counts), len(stimuli) + len(bilaterals)))
    out_path = os.path.join(project_dir, 'Data', 'plot',
                            f'distances_{subject}_{mode}_{suffix}_24targets.csv')

    if not overwrite and os.path.isfile(out_path):
        print(f"Output file {out_path} exists and overwrite = False, skipping.")
        return

    # Load target vertices and source space with distances
    lbl_subject = subject if mode == 'stc' else 'fsaverage'
    targets = np.genfromtxt(os.path.join(project_dir,
                                         'MFMEG_stimorder_vertices_rh_lh.txt'),
                            dtype='int')
    targets = [x for x in targets.flatten() if x != -1]
    print(targets)

    dense = [mne.surface.read_surface(os.path.join(subjects_dir, lbl_subject,
                                                   'surf', 'lh.white'),
                                      return_dict = True),
             mne.surface.read_surface(os.path.join(subjects_dir, lbl_subject,
                                                   'surf', 'rh.white'),
                                      return_dict = True)]

    fname_src = get_fname(lbl_subject, 'src', src_spacing = src_spacing)
    src = mne.read_source_spaces(os.path.join(project_dir, 'Data', 'src', fname_src),
                                 verbose = False)

    # Hemisphereres for the targets as they appear in the stimorder txt file
    target_hemi = [1, 1, 1, 0, 0, 0, 0, 1, 0, 1] * 3

    # Comparison list to get inf distance if the peak is on the wrong hemi
    expected_hemi = [side + 'h' for side in "rrlrllllrr"] * 3

    for n_idx, suffix in enumerate([str(n) + suffix for n in counts]):
        if mode == 'stc_m' and suffix == '1':
            suffix = None
        print(f'Solving geodesics for {suffix}')

        peaks, peak_hemis = find_peaks(project_dir, src_spacing, stc_method,
                                       task, stimuli, bilaterals, suffix,
                                       return_index = False, subject = subject,
                                       mode = mode, time = time)
        
        # Count average geodesic distance between peaks and target vertices on V1
        for peak_idx, peak in enumerate(peaks):
            hemi_idx = 0 if peak_hemis[peak_idx] == 'lh' else 1
            if peak_hemis[peak_idx] == expected_hemi[peak_idx]:
                target = dense_to_sparse_idx(src[hemi_idx],
                                             dense[target_hemi[peak_idx]], 
                                             targets[peak_idx])
                dist = src[hemi_idx]['dist'][peak, target] * 1000
            else:
                dist = np.inf
            distances[n_idx, peak_idx] = dist
    
    np.savetxt(out_path, distances, delimiter = ',', fmt = '%.3f')

def crop_whitespace(img, borders_only = False):
    '''
    Crops white rows and columns from given image.

    Parameters
    ----------
    img : numpy.ndarray
        Image to crop, e.g. a screenshot of the brain or colorbar
    borders_only : bool, optional
        Retain whitespace inside the image, e.g. between colorbar and units,
        by default False.
    
    Returns
    -------
    cropped_img : numpy.ndarray
        Cropped image
    '''

    nonwhite_pix = (img != 255).any(-1)
    nonwhite_rows = nonwhite_pix.any(1)
    nonwhite_cols = nonwhite_pix.any(0)

    # Set white rows and cols nonwhite if they are inside rows or cols
    # of nonwhite pixels
    if borders_only:
        for axis in [nonwhite_rows, nonwhite_cols]:
            axis_ids = np.where(axis == True)[0]
            axis[axis_ids[0] : axis_ids[-1]] = True

    cropped_img = img[nonwhite_rows][:, nonwhite_cols]

    return cropped_img

def multi_sort(primary, secondaries):
    '''
    Sort multiple lists based on the primary list's order.

    Parameters
    ----------
    primary : list
        The first list to be sorted based on <primary>.
    secondaries : list of list
        Any number of lists to be sorted based on <primary>.
    
    Returns
    -------
    primary : list
        The first list ordered based on <primary>.
    secondaries : list of list
        All of the secondary lists ordered based on <primary>.
    '''

    for i in range(len(secondaries)):
        secondaries[i] = [x for _, x in sorted(zip(primary, secondaries[i]))]
    primary.sort()

    return primary, secondaries
