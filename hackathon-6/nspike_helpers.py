# ------------------------------------------------
#    FRANK LAB FILTER FRAMEWORK --> NWB HELPERS
# ------------------------------------------------
#  Overview: 
#     The following functions assist in parsing old Frank Lab data stored as .mat files in the
#     'filter framework' format. We use these helpers for storing the data as NWB files. 
#
#     For an example of how these methods can be used to help in creating an
#     NWB file, see 'create_franklab_nwbfile.ipynb'. This notebook shows how to parse
#     publicly available Frank Lab data--available from CRCNS in the filter framework
#     format--and to store the data back as a single NWB file per day of an animal.
# ------------------------------------------------
# ------------------------------------------------



#import the necessary classes from the pynwb package
import glob as glob
import argparse
import scipy.io as sio
import numpy as np
import os as os
import re as re
from datetime import datetime
import warnings


from struct import unpack
from array import array



def parse_franklab_behavior_data(data_dir, animal, day):
    '''
    Purpose:
        Parse the 'pos' file from Frank Lab Filter Framework for a given animal and day.
        This contains data for animal position, head direction, and speed. By Frank Lab
        conventions, this file should be named 'XYZpos04.mat' for animal 'XYZ' day 4
        (e.g. 'bonpos04.mat').
    Argument:
        data_dir: full path to the directory containing the 'pos' files
        animal: animal name (lowercase)
        day: day of the experiment
    Returns:
        behavior_dict: dictionary containing the behavior data for the given animal and day
    '''
    behavior_files = get_files_by_day(data_dir, animal.lower(), 'pos')
    behavior_dict = loadmat_ff(behavior_files[day], 'pos')
    return behavior_dict[day]


def parse_franklab_task_data(data_dir, animal, day):
    '''
    Purpose:
        Parse the 'task' file from Frank Lab Filter Framework for a given animal and day.
        This contains metadata about epochs, such as which task and apparatus the animal was
        running on. By Frank Lab conventions, this file should be named 'XYZtask04.mat' for 
        animal 'XYZ' day 4 (e.g. 'bontask04.mat').
    Argument:
        data_dir: full path to the directory containing the 'task' files
        animal: animal name (lowercase)
        day: day of the experiment
    Returns:
        task_dict: dictionary containing the task data for the given animal and day
    '''
    task_files = get_files_by_day(data_dir, animal.lower(), 'task')
    task_dict = loadmat_ff(task_files[day], 'task')
    return task_dict[day]        

def get_exposure_num(epoch_metadata):
    '''
    Purpose:
        Given a particular epoch's metadata (a subset of the Filter Framework 'task' data), 
        returns the number of times the animal has been exposed to the apparatus of this epoch.
    Arguments:
        epoch_metadata (dict): the metadata dictionary from a particular epoch of the 'task' data.
            The 'task' data returned from 'parse_franklab_task_data()' is a dictionary
            keyed by epoch number. Each of these entries is itself a dictionary containing data specifically
            for that epoch.
    Returns:
        Cumulative number of exposures of animal to the apparatus in this epoch. (e.g. If this is the 
        third time the animal has run on the W-track, it would return 3.)
    '''
    if 'exposure' in epoch_metadata.keys():
        return epoch_metadata['exposure'][0][0]
    else:
        return 'NA'
      
def parse_franklab_tetrodes(data_dir, animal, day):
    '''
    Purpose:
        Parse the 'tetinfo' file from Frank Lab Filter Framework for a given animal and day.
        This contains data for tetrodes. By Frank Lab conventions, this file should be named 
        'XYZtetinfo.mat' for animal 'XYZ' (e.g. 'bontetinfo.mat'). Note that this is a single
        file per animal, but this function will return only the tetinfo data from a given day.
    Argument:
        data_dir: full path to the directory containing the 'tetinfo' file
        animal: animal name (lowercase)
        day: day of the experiment
    Returns:
        tets_dict: dictionary containing the tetrodes data for the given animal and day
    '''
    tetinfo_filename = "%s/%s%s" % (data_dir, animal.lower(), "tetinfo.mat")
    tets_dict = loadmat_ff(tetinfo_filename, 'tetinfo')
    # only look at first epoch (1-indexed) because rest are duplicates
    return tets_dict[day][1] 


def get_franklab_tet_location(tet):
    '''
    Purpose:
        Returns the location in the brain (e.g. 'CA3') of a single tetrode, from the 
        Filter Framework 'tetinfo' data for a particular animal and day.
    Arguments:
        tet (dict): the metadata dictionary from a single tetrode of the 'tetinfo' data.
            The 'tetinfo' data returned from 'parse_franklab_tetrodes()' is a dictionary
            keyed by tetrode number. Each of these entries is itself a dictionary containing data 
            specifically for a given tetrode.
    Returns:
        location (str): location in the brain of this tetrode
    '''
    # tet.area/.subarea are 1-d arrays of Unicode strings
    # cast to str() because h5py barfs on numpy.str_ type objects?
    area = str(tet['area'][0]) if 'area' in tet else '?'
    if 'sub_area' in tet: 
        sub_area = str(tet['sub_area'][0])
        location = area + ' ' + sub_area
    else:
        sub_area = '?'
        location = area 
    return location
        
    
def get_franklab_tet_depth(tet):
    '''
    Purpose:
        Returns the depth in the brain of a single tetrode, from the 
        Filter Framework 'tetinfo' data for a particular animal and day.
    Arguments:
        tet (dict): the metadata dictionary from a single tetrode of the 'tetinfo' data.
            The 'tetinfo' data returned from 'parse_franklab_tetrodes()' is a dictionary
            keyed by tetrode number. Each of these entries is itself a dictionary containing data 
            specifically for a given tetrode.
    Returns:
        depth: depth of the tetrode (speak to a Frank Lab member for explanation of units) 
    '''
    # tet.depth is a 1x1 cell array in tetinfo struct for some reason (multiple depths?)
    # (which contains the expected 1x1 numeric array)
    if 'depth' in tet:
        depth = tet['depth'][0, 0][0, 0] / 12 / 80 * 25.4
    else:
        depth = np.nan
    return depth


def parse_franklab_spiking_data(data_dir, animal, day):
    '''
    Purpose:
        Parse the 'spikes' file from Frank Lab Filter Framework for a given animal and day.
        This contains data for clustered unit spiking. By Frank Lab conventions, this file should be named 
        'XYZspikes04.mat' for animal 'XYZ' day 4 (e.g. 'bonspikes04.mat'). Note that we re-arrange these
        data into a nested dictionary keyed by 1) tetrode, 2) cluster number, then 3) epoch.
    Argument:
        data_dir: full path to the directory containing the 'spikes' file
        animal: animal name (lowercase)
        day: day of the experiment
    Returns:
        cluster_by_tet: dictionary containing the clustered spiking for this animal and day, keyed by
            tetrode, then cluster number, then epoch.
    '''
    spike_files = get_files_by_day(data_dir, animal.lower(), 'spikes')
    spike_dict = loadmat_ff(spike_files[day], 'spikes')
    spike_struct = spike_dict[day]
    # Matlab structs are nested by: day, epoch, tetrode, cluster, but we will want to save all spikes from a give cluster
    # *across multiple epochs* in same spike list. So we rearrange the nested matlab structures for convenience. We 
    # create a nested dict, keyed by 1) tetrode, 2) cluster number, then 3) epoch. NB the keys are 1-indexed, to be 
    # consistent with the original data collection. We only process one day at a time for now, so no need to nest days.
    cluster_by_tet = {}
    for epoch_num, espikes in spike_struct.items():
        for tet_num, tspikes in espikes.items():
            if tet_num not in cluster_by_tet.keys():
                cluster_by_tet[tet_num] = {}
            for cluster_num, cspikes in tspikes.items():
                if cluster_num not in cluster_by_tet[tet_num].keys():
                    cluster_by_tet[tet_num][cluster_num] = {}
                cluster_by_tet[tet_num][cluster_num][epoch_num] = cspikes
    return cluster_by_tet



def loadmat_ff(filename, varname):
    '''
    Parse Frank Lab FilterFramework .mat files (with nested row vector cell arrays
    representing day/epoch/tetrode[/cluster].
    '''
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=False)
    data = data[varname]
    if isinstance(data, np.ndarray):
        return _check_arr_ff(data, cellvec_to_dict=True)
    else:
        warnings.warn("Matlab variable is not an ndarray, returning as-is; type = %s" % type(elem))
        return (data)

def _check_dict_ff(d):
    '''
    A recursive function which converts matobjects (e.g. structs) to dicts
    '''
    for k, v in d.items():
        if isinstance(v, sio.matlab.mio5_params.mat_struct):
            d[k] = _check_dict_ff(_struct_to_dict(v)) 
        elif isinstance(v, np.ndarray):
            d[k] = _check_arr_ff(v, cellvec_to_dict=False)
    return d

    
def _check_arr_ff(arr, cellvec_to_dict):
    '''
    A recursive function which constructs dicts with 1-indexed keys from 
    row-vector Matlab cellarrays (other-shaped cell arrays are left as ndarrays), 
    recursing into the non-empty elements. We also squeeze singleton nodes if 
    they are 'leaf' nodes.
    '''
    # If this is a numeric ndarray, then bail, we only want to handle 
    # Matlab cell arrays, which are of dtype 'O'/np.object
    if arr.dtype != np.object: return arr

    # If this is a 1x1 ndarray of matstructs, squeeze it.
    if arr.size == 1:
        a0 = arr.flatten()[0]
        if isinstance(a0, sio.matlab.mio5_params.mat_struct):
            return _check_dict_ff(_struct_to_dict(a0))

    # we'll convert row vectors to 1-indexed dicts, with no entries for empty cells
    if cellvec_to_dict and arr.shape[0] == 1:
        d = {}
        for idx, elem in enumerate(arr[0,:]):
            key = idx+1
            if isinstance(elem, sio.matlab.mio5_params.mat_struct):
                d[key] = _check_dict_ff(_struct_to_dict(elem))
            elif isinstance(elem, np.ndarray):
                if elem.size == 0:
                    continue
                d[key] = _check_arr_ff(elem, cellvec_to_dict=True)
            else:
                warnings.warn("Array element is not a mat_struct, not an ndarray; type = %s" % type(elem))
                d[key] = elem
        return d

    # For cell arrays that are not row vectors, return them as ndarrays
    for index, elem in np.ndenumerate(arr):
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            arr[index] = _check_dict_ff(_struct_to_dict(elem))
        elif isinstance(elem, np.ndarray):
            arr[index] = _check_arr_ff(elem, cellvec_to_dict=False)
        else:
            warnings.warn("Array element is not a mat_struct, not an ndarray; type = %s" % type(elem))
            arr[index] = elem
    return arr

def _struct_to_dict(matstruct):
    d = matstruct.__dict__ # use dict provided by sio.loadmat
    del d['_fieldnames'] # remove private housekeeping item from dict
    return d

    
def get_files_by_day(base_dir, prefix, data_type):
        ''' Get files of a specific data types and return them in a dictionary indexed by day'''
        files = glob.glob("%s/%s%s*.mat" % (base_dir, prefix, data_type))
        f_re = re.compile('%s(?P<data_type>[a-z]+)(?P<day>[0-9]{2})' % prefix)
        ret = dict()
        for f in files:
                d = f_re.search(f)
                if d.group('data_type') != data_type:
                        continue
                day = int(d.group('day'))
                ret[day] = f

        return ret


def get_eeg_by_day(EEG_dir, prefix, data_type):
        '''
                Get eeg files in EEG dir and return separated by day and tetrode, and sorted by epoch

                Args:
                        EEG_dir: the path to the 'EEG' directory
        '''
        # can't do *eeg*.mat since that would get *eeggnd*.mat files as well
        files = glob.glob("%s/*.mat" % EEG_dir)
        fp_re = re.compile('%s(?P<data_type>[a-z]+)(?P<day>[0-9]{2})-(?P<epoch>[0-9]+)-(?P<tetrode>[0-9]{2})' % prefix)
        ret = dict()
        for f in files:
                d = fp_re.search(f)
                if d.group('data_type') != data_type:
                        continue
                day = int(d.group('day'))
                epoch = int(d.group('epoch'))
                tetrode = int(d.group('tetrode'))
                if day not in ret:
                        ret[day] = dict()
                if tetrode not in ret[day]:
                        ret[day][tetrode] = dict()
                ret[day][tetrode][epoch] = f
        for day in ret.keys():
                for tetrode in ret[day].keys():
                        sorted_list = list()
                        for epoch in sorted(ret[day][tetrode].keys()):
                                # create a sorted list of the epochs and store the epoch and tetrodes so we can index into the matlab class
                                # properly
                                sorted_list.append((ret[day][tetrode][epoch], (day, epoch, tetrode)))
                        ret[day][tetrode] = sorted_list
        return ret

def build_day_eeg(files_by_tetrode, samprate):
        d = np.zeros([0,1], np.float64)
        t = np.zeros([0,1], np.float64)
        for file_info in files_by_tetrode:
#                 print('loading file: %s' % file_info[0])
                (day, epoch, tet_num) = file_info[1]
                mat_eeg = loadmat_ff(file_info[0], 'eeg') # use loadmat2 to avoid squeezing when day, epoch or tet=1
                eeg = mat_eeg[day][epoch][tet_num] # last index is for struct array
                # create a list of times for the data
                t = np.concatenate((t, eeg['starttime'] + (np.arange(0,len(eeg['data'])).reshape(-1,1) / samprate)))
                d = np.concatenate((d, eeg['data']))
        return t, d

    
    