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
                print('loading file: %s' % file_info[0])
                (day, epoch, tet_num) = file_info[1]
                mat_eeg = loadmat_ff(file_info[0], 'eeg') # use loadmat2 to avoid squeezing when day, epoch or tet=1
                eeg = mat_eeg[day][epoch][tet_num] # last index is for struct array
                # create a list of times for the data
                t = np.concatenate((t, eeg['starttime'] + (np.arange(0,len(eeg['data'])).reshape(-1,1) / samprate)))
                d = np.concatenate((d, eeg['data']))
        return t, d

    
    