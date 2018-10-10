#import the necessary classes from the pynwb package
import glob as glob
import argparse
import scipy.io as sio
import numpy as np
import os as os
import re as re
from datetime import datetime


from struct import unpack
from array import array


def loadmat2(filename):
        '''
        this function is the same as above but needs to be used when getting rid of empty entries doesn't work
        '''
        data = sio.loadmat(filename, struct_as_record=False, squeeze_me=False)
        return _check_keys(data)

def _check_keys(data):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in data:
            if isinstance(data[key], sio.matlab.mio5_params.mat_struct):
                data[key] = _todict(data[key])
            elif isinstance(data[key],np.ndarray):
                data[key] = _check_arr(data[key])
        return data

def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        # reuse the dict that sio.loadmat created for us
        d = matobj.__dict__
        del d['_fieldnames'] # remove private housekeeping item from dict
        
        for k, v in d.items():
                if isinstance(v, sio.matlab.mio5_params.mat_struct):
                        d[k] = _todict(v)
                elif isinstance(v, np.ndarray):
                        d[k] = _check_arr(v)
                else:
                        pass
        return d

def _check_arr(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        '''
        # TODO: test datatype for 'O' and bail
        if ndarray.dtype != np.object: return ndarray
        
        for index, elem in np.ndenumerate(ndarray):
                if isinstance(elem, sio.matlab.mio5_params.mat_struct):
                        ndarray[index] = _todict(elem)
                elif isinstance(elem, np.ndarray):
                        ndarray[index] = _check_arr(elem)
                else:
                        pass
        return ndarray

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
                mat = loadmat2(file_info[0]) # use loadmat2 to avoid squeezing when day, epoch or tet=1
                eeg = mat['eeg'][0,day-1][0,epoch-1][0,tet_num-1][0,0] # last index is for struct array
                # create a list of times for the data
                t = np.concatenate((t, eeg['starttime'] + (np.arange(0,len(eeg['data'])).reshape(-1,1) / samprate)))
                d = np.concatenate((d, eeg['data']))
        return t, d