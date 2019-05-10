import os

import numpy as np
from datetime import datetime
from dateutil import tz

import pynwb
import nspike_helpers as ns 
import query_helpers as qu


def nspike_day_to_nwb(
        data_path,   # Path to anim directory 
        anim_name,   # e.g. Bond
        anim_prefix, # 3-letter prefix, e.g. Bon 
        day_num,
        nspike_zero_time, # tz-aware datetime corresponding to NSpike t=0 
        limit_num_of_tets=None, # debugging
        

# TODO
# -validate nspike_zero_time is tz-aware datetime

# General params/presets
file_create_date = datetime.now(tz.tzlocal())

source = 'NSpike data acquisition system'
eeg_samprate = 1500.0 # Hz

eeg_subdir = "EEG"
epochs_file = "times.mat"
tetinfo_file = "tetinfo.mat"
NSpike_timestamps_per_sec = 10000

print(file_create_date)
print(dataset_zero_time)

### Parse inputs and create NWBfile

day_str = '%02d' % day

nwb_filename = anim + day_str + '_test.nwb'

# check the input arguments
if not os.path.exists(data_dir):
        print('Error: data_dir %s does not exist' % data_dir)
        exit(-1)

# get filename prefix and file locations
prefix = anim.lower()
eeg_path = os.path.join(data_dir, eeg_subdir)

# Calculate the POSIX timestamp when Nspike clock = 0 (seconds)
NSpike_posixtime_offset = dataset_zero_time.timestamp()

# We'll still store NSpike/Trodes zero time in nwbfile.session_start_time, 
# so that we can recreate the experimental timestamps, even though that
# violates the definition of session_start_time as zero time. We need some
# other way to indicate that the timestamps are POSIX (easy: if they are over
# a trillion, then they're POSIX!)

nwbf = pynwb.NWBFile(data_source,
               'Converted NSpike data from %s' % data_dir,
               anim+day_str,
               session_start_time=dataset_zero_time,
               file_create_date=file_create_date,
               lab='Frank Laboratory',
               experimenter='Mattias Karlsson',
               institution='UCSF',
               experiment_description='Recordings from awake behaving rat',
               session_id=data_dir)

### Animal Behavior

# create position, direction and speed
position_list = []
pos = pynwb.behavior.Position(data_source, position_list, 'Position')

direction_list = []
dir = pynwb.behavior.CompassDirection(data_source, direction_list, 'Head Direction')

speed_list = []
speed = pynwb.behavior.BehavioralTimeSeries(data_source, speed_list, 'Speed')

# NOTE that day_inds is 0 based
time_list = {}
nwb_epoch = {}
pos_files = ns.get_files_by_day(data_dir, prefix, 'pos')
task_files = ns.get_files_by_day(data_dir, prefix, 'task')

mat = ns.loadmat_ff(task_files[day], 'task')
task_struct = mat[day]
# find the pos file for this day and load it

mat = ns.loadmat_ff(pos_files[day], 'pos')
pos_struct = mat[day]

# compile a list of time intervals in an array and create the position, head direction and velocity structures
time_list = []

# Assume field order: (time,x,y,dir,vel)
(time_idx, x_idx, y_idx, dir_idx, vel_idx) = range(5)

for epoch_num, pos_epoch in pos_struct.items():

    # convert times to POSIX time
    timestamps = pos_epoch['data'][:,time_idx] + NSpike_posixtime_offset

    # TODO: create a shared TimeSeries for timestamps, across all behavioral timeseries
    # ?? timestamps_obj = pynwb.TimeSeries(timestamps=timestamps...)

    # collect times of epoch start and end
    time_list.append([timestamps[0], timestamps[-1]])

    m_per_pixel = pos_epoch['cmperpixel'][0,0]/100 # NWB wants meters per pixel

    # we can also create new SpatialSeries for the position, direction and velocity information
    #NOTE: Each new spatial series has to have a unique name.
    pos.create_spatial_series(name='Position d%d e%d' % (day, epoch_num), 
                              source='overhead camera',
                              timestamps = timestamps,
                              data=pos_epoch['data'][:, (x_idx, y_idx)] * m_per_pixel,
                              reference_frame='corner of video frame',
                              #conversion=m_per_pixel,
                              #unit='m'
                              ) # *after* conversion

    dir.create_spatial_series(name='Head Direction d%d e%d'% (day, epoch_num), 
                              source='overhead camera',
                              timestamps=timestamps,
                              data=pos_epoch['data'][:, dir_idx],
                              reference_frame='0=facing top of video frame (?), positive clockwise (?)',
                              #unit='radians'
                              )

    speed.create_timeseries(name='Speed d%d e%d' % (day, epoch_num),
                             source='overhead camera',
                             timestamps=timestamps,
                             data=pos_epoch['data'][:, vel_idx] * m_per_pixel,
                             unit='m/s',
                             #conversion=m_per_pixel,
                             description='smoothed movement speed estimate')
time_list = np.asarray(time_list)
                                
    

# create a Processing module for behavior
behav_mod = nwbf.create_processing_module('Behavior', data_source, 'Behavioral variables')
# add the position, direction and speed data
behav_mod.add_data_interface(pos)
behav_mod.add_data_interface(dir)
behav_mod.add_data_interface(speed)


# # Now add the complete list of task intervals to the behav_mod module
# for interval in task_intervals:
#         behav_mod.add_data_interface(BehavioralEpochs('task information', interval))

### Tetrode info
# Load in `tetinfo` struct and populate ElectrodeTable, electrode groups, etc.

# Create the electrode table.
# The logic here is as follows:
#   Each Tetrode gets its own ElectrodeGroup and ElectrodeTableRegion
#   Each individual recording channel gets its own row in nwbfile.electrodes

# we first create the ElectrodeTable that all the electrodes will go into
nchan_per_tetrode = 4 #these files all contain tetrodes, so we assume four channels
tetinfo_filename = "%s/%s%s" % (data_dir, prefix, tetinfo_file)
recording_device = nwbf.create_device('NSpike acquisition system', data_source)
tet_electrode_group = {}
tet_electrode_table_region = {}
lfp_electrode_table_region = {}

mat = ns.loadmat_ff(tetinfo_filename, 'tetinfo')
#only look at first epoch because rest are duplicates
tets = mat[day][1]

# For debugging, limit number of tets to import
subset_keys = sorted(tets.keys())[0:limit_num_of_tets]
tets = {k:v for (k,v) in tets.items() if k in subset_keys}

print(limit_num_of_tets)
print("Using tetrode numbers:")
print(subset_keys)

tets[1]['depth'][0,0][0,0]

# kenny's data has a nested [day][epoch][tetrode] structure but duplicates the info across epochs, so we can just
# use the first epoch for everything
chan_num = 0 # this will hold an incrementing channel number for the entire day of data
for tet_num, tet in tets.items():
    #print('making electrode group for day %d, tet %d' % (day, tet_ind))
    # go through the list of fields
    hemisphere = '?'
    # tet.area/.subarea are 1-d arrays of Unicode strings
    area = str(tet['area'][0]) if 'area' in tet else '?' # h5py barfs on numpy.str_ type objects?
    if 'sub_area' in tet: 
        sub_area = str(tet['sub_area'][0]) # h5py barfs on numpy.str_ type objects?
        location = area + ' ' + sub_area
    else:
        sub_area = '?'
        location = area 

    # tet.depth is a 1x1 cell array in tetinfo struct for some reason (multiple depths?)
    # (which contains the expected 1x1 numeric array)
    coord = [np.nan, np.nan, tet['depth'][0, 0][0, 0] / 12 / 80 * 25.4] if 'depth' in tet else [np.nan, np.nan, np.nan]
    impedance = np.nan
    filtering = 'unknown - likely 600Hz-6KHz'

    channel_location = [location, location, location, location]
    channel_coordinates = [coord, coord, coord, coord]
    electrode_name = "%02d-%02d" % (day, tet_num)
    description = "tetrode {tet_num} located in {location} on day {day}".format(tet_num=tet_num,
                                                                               location=location,
                                                                               day=day)

    # we need to create an electrode group for this tetrode
    tet_electrode_group[tet_num] = nwbf.create_electrode_group(electrode_name,
                                                        data_source,
                                                        description,
                                                        location,
                                                        recording_device)

    for i in range(nchan_per_tetrode):
            # now add an electrode
            nwbf.add_electrode(x = coord[0],
                               y = coord[1],
                               z = coord[2],
                               imp = impedance,
                               location = location,
                               filtering = filtering,
                               group = tet_electrode_group[tet_num],
                               group_name = tet_electrode_group[tet_num].name,
                               id = chan_num)
            chan_num = chan_num + 1

    # now that we've created four entries, one for each channel of the tetrode, we create a new
    # electrode table region for this tetrode and number it appropriately
    table_region_description = 'ntrode %d region' % tet_num
    table_region_name = '%d' % tet_num
    tet_electrode_table_region[tet_num] = nwbf.create_electrode_table_region(
        region=list(range(chan_num-nchan_per_tetrode,chan_num)),
        description=table_region_description,
        # BUG #679: name must be 'electrodes' or NWB file will not be readable
        name='electrodes') #        name=table_region_name)



    # Also create electrode_table_regions for each tetrode's LFP recordings
    # (Assume that LFP is taken from the first channel)
    lfp_electrode_table_region[tet_num] = nwbf.create_electrode_table_region(
        [chan_num-nchan_per_tetrode],
        table_region_description,
        # BUG #679: name must be 'electrodes' or NWB file will not be readable
        name='electrodes') #        name=table_region_name)

# tet_electrode_table_region[1].region
# nwbf.ec_electrode_groups['03-01'].description
# tet_electrode_group

## LFP

%%time

eeg_files = ns.get_eeg_by_day(eeg_path, prefix, 'eeg')
lfp_data = []

lfp = pynwb.ecephys.LFP(data_source, lfp_data)
# read data from EEG/*eeg*.mat files and build TimeSeries object

print('processing LFP data for day %2d' % day)
for tet_num in tets.keys():
    print(' -> tet_num: %d' % tet_num)
    timestamps, data = ns.build_day_eeg(eeg_files[day][tet_num], eeg_samprate)
    timestamps += NSpike_posixtime_offset
    name = "{prefix}eeg-{day}-{tet}".format(prefix=prefix, day=day, tet=tet_num)
    lfp.create_electrical_series(name=name, 
                                 source=source,
                                 data=data / 1000, # convert mV to V, as expected
                                 electrodes=lfp_electrode_table_region[tet_num],
                                 timestamps=timestamps)
nwbf.add_acquisition(lfp)


## Spikes

# Create unit metadata first
# External clustering software gives names for each cluster--we want to preserve these
nwbf.add_unit_column('cluster_name',  '(str) cluster name from clustering software')
nwbf.add_unit_column('elec_group',    '(electrodeGroup) nTrode on which spikes were recorded')

# # For tetrode data, this will usually be all channels in the tetrode
# nwbf.add_unit_column('neighborhood',  '(electrodeTableRegion) list of electrodes on which spikes were clustered')

# AKA 'Valid_times'--the times during which a spike from this cluster could have possibly been observed.
# (handle periods between behavior epochs, acquisition system dropouts, etc.)
nwbf.add_unit_column('obs_intervals', '(intervalSeries) Observation Intervals for the spike times')

#get the spike times from the spikes files
#each cluster gets a unique number starting at zero

spike_files = ns.get_files_by_day(data_dir, prefix, 'spikes')
print('\nLoading spikes file :' + spike_files[day])
mat = ns.loadmat_ff(spike_files[day], 'spikes')

spike_mod = nwbf.create_processing_module('Spike Data', data_source, 'Clustered Spikes')
spike_UnitTimes = pynwb.misc.UnitTimes(data_source)

spike_unit = []
obs_intervals = {}
cluster_by_tet = {}
cluster_id = 0

# Matlab structs are nested by: day, epoch, tetrode, cluster, but we will want to save all spikes from a give cluster
# *across multiple epochs* in same spike list. So we rearrange the nested matlab structures for convenience. We 
# create a nested dict, keyed by 1) tetrode, 2) cluster number, then 3) epoch. NB the keys are 1-indexed, to be 
# consistent with the original data collection. (We only process one day at a time for now, so no need to nest days).

spike_struct = mat[day]
for epoch_num, espikes in spike_struct.items():
    for tet_num, tspikes in espikes.items():
        # respect tet subset selection done above
        if tet_num not in tets.keys():
            continue
        if tet_num not in cluster_by_tet.keys():
            cluster_by_tet[tet_num] = {}
        for cluster_num, cspikes in tspikes.items():
            if cluster_num not in cluster_by_tet[tet_num].keys():
                cluster_by_tet[tet_num][cluster_num] = {}
            cluster_by_tet[tet_num][cluster_num][epoch_num] = cspikes
                                

# now we create the SpikeEventStructures and their containing EventWaveform objects
colidx_timestamps = 0

for tet_num in cluster_by_tet.keys():
    obs_intervals[tet_num] = {}
    for cluster_num in cluster_by_tet[tet_num].keys():
        cluster_name = 'd%d t%d c%d' % (day, tet_num, cluster_num)
        print('Adding cluster id: %3d, name: %s' % (cluster_id, cluster_name))

        cluster_tmp = cluster_by_tet[tet_num][cluster_num]

        # construct a full data array and a parallel list of observation intervals
        obs_intervals[tet_num][cluster_num] = pynwb.misc.IntervalSeries(name = cluster_name, 
                                                    source = source,
                                                    description = 'Observation intervals for spikes from cluster ' +
                                                    str(cluster_num) + ' on tetrode ' + str(tet_num))

        spikes_ep = []
        for epoch in cluster_tmp.keys():
            if cluster_tmp[epoch]['data'].shape[0]:
                spikes_ep.append(cluster_tmp[epoch]['data'][:,colidx_timestamps] + NSpike_posixtime_offset)
            for obs_intervals_cl_ep in cluster_tmp[epoch]['timerange']:
                # 'timerange' for each cell is given in NSpike timestamp units
                obs_int = (obs_intervals_cl_ep.T.astype(float)/NSpike_timestamps_per_sec) + NSpike_posixtime_offset
                obs_intervals[tet_num][cluster_num].add_interval(*obs_int)

        spiketimes = np.concatenate(spikes_ep)

        # Add Observation Intervals to nwbfile willy-nilly (1 per cluster), 
        # so that we can successfully refer to them in the Unit metadata table
        spike_mod.add_data_interface(obs_intervals[tet_num][cluster_num])

        nwbf.add_unit(data = {'cluster_name': cluster_name, 
                              'elec_group': tet_electrode_group[tet_num], 
                              # can't just refer to electrode_table_region itself?: are never added to nwbfile to 
                              # begin with. Instead, use 'data' field, which is a list of electrodeTable indices.
#                               'neighborhood': tet_electrode_table_region[tet_num], # tet_electrode_table_region[tet_num].data, 
                              'obs_intervals': obs_intervals[tet_num][cluster_num]},
                      id = cluster_id)

        spike_UnitTimes.add_spike_times(cluster_id, spiketimes)

        cluster_id += 1

# Add UnitTimes to the spike_mod ProcessingModule
spike_mod.add_data_interface(spike_UnitTimes)

### Write out NWBfile!

# make an NWBFile
with pynwb.NWBHDF5IO(nwb_filename, mode='w') as iow:
    iow.write(nwbf)
print('Wrote nwb file: ' + nwb_filename)