import copy
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from scipy.interpolate import interp1d
import intervals as iv  # backend for TimeIntervals


class TimeIntervals():
    """
    Class for representing a list of non-overlapping time intervals.
    
    Currently uses python-intervals as a backend, abstracting this from other classes that use intervals. 
    To replace with a different backend, change the methods accordingly.
    """
    
    def __init__(self, bounds=None):
        self.intervals = self.__make_intervals(bounds)
        
    def __make_intervals(self, bounds):
        '''Create an interval.Interval from start/end times'''
        if bounds is None:
            return iv.empty()
        elif (isinstance(bounds, np.ndarray) and bounds.ndim == 2 and bounds.shape[1] == 2):
            # input is a 2D (m x 2) numpy array
            intervals = iv.empty()
            for ivl in bounds:
                intervals = intervals | iv.closed(*ivl)
            return intervals
        elif (isinstance(bounds, np.ndarray) and bounds.ndim == 1 and bounds.size == 2):
            # input is a 1D (1 x 2) numpy vector
            return iv.closed(*bounds)
        else:
            raise TypeError("'bounds' must be an m x 2 numpy array (where m is the number of obs intervals) or " +
                            "a 1x2 numpy vector (for a single obs interval)")
        
    def to_array(self):
        '''Create m x 2 numpy array from the set of intervals in an TimeIntervals object'''
        return np.array([[atomic_ivl.lower, atomic_ivl.upper] for atomic_ivl in self.intervals])
    
    def durations(self):
        '''Return duration of each obs_interval'''
        return np.diff(self.to_array(), axis=1)
        
    def __and__(self, time_intervals):
        '''Return the intersection of two TimeIntervals'''
        intersection = self.intervals & time_intervals.intervals  # intersect using python-intervals
        return TimeIntervals(np.array([[ivl.lower, ivl.upper] for ivl in intersection]))

    def intersect(self, time_intervals):
        '''Return the intersection of two TimeIntervals'''
        return self & time_intervals
    
    def __or__(self, time_intervals):
        '''Return the union of two TimeIntervals'''
        union = self.intervals | time_intervals.intervals  # union using python-intervals
        return TimeIntervals(np.array([[ivl.lower, ivl.upper] for ivl in union]))

    def union(self, time_intervals):
        '''Return the union of two TimeIntervals'''
        return self | time_intervals

    def __contains__(self, t):
        """Check whether time t is in TimeIntervals. Supports the 'v in TimeIntervals' pattern."""
        return t in self.intervals
    
    def __len__(self):
        """Return number of non-overlapping intervals (i.e. start/stop) in this TimeIntervals."""
        if self.intervals.is_empty(): # iv.empty() has len 1; it contains a special empty interval (I.inf, -I.inf)
            return 0
        return len(self.intervals)
    
    def __iter__(self):
        self.iter_idx = 0
        return self
    
    def __next__(self):
        # make TimeIntervals an iterator object
        if self.iter_idx >= self.__len__():
            raise StopIteration
        else:
            ivl = self.intervals[self.iter_idx]
            self.iter_idx += 1
            return np.array([ivl.lower, ivl.upper])

    
class TimeBasedData(ABC):
    """Abstract base class for representing data observed over some observation intervals."""
    
    # Abstract method. All subclasses must implement a time_query method.
    @abstractmethod     
    def time_query(self, query):
        raise NotImplementedError("Subclasses of abstract class Data must implement 'time_query' method.")
        
    # All subclasses will inherit an obs_intervals property. May be overridden.
    @property
    def obs_intervals(self):
        return self._obs_intervals
    
    @obs_intervals.setter
    def obs_intervals(self, new_obs_intervals):
        if not(isinstance(new_obs_intervals, TimeIntervals)):
            raise TypeError("'obs_intervals' must be of type nwb_query.TimeIntervals")
        self._obs_intervals = new_obs_intervals
        
        
        
class PointData(TimeBasedData):
    '''
    Represent a list of discrete times occurring during defined intervals,
    optionally with mark data associated with each time.
    '''    
    def __init__(self, point_times, obs_intervals, marks=None):
        self.typecheck(point_times, marks=marks) # obs_intervals is typechecked in TimeBasedData
        self.point_times = point_times
        self.obs_intervals = obs_intervals  # this uses the inherited setter
        self.marks = marks
    
    def typecheck(self, point_times, marks=None):
        if not isinstance(point_times, np.ndarray):
            raise TypeError("'point_times' must be a numpy.array")
        if not(point_times.ndim==1):
            raise ValueError("'point_times' must be a vector (1-dimensional array).")
        if marks and not isinstance(marks, np.ndarray):
            raise TypeError("'marks' must be a numpy.array")
        if marks and not(marks.shape[0]==self.point_times.shape[0]):
            raise ValueError("'marks' must have same # of entries (rows) as 'point_times'.")
    
    def time_query(self, query):
        '''Return PointData with data available during requested query.'''
        
        # Find the resulting obs_intervals where the data have support (i.e. intersect with the selection intervals)
        if isinstance(query, EventData):
            result_obs_intervals = self.obs_intervals & query.event_intervals
        elif isinstance(query, TimeIntervals):
            result_obs_intervals = self.obs_intervals & query
        else:
            raise TypeError("PointData.query currently only supports queries of " +
                            "type nwb_query.EventData or nwb_query.TimeIntervals")
        # Collect the valid point_times in the query
        valid_indices = [i for (i, t) in enumerate(self.point_times) if t in query]
        result_point_times = self.point_times[valid_indices]
        if self.marks:
            result_marks = self.marks[valid_indices, :]
            return PointData(result_point_times, result_obs_intervals, result_marks)
        else:
            return PointData(result_point_times, result_obs_intervals)
        
        
    def mark_with_ContinuousData(self, continuous_data, merge_obs_intervals=True, interpolation='linear'):
        """
        Evaluate ContinuousData at each point_time and add result as a corresponding mark. 
        
        If merge_obs_intervals=True, the resulting PointData will have obs_intervals that 
        are the intersection of the obs_intervals of the inputs, and it may not contain all 
        of the original point_times. Otherwise, all point_times will be returned, but will be
        marked with 'None' at times when continuous_data is undefined (i.e. outside its 
        obs_intervals).
        
        'interpolation' is passed to scipy.interp1d as 'kind'. Available kinds include ('nearest', 'linear'
        'quadratic', 'cubic')
        
        """
        if continuous_data.samples.ndim > 1 and not (interpolation == 'nearest' or interpolation == 'linear'):
            raise NotImplementedError("For data > 1-D, only 'nearest' and 'linear' interpolation are currently suppported")

        # Make an interpolation function using the continuous data and sample_times, which we will use to
        # evaluate the ContinuousData at the point_times of this PointData
        interpolator = interp1d(x=continuous_data.sample_times, 
                                y=continuous_data.samples, 
                                kind=interpolation, 
                                axis=0)
        
        mark_shape = continuous_data.samples.shape[1:] # 1 row per time point, but mark data can be multi-d
        
        if merge_obs_intervals:
            # timequery on pp: intersects obs_intervals and discards point_times outside overlapping region
            result_pp = self.time_query(continuous_data.obs_intervals)
            # don't need to worry about time points outside of the continuous data, b/c we have already filtered point_times
            result_pp.marks = interpolator(result_pp.point_times)
            return result_pp
        else:
            result_marks = []
            for t in self.point_times:
                if t in continuous_data.obs_intervals:
                    result_marks.append(interpolator(t), 'extrapolate') # extrapolate when within obs_intervals but outside last sample?
                else:
                    result_marks.append(np.full(mark_shape, None))  # set mark to None if the time point occurs outside the continuous data   
                    
            return PointData(self.point_times, self.obs_intervals, marks=np.concatenate(result_marks, axis=0))


class ContinuousData(TimeBasedData):
    
    def __init__(self, samples, sample_times, obs_intervals=None, find_gaps=False):
        if not obs_intervals and find_gaps:
            self.obs_intervals = self.__find_obs_intervals(self.sample_times)
        elif not obs_intervals:
            sample_times_range = np.array([[sample_times[0], sample_times[-1]]])
            self.obs_intervals = TimeIntervals(sample_times_range)
        else:
            self.obs_intervals = obs_intervals
        self.samples = samples
        self.sample_times = sample_times
    
    def __find_obs_intervals(self, sample_times, gap_threshold_samps=1.5):
        """Optionally build obs_intervals from any gaps in the data.
        
        This is currently not tested.
        """
        import warnings
        warnings.warn("Deducing obs_interval is currently untested, may be bogus.")
        stepsize = np.mean(np.diff(sample_times, 1)) # use first derivatives to estimate the stepsize
        diffs = np.diff(sample_times, 2)  # use second derivative to identify gaps
        epsilon = gap_threshold_samps * stepsize  # only count if the gap is big with respect to the stepsize
        ivl_end_indices = np.where(diffs > epsilon)[0] + 1  
        if ivl_end_indices.size == 0:  # no gaps in observation
            return TimeIntervals(np.array([[sample_times[0], sample_times[-1]]]))
        else:
            # append the last valid index of the array to the end indices
            np.append(ivl_end_indices, ivl_end_indices.size-1) 
            # build the obs_intervals
            bounds = []  
            for i, end_idx in enumerate(ivl_end_indices):
                if i == 0:   # handle the first interval
                    bounds.append([self.sample_times[0], self.sample_times[end_idx]])
                else:
                    previous_end_idx = ivl_end_indices[i-1]
                    new_start_idx = previous_end_idx + 1
                    bounds.append([self.sample_times[new_start_idx], self.sample_times[end_idx]])
            return TimeIntervals(np.array(bounds))
            

    def time_query(self, query):
        """Return ContinuousData in the specified time_intervals.
        
        The resulting obs_intervals is the intersection of the obs_intervals of this ContinuousData and
        the provided. time_intevals. The resulting samples and sample_times are those occurring in the 
        resulting obs_intervals.
        """
        # Constrain the resulting obs_intervals to where the data have support (i.e. intersect with selection intervals)
        if isinstance(query, EventData):
            result_obs_intervals = self.obs_intervals & query.event_intervals
        elif isinstance(query, TimeIntervals):
            result_obs_intervals = self.obs_intervals & query
        else:
            raise TypeError("'query' must be of type nwb_query.EventData or nwb_query.TimeIntervals")
        
        # Get index into samples and sample_times of interval starts/ends
        result_bounds = result_obs_intervals.to_array()
        result_lower_bounds = result_bounds[:,0]
        result_upper_bounds = result_bounds[:,1]
        # Intervals are closed; find first/last matching sample_times for lower/upper bounds
        result_lower_index = np.searchsorted(self.sample_times, result_lower_bounds, side='left')
        result_upper_index = np.searchsorted(self.sample_times, result_upper_bounds, side='right')

        # TODO: speedup by initializing output arrays (use index to compute size)
        result_samples = []
        result_sample_times = []
        for idx_lower, idx_upper in zip(result_lower_index, result_upper_index):
            result_samples.append(self.samples[idx_lower:idx_upper, :])
            result_sample_times.append(self.sample_times[idx_lower:idx_upper])
        
        return ContinuousData(samples=np.concatenate(result_samples),
                              sample_times=np.concatenate(result_sample_times),
                              obs_intervals=result_obs_intervals)
    
    
    def filter_intervals(self, func, domain_cols=False):
        """Return a EventData where the ContinuousData fulfills a boolean lambda function ('func').
        
        By default, 'func' should accept all columns of ContinuousData.samples as input.
        Otherwise, provide a list of the column indices that should be used.
        """
        if self.samples.shape[0] == 0:
            return EventData(event_intervals=TimeIntervals(),
                             obs_intervals=self.obs_intervals)
                                 
        # apply the function to the correct columns of the samples
        if domain_cols:
            assert max(domain_cols) < self.samples.shape[1]
            func_of_data = func(self.samples[:, domain_cols])
        else:
            func_of_data = func(self.samples)
        
        # Get the up/down crossing indices, i.e. the first/last elements in each interval that fulfill 'func'
        assert func_of_data.dtype == 'bool'
        df = np.diff(func_of_data.astype(np.int8))
        up_crossings = np.where(df == 1)[0] + 1
        down_crossings = np.where(df == -1)[0]

        # if data begins while function is true, include this as an up-crossing
        if func_of_data[0]:
            up_crossings = np.insert(up_crossings, 0, 0)

        # if data ends while function is true, include this as a down-crossing    
        if func_of_data[-1]:
            down_crossings = np.append(down_crossings, func_of_data.shape[0]-1)
        
        # Create the time intervals
        up_times = self.sample_times[up_crossings]
        down_times = self.sample_times[down_crossings]
        interval_bounds = np.array((up_times, down_times)).T
        filtered_intervals = TimeIntervals(interval_bounds)
        
        # Return a EventData instance, with the same obs_intervals as this ContinuousData instance
        return EventData(event_intervals=filtered_intervals,
                         obs_intervals=self.obs_intervals)
        

    

class EventData(TimeBasedData):
    """Represent events occurring during a period of observation.
    
    
    A EventData object can be used for querying additional datasets, while matinating the 
    correct observation intervals from the initial dataset. For example, selecting time intervals 
    where an animal was running faster than a threshold speed, and using the resulting 
    EventData object to query spiking data for a clustered unit. 
    """
    
    def __init__(self, event_intervals, obs_intervals):
        if not isinstance(event_intervals, TimeIntervals):
            raise TypeError("'event_intervals' must be a query.TimeIntervals instance")
        if not np.all([event_ivl in obs_intervals for event_ivl in event_intervals.intervals]):
            raise ValueError("'event_intervals' must be fully contained in 'obs_intervals'.")
        self.event_intervals = event_intervals
        self.obs_intervals = obs_intervals # uses the inherited setter
    
    
    def time_query(self, query):
        '''Return EventData with events and observation intervals available during the requested query.'''
        
        # Find the resulting event_intervals and obs_intervals
        if isinstance(query, EventData):
            result_event_intervals = self.event_intervals & query.event_intervals
            result_obs_intervals = self.obs_intervals & query.event_intervals   # confirm this
        elif isinstance(query, TimeIntervals):
            result_event_intervals = self.event_intervals & query
            result_obs_intervals = self.obs_intervals & query
        else:
            raise TypeError("EventData.query currently only supports queries of " +
                            "type nwb_query.EventData or nwb_query.TimeIntervals")
        return EventData(event_intervals=result_event_intervals,
                         obs_intervals=result_obs_intervals)
        
    def __contains__(self, t):
        """Check whether time t is in the event intervals."""
        return t in self.event_intervals
    
    def obs_contain(self, t):
        """Check whether time t is in the EventData's observation intervals."""
        return t in self.obs_intervals
    
    def durations(self):
        """Durations of the event intervals."""
        return self.event_intervals.durations()
    
    def obs_durations(self):
        """Durations of the observation intervals."""
        return self.obs_intervals.durations()
    
    def __and__(self, other):
        '''Return the intersection of two EventData objects'''
        if not isinstance(other, EventData):
            raise TypeError("'other' must be a nwb_query.EventData object")
        result_event_ivl = self.event_intervals & other.event_intervals
        result_obs_ivl = self.obs_intervals & other.obs_intervals
        return EventData(result_event_ivl, result_obs_ivl)

    def intersect(self, other):
        '''Return the intersection of two EventData objects'''
        return self & other
    
    def __or__(self, other):
        '''Return the union of two EventData objects'''
        if not isinstance(other, EventData):
            raise TypeError("'other' must be a nwb_query.EventData object")
        result_event_ivl = self.event_intervals | other.event_intervals
        result_obs_ivl = self.obs_intervals | other.obs_intervals
        return EventData(result_event_ivl, result_obs_ivl)

    def union(self, other):
        '''Return the union of two EventData objects'''
        return self | other

        
