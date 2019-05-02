# ------------------------------------------------
#        FRANK LAB NWB -- HELPER FUNCTIONS
# ------------------------------------------------
#  Overview: 
#     The following functions assist in parsing old Frank Lab data using the 
#     Frank Lab NWB extensions (franklab.extensions.yaml) for representing 
#     aspects of the data that are not easily stored in vanilla NWB (e.g. Behavioral apparatus/track
#     geometries). There are also some functions for working between with Network X graphs and
#     Frank Lab Apparatus geometries. See the headers below for a rough organization.
#
#     See nspike_helpers.py for general helper functions for parsing Frank Lab filter 
#     framework data.
# ------------------------------------------------
# ------------------------------------------------



import numpy as np
import pandas as pd
import networkx as nx
import nspike_helpers as ns
import matplotlib.pyplot as plt
import nwb_query as query
from fl_extension import *



# ------------------------------------------------
#     Frank Lab Apparatus Extension Helpers 
# ------------------------------------------------

def get_apparatus_from_linpos(linpos_this_epoch, name='some apparatus', conversion=1.0):
    '''
    Purpose: 
        Make an Apparatus object representing a linearized apparatus from animal Bon data
        extracted from Frank Lab Filter Framework. This method might not work for all 
        Frank Lab Filter Framework data. In particular, publicly available CRCNS data
        do not contain linpos data at all.
    
    Arguments: 
        linpos_this_epoch (dict): Data extracted from Frank Lab Filter Framework linpos 
            for this epoch (see convert_nspike.ipynb)
        name (str)[optional]: name of this apparatus
        conversion (float): conversion factor to multiply linpos coordinates 
                            (i.e. for converting pixels to meters)
        
    Returns:
        appar (Apparatus): Apparatus object representing the linearized apparatus
    '''
    # Dummy Apparatus for epochs without one
    if not linpos_this_epoch:
        return Apparatus(name='NA', nodes=[], edges=[])  # apparatus NA for this epoch
    
    nodes=[] 
    
    # Segment nodes
    seg_coords = linpos_this_epoch['segmentInfo']['segmentCoords']
    for i in range(seg_coords.shape[0]):
        # convert coords to meters using m/pixel conversion factor
        start = [conversion * c for c in list(seg_coords[i, 0:2])]
        end = [conversion * c for c in list(seg_coords[i, 2:])]
        seg_node = SegmentNode(name='segment' + str(i), coords=[start, end])
        nodes.append(seg_node)
        
    # Well nodes
    well_coords = linpos_this_epoch['wellSegmentInfo']['wellCoord']
    for i in range(well_coords.shape[0]):
        coords_m = [conversion * c for c in list(well_coords[i])]  # m/pixel
        well_node = PointNode(name='well' + str(i), coords=[coords_m])
        nodes.append(well_node)
        
    # Edges
    edges = find_edges(nodes)
    
    # Graph
    appar = Apparatus(name=name, nodes=nodes, edges=edges)
    return appar


        
def get_franklab_apparatus(epoch_metadata, behav_mod):
    '''
    Purpose:
        Get the Frank Lab Apparatus object for a particular epoch. Note that all of the
        Apparatus objects must already be created and added to the ProcessingModule 'behav_mod'
        that is passed into this function. This function is useful when iteratively processing
        each epoch of data and mapping it onto one of a subset of pre-defined apparatuses.
    Arguments:
        epoch_metadata: the metadata dictionary from a particular epoch of the Filter Framework 'task' data.
            The 'task' data returned from 'nspike_helpers.parse_franklab_task_data()' is a dictionary
            keyed by epoch number. Each of these entries is itself a dictionary containing data specifically
            for that epoch, including metadata about the apparatus the animal was on.
        behav_mod (PyNWB ProcessingModule): a PyNWB ProcessingModule containing all of the Frank Lab Apparatus
            objects. This is where we will look to find the appropriate apparatus for this epoch.
    Returns:
        appar (Apparatus): Frank Lab Apparatus (franklab.extensions.yaml) for this epoch
            
    '''
    # Extract 'type' and 'environment' from the parsed Matlab data
    if 'type' in epoch_metadata.keys():
        epoch_type = epoch_metadata['type'][0]
    else:
        epoch_type = 'NA'
    if 'environment' in epoch_metadata.keys():
        epoch_env = epoch_metadata['environment'][0]
    else:
        epoch_env = 'NA'
    # Get the Frank Lab Apparatus for this epoch
    if epoch_type == 'sleep':
        return behav_mod.data_interfaces['Sleep Box']
    elif epoch_type == 'run':
        # Which enviroment?
        if epoch_env == 'TrackA':
            return behav_mod.data_interfaces['W-track A']
        elif epoch_env == 'TrackB':
            return behav_mod.data_interfaces['W-track B']
        else:
            raise RuntimeError("Epoch 'environment' {} not supported.".format(epoch_env))
        appar = behav_mod.data_interfaces[epoch_env]
    else:
        raise RuntimeError("Epoch 'type' {} not supported.".format(epoch_type))  
        
        
def get_franklab_task(epoch_metadata, behav_mod): 
    '''
    Purpose:
        Get the Frank Lab Task object for a particular epoch. Note that all of the
        Task objects must already be created and added to the ProcessingModule 'behav_mod'
        that is passed into this function. This function is useful when iteratively processing
        each epoch of data and mapping it onto one of a subset of pre-defined Tasks.
    Arguments:
        epoch_metadata: the metadata dictionary from a particular epoch of the Filter Framework 'task' data.
            The 'task' data returned from 'nspike_helpers.parse_franklab_task_data()' is a dictionary
            keyed by epoch number. Each of these entries is itself a dictionary containing data specifically
            for that epoch, including metadata about the task the animal was doing.
        behav_mod (PyNWB ProcessingModule): a PyNWB ProcessingModule containing all of the Frank Lab Task
            objects. This is where we will look to find the appropriate Task for this epoch.
    Returns:
        (Task): Frank Lab Task (franklab.extensions.yaml) for this epoch
            
    '''
    # Extract epoch 'type' from the parsed Matlab data
    if 'type' in epoch_metadata.keys():
        epoch_type = epoch_metadata['type'][0]
    else:
        epoch_type = 'NA'
    # Return the appropriate Frank Lab Taks
    if epoch_type == 'sleep':
         return behav_mod.data_interfaces["Sleep"]
    elif epoch_type == 'run':
         return behav_mod.data_interfaces["W-Alternation"]
    else:
        raise RuntimeError("Epoch 'type' {} not supported.".format(epoch_type))  

        

def get_franklab_nodes(points, segments, polygons):
    '''
    Purpose:
        Create a list of Frank Lab PointNode, SegmentNode, and PolygonNode objects, given a set of 
        Python dictionaries where each entry represents the geometrical coordinates of a single point,
        segment, or polygon that we want to make a Node for.
    Arguments:
        points (dict): dictionary where each key is the name of a point, and each value is a 1x2 list
            with the x/y coordinates of the point
        segments (dict): dictionary where each key is the name of a segement, and each value is a 2x2 list
            with the x/y coordinates of the start and end point of the segment
        polygons (dict): dictionary where each key is the name of a polygon, and each value is a nx2 list
            with the x/y coordinates of the vertices in the polygon
    Returns:
        nodes (list): list of Frank Lab PointNode, SegmentNode, and PolygonNode objects
        
    '''
    nodes = []
    for name, point in points.items():
        # wrap [x,y] point in a list so all apparatus coords (point, segment, or polygon) are nested lists
        nodes.append(PointNode(name=name, coords=[point])) 
    for name, segment in segments.items():
        nodes.append(SegmentNode(name=name, coords=segment))
    for name, polygon in polygons.items():
        nodes.append(PolygonNode(name=name, coords=polygon, interior_coords=None))
    return nodes

        
def separate_epochs_by_apparatus(data_dir, animal, day):
    '''
    Purpose:
        For data from the CRCNS hc-6 dataset. Returns which epochs in a given day
        that the animal was in the Sleep Box, W-track A, or W-track B.
    Arguments:
        animal: the animal to look at
        day: the day to look at
    Returns:
        sleep_epochs (list): list of epoch numbers where the animal was in the sleep box
        wtrackA_epochs (list): list of epoch numbers where the animal was on W-track A
        wtrackB_epochs (list): list of epoch numbers where the animal was on W-track B
    '''
    
    sleep_epochs, wtrackA_epochs, wtrackB_epochs = [], [], []
    all_epochs_metadata = ns.parse_franklab_task_data(data_dir, animal, day)
    for epoch_num, epoch_metadata in all_epochs_metadata.items():
        if 'environment' in epoch_metadata.keys():
            if epoch_metadata['environment'][0] == 'TrackA':
                wtrackA_epochs.append(epoch_num)
            else:
                wtrackB_epochs.append(epoch_num)
        else:
            sleep_epochs.append(epoch_num)
    return sleep_epochs, wtrackA_epochs, wtrackB_epochs


def plot_position_by_epochs(animal, pos_series, epoch_ivls, epochs, title):
    '''
    Purpose:
        Plot animal position for particular epochs of a single day.
    Arguments:
        animal: the animal
        day: the day
        nwbf: the PyNWB NWBFile object, which must contain a position module
            with a particular formatting
        epochs: list of epochs to plot position for
        title: title for the plot
    Returns:
        f: the Matplotlib figure object
    '''
    # Convert position across-epochs into a nwb_query.ContinuousData object
    samples = pd.DataFrame(data=pos_series.data[()], columns=['x', 'y'])
    sample_times = pos_series.timestamps[()]
    valid_intervals = query.TimeIntervals(epoch_ivls)
    position = query.ContinuousData(samples=samples, 
                                    sample_times=sample_times, 
                                    valid_intervals=valid_intervals)
    
    f = plt.figure()
    plt.title(title)
    plt.xlabel('X position (meters)')
    plt.ylabel('Y position (meters)')
    for epoch in epochs:
        epoch_query = query.TimeIntervals(epoch_ivls[epoch-1, :])
        epoch_pos = position.time_query(epoch_query)
        plt.plot(epoch_pos.samples['x'], epoch_pos.samples['y'], label='epoch%s' % epoch, zorder=1)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', borderaxespad=0.)
    return f
    

def overlay_apparatus_geom(fig, points, segments, polygons):
    '''
    Purpose:
        Plot apparatus geometry for particular epochs of a single day.
        Must already have a figure that you want to overlay the apparatus geometry onto.
    Arguments:
        fig: Matplotlib figure object to overlay onto
        points (dict): dictionary where each key is the name of a point, and each value is a 1x2 list
            with the x/y coordinates of the point
        segments (dict): dictionary where each key is the name of a segement, and each value is a 2x2 list
            with the x/y coordinates of the start and end point of the segment
        polygons (dict): dictionary where each key is the name of a polygon, and each value is a nx2 list
            with the x/y coordinates of the vertices in the polygon
    Returns:
        None
    '''
    plt.figure(fig.number)
    for name, point in points.items():
        plt.scatter(point[0], point[1], marker='o', edgecolors='r', s=100, facecolors='none', linewidth=2, zorder=3)
    for name, segment in segments.items():
         plt.plot([c[0] for c in segment], [c[1] for c in segment], marker='o', color='k', zorder=3)
    for name, polygon in polygons.items():
        for i in range(len(polygon)-1):
            plt.plot([polygon[i][0], polygon[i+1][0]], [polygon[i][1], polygon[i+1][1]], color='k', zorder=2)
        plt.plot([polygon[i+1][0], polygon[0][0]], [polygon[i+1][1], polygon[0][1]], color='k', zorder=2)


def coords_intersect(n1, n2, tol=0.01):
    '''
    Purpose:
        Returns True if the input points are equivalent.
    
    Arguments:
        n1 (1x2 list): [x, y] coordinate of the first point
        n2 (1x2 list): [x, y] coordinate of the second point
        tol (float): tolerance within which to consider floating-point values equal
        
    Returns:
        True if and only if n1 == n2 within some tolerance. False otherwise.
    '''
    for c1 in n1.coords:
        for c2 in n2.coords:
            if (abs(c1[0] - c2[0]) <= tol and abs(c1[1] - c2[1]) <= tol):
                return True
    return False

def find_edges(node_list):
    '''
    Purpose:
        Find pairs of Frank Lab nodes that share at least one x/y coordinate.
    Arguments:
        node_list (1 x n list): list of objects inheriting from Frank Lab Node
    Returns:
        ret_edges (1 x m list): list of Frank Lab Edge objects, each of which contains the
            names of two Nodes that share at least one coordinate
    '''
    edges = []
    for n1 in node_list:
        for n2 in node_list:
            if ((n1.name == n2.name) or 
                ([n1.name, n2.name] in edges) or 
                ([n2.name, n1.name] in edges)):
                continue
            elif coords_intersect(n1, n2):
                edges.append([n1.name, n2.name])
    ret_edges = []
    for (n1, n2) in edges:
        ret_edges.append(Edge(name=n1 + '<->' + n2, edge_nodes=[n1, n2]))           
    return ret_edges
                        
    

    



        
# ------------------------------------------------
#            Network X Helpers
#  Helpers for working between Frank Lab Apparatus and Network X,
#  typically for plotting topology and geometry of the apparatus.
# ------------------------------------------------
        
def nx_to_fl_node(node_name, attrs):
    '''
    Purpose:
        Generate a Frank Lab Node from an appropriately formated Network X node.
    Arguments:
        node_name (str): name of the node
        attrs (1 x n list of str): list of the Network X node's attributes. Must contain 'kind' and 'coords'.
    Returns: 
        A FrankLab Node representing the Network X node. The specific type of Frank Lab node returned
        depends on the 'kind' attribute of the Network X node.
    '''
    if 'kind' not in attrs:
        raise TypeError("NX node attributes must contain a 'kind' field")
    if 'coords' not in attrs:
        raise TypeError("NX node attributes must contain a 'coords' field")
    if attrs['kind']=='segment':
        return SegmentNode(name=node_name, coords=attrs['coords'])
    elif attrs['kind']=='point':
        return PointNode(name=n, coords=attrs['coords'])
    elif attrs['kind']=='polygon':
        if 'interior_coords' not in attrs:
            raise TypeError("NX 'polygon' nodes must contain a 'interior_coords' field. It can be set to None.")
        return PolygonNode(name=n, coords=attrs['coords'], 
                              interior_coords=attrs['interior_coords'])
    else:
        raise TypeError('Nodes must be of type point, segment, or polygon.')

        
def add_fl_node_to_nx_graph(fl_node, nx_graph):
    '''
    Purpose:
        Add a Frank Lab Node to an existing Network X graph.
    Arguments:
        fl_node (Node): A Frank Lab PointNode, SegmentNode, or PolygonNode
        nx_graph: A network X graph object.
    Returns:
        nx_graph: A network X graph object with a new node representing the fl_node.
        
    '''
    if isinstance(fl_node, SegmentNode):
        nx_graph.add_node(fl_node.name, coords=fl_node.coords, kind='segment')
    elif isinstance(fl_node, PointNode):
        nx_graph.add_node(fl_node.name, coords=fl_node.coords, kind='point')
    elif isinstance(fl_node, PolygonNode):
        nx_graph.add_node(fl_node.name, coords=fl_node.coords, interior_coords=fl_node.interior_coords, kind='polygon')
    else:
        print(fl_node.__class__)
        raise TypeError("'fl_node' must be of type SegmentNode, PointNode, or PolygonNode")
    return nx_graph


def plot_nx_appar_geom(nx_graph, ax=None, label_nodes=True):
    '''
    Purpose:
        Plot a Network X graph representation of a Frank Lab Apparatus geometry.
    Arguments:
        nx_graph: The Network X graph representation of a Frank Lab Apparatus.
        ax (optional, default=None): A Matplotlib Axes object.
        label_nodes (optional, default=True): Whether to label the nodes on the plot.
    Returns:
        ax: A Matplotlib Axes object with the graph plotted.
    '''
    if not ax:
        plt.figure()
        ax = plt.subplot(111)
    for n, attrs in list(nx_graph.nodes.data()):
        if attrs['kind']=='point':
            coord = attrs['coords'][0]
            ax.scatter([coord[0]], [coord[1]], color='r')
            if label_nodes:
                ax.text(coord[0], coord[1], n, fontsize=12)
        elif attrs['kind']=='segment':
            start, end = attrs['coords']
            ax.plot([start[0], end[0]], [start[1], end[1]], color='k')
            if abs(start[0] - end[0]) > abs(start[1] - end[1]):
                midx = (start[0] + end[0]) / 2
                if label_nodes:
                    ax.text(midx, start[1], n, fontsize=12)
            else:
                midy = (start[1] + end[1]) / 2
                if label_nodes:
                    ax.text(start[0], midy, n, fontsize=12)
        elif attrs['kind']=='polygon':
            poly = attrs['coords']
            xs = [e[0] for e in poly]
            ys = [e[1] for e in poly]
            ax.fill(xs, ys, color='grey', alpha=0.3)
        else:
            raise TypeError("Nodes must have 'kind' point, segment, or polygon.") 
    return ax


def plot_nx_appar_topo(nx_graph, ax=None):
    '''
    Purpose:
        Plot the topology of a Network X graph representing a Frank Lab Apparatus geometry.
    Arguments:
        nx_graph: A Network X graph representing a Frank Lab Apparatus.
        ax (optional, default=None): A Matplotlib Axes on which to plot
    Returns:
        ax: Matplotlib Axes with the topology drawn
    '''
    if not ax:
        plt.figure()
        ax = plt.subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    nx.draw(H, with_labels=True)
    return ax

    
def plot_fl_appar_geom(appar, ax=None, label_nodes=True):
    '''
    Purpose:
        Plot the geometry of a Frank Lab Apparatus using Network X.
    Arguments:
        appar: a Frank Lab Apparatus
        ax (optional, default=None): a Matplotlib axes on which to plot
        label_nodes (optional, default=True): whether or not to label the nodes
    Returns:
        H (Network X graph): the Network X graph representation of this Frank Lab Apparatus.
    '''
    if not ax:
        plt.figure()
        ax = plt.subplot(111)
    # Plot apparatus geometry
    if len(appar.nodes) > 0:
        # First we have to build a Network X graph
        H = nx.Graph(name='w-track test')
        for n in appar.nodes.values():
            add_fl_node_to_nx_graph(n, H)
        for e in appar.edges.values():
            n1, n2 = e.edge_nodes
            H.add_edge(n1, n2)
        # Then plot the Network X graph
        plot_nx_appar_geom(H, ax, label_nodes=label_nodes)
    return H