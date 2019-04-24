import numpy as np
import networkx as nx
import nspike_helpers as ns
from fl_extension import *


# ------------------------------------------------
# Helper functions for working with franklab_apparatus objects
# ------------------------------------------------

def get_apparatus_from_linpos(linpos_this_epoch, name='some apparatus', conversion=1.0):
    '''
    Purpose: 
        Make an Apparatus object representing a linearized apparatus from animal Bon data
        extracted from Frank Lab Filter Framework. This method might not work for all 
        Frank Lab Filter Framework data.
    
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


def coords_intersect(n1, n2, tol=0.01):
    for c1 in n1.coords:
        for c2 in n2.coords:
            if (abs(c1[0] - c2[0]) <= tol and abs(c1[1] - c2[1]) <= tol):
                return True
    return False

def find_edges(node_list):
    edges = []
    for n1 in node_list:
        for n2 in node_list:
            if ((n1.name == n2.name) or ([n1, n2] in edges) or ([n2, n1] in edges)):
                continue
            elif coords_intersect(n1, n2):
                edges.append([n1.name, n2.name])
    ret_edges = []
    for (n1, n2) in edges:
        ret_edges.append(Edge(name=n1 + '<->' + n2, edge_nodes=[n1, n2]))           
    return ret_edges
                        
    
def nx_to_fl_node(node_name, attrs):
    if 'kind' not in attrs:
        raise TypeError("NX node attributes must contain a 'kind' field")
    if 'coords' not in attrs:
        raise TypeError("NX node attributes must contain a 'coords' field")
    if attrs['kind']=='segment':
        if 'intermediate_coords' not in attrs:
            raise TypeError("NX 'segment' nodes must contain a 'intermediate_coords' field. It can be set to None.")
        return SegmentNode(name=node_name, coords=attrs['coords'], 
                              intermediate_coords=attrs['intermediate_coords'])
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
    if isinstance(fl_node, SegmentNode):
        nx_graph.add_node(fl_node.name, coords=fl_node.coords, intermediate_coords=None, kind='segment')
    elif isinstance(fl_node, PointNode):
        nx_graph.add_node(fl_node.name, coords=fl_node.coords, kind='point')
    elif isinstance(fl_node, PolygonNode):
        nx_graph.add_node(fl_node.name, coords=fl_node.coords, interior_coords=fl_node.interior_coords, kind='polygon')
    else:
        print(fl_node.__class__)
        raise TypeError("'fl_node' must be of type SegmentNode, PointNode, or PolygonNode")
    return nx_graph


def plot_nx_appar_geom(nx_graph, ax=None, label_nodes=True):
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
    if not ax:
        plt.figure()
        ax = plt.subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    nx.draw(H, with_labels=True)

    
def plot_fl_appar_geom(appar, ax=None, label_nodes=True):
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


def parse_franklab_behavior_data(data_dir, animal, day):
    behavior_files = ns.get_files_by_day(data_dir, animal.lower(), 'pos')
    behavior_dict = ns.loadmat_ff(behavior_files[day], 'pos')
    return behavior_dict[day]


def parse_franklab_task_data(data_dir, animal, day):
    task_files = ns.get_files_by_day(data_dir, animal.lower(), 'task')
    task_dict = ns.loadmat_ff(task_files[day], 'task')
    return task_dict[day]


def get_sleepbox_nodes():    
    # The vertices of the sleepbox are hardcoded here for demo purposes.
    min_x, min_y = (0.6966000000000001, 0.64395)
    max_x, max_y = (1.5957000000000003, 1.47015)
    
    # The sleepbox consists of a single PolygonNode
    sleep_box_polygon_node = PolygonNode(name='Sleep box polygon', 
                                   coords=[[min_x, min_y], 
                                           [min_x, max_y], 
                                           [max_x, max_y], 
                                           [max_x, min_y]],
                                   interior_coords=None)
    
    # Return as a list, as this is what the Apparatus constructor will expect
    return [sleep_box_polygon_node]


def get_wtrack_A_nodes():
    # Track segments (coords are hard coded here for demo purposes)
    segment1 = SegmentNode(name='segment1',
                               coords=[[1.27202765, 1.10184211],
                                       [0.62004608, 1.14605263]])
    segment2 = SegmentNode(name='segment2',
                               coords=[[0.62004608, 1.14605263],
                                       [0.64990783, 1.47763158]])
    segment3 = SegmentNode(name='segment3',
                               coords=[[0.64990783, 1.47763158],
                                       [1.30686636, 1.41868421]])
    segment4 = SegmentNode(name='segment4',
                               coords=[[0.62004608, 1.14605263],
                                       [0.61258065, 0.81447368]])
    segment5 = SegmentNode(name='segment5',
                               coords=[[0.61258065, 0.81447368],
                                       [1.21479263, 0.76657895]])
    # Reward wells (coords are hard coded here for demo purposes)
    well1 = PointNode(name='well1',
                          coords=[[1.27202765, 1.10184211]])
    well2 = PointNode(name='well2',
                          coords=[[1.30686636, 1.41868421]])
    well3 = PointNode(name='well3',
                          coords=[[1.21479263, 0.76657895]])
    
    return [segment1, segment2, segment3, segment4, segment5, well1, well2, well3]
    
def get_wtrack_B_nodes():
    # Track segments (coords are hard coded here for demo purposes)
    segment1 = SegmentNode(name='segment1',
                               coords=[[1.95884793, 1.25842105],
                                       [1.91903226, 0.64578947]])
    segment2 = SegmentNode(name='segment2',
                               coords=[[1.91903226, 0.64578947],
                                       [1.61792627, 0.66157895]])
    segment3 = SegmentNode(name='segment3',
                               coords=[[1.61792627, 0.66157895],
                                       [1.64529954, 1.27421053]])
    segment4 = SegmentNode(name='segment4',
                           coords=[[1.91903226, 0.64578947],
                                   [2.21267281, 0.63]])
    segment5 = SegmentNode(name='segment5',
                           coords=[[2.21267281, 0.63],
                                   [2.25995392, 1.27105263]])
    # Reward wells (coords are hard coded here for demo purposes)
    well1 = PointNode(name='well1',
                      coords=[[1.95884793, 1.25842105]])
    well2 = PointNode(name='well2',
                      coords=[[1.64529954, 1.27421053]])
    well3 = PointNode(name='well3',
                      coords=[[2.25995392, 1.27105263]])
    
    return [segment1, segment2, segment3, segment4, segment5, well1, well2, well3]


def get_franklab_task(epoch_metadata, behav_mod): 
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
        
def get_franklab_apparatus(epoch_metadata, behav_mod):
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
        
def get_exposure_num(epoch_metadata):
    if 'exposure' in epoch_metadata.keys():
        return epoch_metadata['exposure'][0][0]
    else:
        return 'NA'
      
def parse_franklab_tetrodes(data_dir, animal, day):
    tetinfo_filename = "%s/%s%s" % (data_dir, animal.lower(), "tetinfo.mat")
    tets_dict = ns.loadmat_ff(tetinfo_filename, 'tetinfo')
    # only look at first epoch (1-indexed) because rest are duplicates
    return tets_dict[day][1] 


def get_franklab_tet_location(tet):
    # tet.area/.subarea are 1-d arrays of Unicode strings
    # cast to str() because h5py barfs on numpy.str_ type objects?
    # ---------
    area = str(tet['area'][0]) if 'area' in tet else '?'
    if 'sub_area' in tet: 
        sub_area = str(tet['sub_area'][0])
        location = area + ' ' + sub_area
    else:
        sub_area = '?'
        location = area 
    return location
        
        
def get_franklab_tet_coord(tet):
    # tet.depth is a 1x1 cell array in tetinfo struct for some reason (multiple depths?)
    # (which contains the expected 1x1 numeric array)
    if 'depth' in tet:
        coord = [np.nan, np.nan, tet['depth'][0, 0][0, 0] / 12 / 80 * 25.4]
    else:
        coord = [np.nan, np.nan, np.nan]
    return coord