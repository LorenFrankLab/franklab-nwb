import numpy as np
import networkx as nx
from hdmf.utils import docval, getargs, popargs, call_docval_func
import pynwb
from pynwb import register_class, load_namespaces
from pynwb.file import MultiContainerInterface, NWBContainer, NWBDataInterface


# ------------------------------------------------
# Load the Frank Lab behavioral extensions
# See 'create_franklab_spec.ipynb' for details on the generation of the spec file.
# ------------------------------------------------
ns_path = "franklab_apparatus.namespace.yaml"
pynwb.load_namespaces(ns_path)


# ------------------------------------------------
# Python classes implementing 'franklab_apparatus.extensions.yaml'
# These classes are required to use the Frank Lab extensions with PyNWB.
# ------------------------------------------------
@register_class('Node', 'franklab_apparatus')
class Node(NWBContainer):
    '''
    Purpose: 
        A generic graph node. Subclass for more specific types of nodes.
    
    Arguments: 
        name (str): name of this node
    '''

    __nwbfields__ = ('name',)
        
    @docval({'name': 'name', 'type': str, 'doc': 'name of this node'})
    def __init__(self, **kwargs):
        super(Node, self).__init__(name=kwargs['name'])
        
@register_class('Edge', 'franklab_apparatus')
class Edge(NWBContainer):
    '''
    Purpose: 
        An undirected edge connecting two nodes in a graph.
    
    Arguments: 
        name (str): name of this edge
        edge_nodes (tuple, list): the two Node objects connected by this edge (e.g. [node1, node2])
        
    '''

    __nwbfields__ = ('name', 'edge_nodes')
    
    @docval({'name': 'name', 'type': str, 'doc': 'name of this segement node'},
            {'name': 'edge_nodes', 'type': ('array_data', 'data'), 
             'doc': 'the names of the two nodes in this undirected edge'})
    def __init__(self, **kwargs):
        super(Edge, self).__init__(name=kwargs['name'])
        self.edge_nodes = kwargs['edge_nodes']
        
@register_class('PointNode', 'franklab_apparatus')
class PointNode(Node):
    '''
    Purpose: 
        A node representing a single point in 2D space.
    
    Arguments: 
        name (str): name of this point node
        coords (1x2 list/array): x/y coordinate of this point node
        
    '''
    
    __nwbfields__ = ('name', 'coords')

    @docval({'name': 'name', 'type': str, 'doc': 'name of this point node'},
            {'name': 'coords', 'type': ('array_data', 'data'), 'doc': 'coords of this node'})
    def __init__(self, **kwargs):
        super(PointNode, self).__init__(name=kwargs['name'])
        self.coords=kwargs['coords']
        
@register_class('SegmentNode', 'franklab_apparatus')
class SegmentNode(Node):
    '''
    Purpose: 
        A node representing a line segment in 2D space.
    
    Arguments: 
        name (str): name of this segment node
        coords (2x2 list/array): x/y coordinates of the start and end points of this segment
        
    '''
    
    __nwbfields__ = ('name', 'coords')
    
    @docval({'name': 'name', 'type': str, 'doc': 'name of this segement node'},
            {'name': 'coords', 'type': ('array_data', 'data'), 'doc': 'start/stoop coords of this segment'})
    def __init__(self, **kwargs):
        super(SegmentNode, self).__init__(name=kwargs['name'])
        self.coords=kwargs['coords']
        
@register_class('PolygonNode', 'franklab_apparatus')
class PolygonNode(Node):
    '''
    Purpose: 
        A node representing a polygon area in 2D space.
    
    Arguments: 
        name (str): name of this polygon node
        coords (nx2 list/array): x/y coordinates of the vertices and any other external 
            control points, like doors, defining the boundary of this polygon, 
            ordered clockwise starting from any point
        internal_coords (nx2 list/array, optional): x/y coordinates of any internal points inside the boundaries
            of this polygon (e.g. interior wells, objects)
        
    '''

    __nwbfields__ = ('name', 'coords', 'interior_coords')

    @docval({'name': 'name', 'type': str, 'doc': 'name of this polygon node'},
            {'name': 'coords', 'type': ('array_data', 'data'), 'doc': 'vertices and exterior control points (i.e. doors) of this polygon'},
            {'name': 'interior_coords', 'type': ('array_data', 'data'), 
             'doc': 'coords inside this polygon area (i.e. wells, objects)', 'default': None})
    def __init__(self, **kwargs):
        super(PolygonNode, self).__init__(name=kwargs['name'])
        self.coords=kwargs['coords']
        self.interior_coords = kwargs['interior_coords']
 

@register_class('Apparatus', 'franklab_apparatus')
class Apparatus(MultiContainerInterface):
    """
    Purpose:
        Topological graph representing connected components of a beahvioral apparatus.
    
    Arguments:
        name (str): name of this apparatus
        nodes (list): list of Node objects contained in this apparatus
        edges (list): list of Edge objects contained in this apparatus
        
    """
    
    __nwbfields__ = ('name', 'edges', 'nodes')
        
    __clsconf__ = [
        {
        'attr': 'edges',
        'type': Edge,
        'add': 'add_edge',
        'get': 'get_edge'
        },
        {
        'attr': 'nodes',
        'type': Node,
        'add': 'add_node',
        'get': 'get_node'
        }
    ]
    __help = 'info about an Apparatus'
    

@register_class('Task', 'franklab_apparatus')
class Task(NWBDataInterface):
    """
    Purpose:
        A behavioral task and the associated apparatus (i.e. track/maze) 
        on which the task was performed 
    
    Arguments:
        name (text): name of this task
        description (text): detailed description of this task        
    """
    
    __nwbfields__ = ('name', 'description')  
    
    @docval({'name': 'name', 'type': str, 'doc': 'name of this task'},
            {'name': 'description', 'type': str, 'doc': 'detailed description of this task'})
    def __init__(self, **kwargs):
        super(Task, self).__init__(name=kwargs['name'])
        self.description=kwargs['description']        
  

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
    if str(fl_node.__class__) == "<class 'fl_apparatus.SegmentNode'>":
        nx_graph.add_node(fl_node.name, coords=fl_node.coords, intermediate_coords=None, kind='segment')
    elif str(fl_node.__class__) == "<class 'fl_apparatus.PointNode'>":
        nx_graph.add_node(fl_node.name, coords=fl_node.coords, kind='point')
    elif isinstance(fl_node, PolygonNode):
        nx_graph.add_node(fl_node.name, coords=fl_node.coords, interior_coords=fl_node.interior_coords, kind='polygon')
    else:
        raise TypeError("'fl_node' must be of type FL_SegmentNode, FL_PointNode, or FL_PolygonNode")
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