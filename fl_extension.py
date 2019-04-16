from hdmf.utils import docval, getargs, popargs, call_docval_func
from pynwb import register_class, load_namespaces
from pynwb.file import MultiContainerInterface, NWBContainer, NWBDataInterface


# ------------------------------------------------
# Load the Frank Lab behavioral extensions
# See 'create_franklab_spec.ipynb' for details on the generation of the spec file.
# ------------------------------------------------
ns_path = "franklab_apparatus.namespace.yaml"
load_namespaces(ns_path)


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
        edge_nodes (tuple, list): the names of the two Node objects connected by this edge (e.g. [node1 name, node2 name])
        
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
            {'name': 'coords', 'type': ('array_data', 'data'), 'doc': 'vertices and exterior control points (e.g. doors) of this polygon'},
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
