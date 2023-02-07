from .node import Node

class DecisionTree:
    def __init__(self, **trepan_parameters):
        self.epselon = trepan_parameters['epselon']
        self.delta = trepan_parameters['delta']
        self.S_min = trepan_parameters['S_min']
        self.max_number_of_iternal_nodes = trepan_parameters['number_of_iternal_nodes']
        self.use_limit_on_iternal_nodes = trepan_parameters['use_limit_on_iternal_nodes']
        self.number_of_nodes = 0
        self.root = None
        
    def append_node(self, node):
        return 
