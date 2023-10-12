import numpy as np

def prime_upper(n):
    return n*np.log(n)

class PolynomialRepresentations():
    def __init__(self,input_graph, degree=1, ring_type="Z", representation_type=''):
        self.input_graph = input_graph
        self.ring_type = ring_type
        self.representation_type = representation_type

    def compute_representation_from_graph(self):

        return representation

class Graphs():
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        self.G = {}

    def generate_graph(self, values):
        for i in range(len(self.n_nodes)):
            self.G[nodes[i]] = self.edges[i]
        return self.G
