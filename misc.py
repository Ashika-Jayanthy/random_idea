import numpy as np

def prime_upper(n):
    return n*np.log(n)

# options: 1) represent by roots + relationship between roots (over different rings). are galois groups fundamental? if not,
# find a different representation
# 2) something else  <- keep thinking.

class PolynomialRepresentations():
    def __init__(self,input_graph, degree=1, ring_type="Z", representation_type=''):
        self.input_graph = input_graph
        self.ring_type = ring_type
        self.representation_type = representation_type

    def compute_representation_from_graph(self):

        return representation

class RandomGraphs():
    def __init__(self,average_degree,n_nodes,n_edges):
        self.average_degree = average_degree
        self.n_nodes = n_nodes
        self.n_edges = n_edges

    def generate_random_graph(self):
        return

    def polynomial_from_graph(self):
        return
