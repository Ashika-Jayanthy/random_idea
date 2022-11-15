import numpy as np

def prime_upper(n):
    return n*np.log(n)

# options: 1) represent by roots + relationship between roots (over different rings). are galois groups fundamental? if not,
# find a different representation
# 2) something else  <- keep thinking.
# 2a) how to order sequence of polynomials. with index set primes ordered by size less than n; or for number fields prime ideals (how to order?).

class PolynomialRepresentations():
    def __init__(self,input_graph, degree=1, ring_type="Z", representation_type=''):
        self.input_graph = input_graph
        self.ring_type = ring_type
        self.representation_type = representation_type

    def compute_representation_from_graph(self):

        return representation

class Graphs():
    #
    def __init__(self,average_nnodes, graph_type="random"):
        self.average_nnodes = average_nnodes
        self.n_nodes = ceil(np.random.normal(self.average_nnodes,size=1))
        self.G = {}
        for i in range(self.n_nodes):
            self.G[i] = []

    def generate_random_graph(self, values):

        return self.G
