import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

class PolynomialRepresentations():
    def __init__(self, input_graph, degree=1):
        self.input_graph = input_graph
        self.degree = degree
        self.representation = None

    def compute_representation_from_graph(self):
        X = self.input_graph['nodes'] # placeholder
        y = self.input_graph['edges'] # placeholder
        
        poly_features = PolynomialFeatures(degree=self.degree)
        X_poly = poly_features.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)

        coefficients = model.coef_

        self.representation = coefficients
        return self.representation


class Graphs():
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        self.G = {}

    def generate_graph(self):
        for i in range(len(self.nodes)):
            self.G[self.nodes[i]] = self.edges[i]
        return self.G
