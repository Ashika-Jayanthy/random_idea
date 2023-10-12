import numpy as np
import random
import collections
from max_entropy import *
from scipy.linalg import expm
from scipy.special import bernoulli
from scipy.special import gamma
from scipy.special import zeta


####


def reimann_functional(z):
    return (2**z * np.pi**z-1 * np.sin(np.pi*z/2.) * gamma(1-z) * zeta(1-z))

def coprime_probability(set_of_values):
    # euler's formula
    return 1./zeta(len(set_of_values))

def chinese_remainder_solve():
    return

icomplex = complex(0,1)

ket_zero = np.array([1,0])
ket_one = np.array([0,1])

rotation_generators = 0.5 * np.array([
[[0.,1.],[1.,0.]],
[[0.,-icomplex],[icomplex,0.]],
[[1.,0.],[0.,-1.]]
])


def heaviside_smooth(x,k=1):
    # logistic
    return 1. / 1. + np.exp(-2*k*x)

def matrix_multiply(a,b):
    n1,m1 = a.shape
    n2,m2 = b.shape
    ans = np.zeros((n1,m2),dtype="complex128")
    for row in range(n1):
        row_value = a[row]
        for column in range(m2):
            column_value = b.T[column]
            ans[row,column] = np.vdot(row_value,column_value)
    return ans

def commutator(a,b):
    return matrix_multiply(a,b) - matrix_multiply(b,a)

def coprime_probability(set_of_values):
    # euler's formula
    return 1./zeta(len(set_of_values))


class Perceptron():
    def __init__(self, input_vector, expectation, quantum_perceptron_type = "Spherical"):
        assert type(input_vector == np.complex64)
        assert input_vector.shape[0] == 2
        assert np.isclose(np.absolute(self.input_vector[:,0])**2, np.absolute(np.input_vector[:,1])**2,1.)
        self.input_vector = input_vector
        self.hilbert_space_dim = self.input_vector.shape[1]
        self.threshold = np.zeros(self.input_vector.shape[1])
        self.expectation = expectation
        # check axes
        def rho_function(weights, dims):
            if quantum_perceptron_type == "Ising":
                y = np.multiply(np.heaviside(weights - 1) + np.heaviside(weights + 1), axis=1)
            elif quantum_perceptron_type == "Spherical":
                y = np.heaviside(np.absolute(weights) **2 - dims)
            return  y / np.sum(y)

    def learn_weights(self):
        self.weights = MaxEntropy(self.input_vector,self.expectation)
        return

    def calculate_volume(self):
        # wrong
        self.perceptron_volume = np.sum(np.multiply(heaviside_smooth(self.X))*rho_function(self.weights))
        return

    def run_all(self):
        self.learn_weights
        self.X = (np.absolute(np.vdot(self.input_vector, self.weights))**2 - self.threshold) / self.hilbert_space_dim
        self.calculate_volume
        return self.perceptron_volume


class C2:
    def __init__(self, y = np.array([[complex(0,0), complex(0,0)], [complex(1,0), complex(0,0)]])):
        self.y = y
    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        if not np.isclose(np.square(np.abs(value[0])) + np.square(np.abs(value[1])), 1.0):
            raise ValueError
        self._y = value
