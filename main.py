import numpy as np
import scipy as sp
import mendeleev as mdl
import random
import collections
from max_entropy import *

icomplex = complex(0,1)
# check again later
primes_list = np.array([
2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199,
211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293,
307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397,
401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499,
503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599,
601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691,
701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797,
809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887,
907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997
])

mersenne_primes = np.array([
3,7,31,127,8191
])

ket_zero = np.array([1,0])
ket_one = np.array([0,1])

max_natural = 100
rotation_generators = 0.5 * np.array([
[[0.,1.],[1.,0.]],
[[0.,-icomplex],[icomplex,0.]],
[[1.,0.],[0.,-1.]]
])

def ladder_operators(type,input):
    if type == 'J_plus':
        return
    elif type == 'J_minus':
        return

def random_rotations(generators,size=10):
    return np.random.choice(generators, size=size)

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


class Prop():
    def __init__(self, values, type):
        self.values = values
        self.type = type

     def topdown(self):
         self.rationals =
         self.integers =
         self.naturals =
         self.binary_equivalences =
         return

    def bottomup(self):

        return

    def run(self):
        if self.type == 'topdown':
            self.topdown()
            return self.binary_equivalences
        elif self.type == 'bottomup':
            return self.primes

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
        self.perceptron_volume = np.sum(np.multiply(heaviside_smooth(self.X))*rho_function(self.weights))
        return

    def run_all(self):
        self.learn_weights
        self.X = (np.absolute(np.vdot(self.input_vector, self.weights))**2 - self.threshold) / self.hilbert_space_dim
        self.calculate_volume
        return self.perceptron_volume


class Particles(Perceptron):
    def __init__(self):


    def map_to_element(self):
        mdl.
        return

    @property
    def spin_angular_momentum(self):
        return


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
