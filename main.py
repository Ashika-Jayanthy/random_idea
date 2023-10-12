import numpy as np
import random
import collections
from max_entropy import *
from scipy.linalg import expm
from scipy.special import bernoulli
from scipy.special import gamma
from scipy.special import zeta
import math

def prime_upper(n):
    return n*np.log(n)

def reimann_functional(z):
    return (2**z * np.pi**z-1 * np.sin(np.pi*z/2.) * gamma(1-z) * zeta(1-z))

def coprime_probability(set_of_values):
    # euler's formula
    return 1./zeta(len(set_of_values))

def extended_gcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, x, y = extended_gcd(b % a, a)
        return (g, y - (b // a) * x, x)

def mod_inverse(a, m):
    g, x, _ = extended_gcd(a, m)
    if g != 1:
        raise ValueError("Modular inverse does not exist")
    else:
        return x % m

def chinese_remainder_theorem(n, a):
    prod = math.prod(n)
    result = 0
    for n_i, a_i in zip(n, a):
        m = prod // n_i
        m_inv = mod_inverse(m, n_i)
        result += a_i * m * m_inv
    
    return result % prod


def heaviside_smooth(x,k=1):
    # logistic
    return 1. / 1. + np.exp(-2*k*x)

def rho_function(weights, dims,perceptron_type):
    if perceptron_type == "Ising":
        y = np.multiply(np.heaviside(weights - 1) + np.heaviside(weights + 1), axis=1)
    elif perceptron_type == "Spherical":
        y = np.heaviside(np.absolute(weights) **2 - dims)
    return  y / np.sum(y)


class Perceptron():
    def __init__(self, input_vector, expectation, perceptron_type = "Spherical"):
        assert type(input_vector) == np.complex64
        assert input_vector.shape[0] == 2
        assert np.isclose(np.absolute(input_vector[:,0])**2, np.absolute(input_vector[:,1])**2,1.)
        self.input_vector = input_vector
        self.space_dim = self.input_vector.shape[1]
        self.threshold = np.zeros(self.input_vector.shape[1])
        self.expectation = expectation
        self.perceptron_type = perceptron_type
        
    def learn_weights(self):
        self.weights = MaxEntropy(self.input_vector,self.expectation) # placeholder
        return

    def calculate_volume(self):
        # wrong
        self.perceptron_volume = np.sum(np.multiply(heaviside_smooth(self.X))*rho_function(self.weights, self.space_dim, self.perceptron_type))
        return

    def run_all(self):
        self.learn_weights()
        self.X = (np.absolute(np.vdot(self.input_vector, self.weights))**2 - self.threshold) / self.hilbert_space_dim
        self.calculate_volume()
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
