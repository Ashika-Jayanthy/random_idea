
from scipy.stats import entropy
from scipy.linalg import expm
import numpy as np
import random
import collections
from scipy.special import bernoulli
from scipy.special import gamma
from scipy.special import zeta
import sympy as syp


####


def reimann_functional(z):
    return (2**z * np.pi**z-1 * np.sin(np.pi*z/2.) * gamma(1-z) * zeta(1-z))

def coprime_probability(set_of_values):
    # euler's formula
    return 1./zeta(len(set_of_values))

def chinese_remainder_solve():
    return
