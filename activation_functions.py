""" A module that provides the properties of the activation functions to be used.

The module provides a class of activation functions that will be used in
other modules. Some of the parameters for the activation functions are
obtained by simulation using Monte Carlo, some are by direct calculation.

Notations:
    theta_k denotes the k-th gaussian moment of the activation function.
    gamma_k denotes the k-th gaussian moment of the derivative of the
    activation function.
    thetastar and varrho are defined as in the definition of V in the
    supplementary pdf.

Useage:
    The activation functions are used in the plotting module
    to plot the curves.
    ReLU is used in the plotting module to plot boxplots and QQ-plots.

"""

import numpy as np
from numpy import sqrt, pi, exp, log
#from sympy import symbols, sympify

#y = symbols("y")
#expr_normal_density = sympify("1 / sqrt(2 * pi) * exp ( - 1 /2 * y ** 2)")


#def gm(expr, n=10 ** 5):  # Gaussian mean
#    """gm. we calculate the Gaussian mean of an expression by Monte Carlo.
#
#    This function is used to obtain the approximation of theta1, theta2
#    and gamma1.
#
#    Args:
#        expr: The expression.
#        n: The number of Monte Carlo samples.
#    """
#    return np.mean([expr.evalf(subs={y: np.random.normal()})
#                   for i in range(n)])


class ActiFunc:
    def __init__(self, theta1, theta2, gamma1, name, sigma=None, sigma_prime=None):
        self.theta1 = theta1
        self.theta2 = theta2
        self.gamma1 = gamma1
        self.name = name
        self.sigma = sigma
        self.sigma_prime = sigma_prime

    @property
    def thetastar(self):
        return sqrt(self.theta2 - self.theta1 ** 2 - self.gamma1 ** 2)

    @property
    def varrho(self):
        return self.gamma1 / self.thetastar

    def set_sigma(self, func):
        self.sigma = func

    def set_sigma_prime(self, func):
        self.sigma_prime = func


SINE = ActiFunc(0, 0.433, 0.605, "SINE", np.sin, np.cos)  # sin(y)
ReLU = ActiFunc(sqrt(2/pi)/2, 0.5, 0.5, "ReLU",
                lambda x: x * (x > 0), lambda x: (x > 0))
leaky_ReLU = ActiFunc(0.95 * sqrt(2 / pi) / 2,
                        1.0025 * 0.5,
                        1.05 * 0.5,
                        "leaky ReLU",
                        lambda x: x * (x > 0) + 0.05 * x * (x < 0),
                        lambda x: (x > 0) + 0.05 * (x < 0),
                        )
tanh = ActiFunc(0, 0.394, 0.604, "tanh",
                lambda x: (exp(x) - exp(-x))/(exp(x) + exp(-x)),
                lambda x: 1 - ((exp(x) - exp(-x))/(exp(x) + exp(-x))) ** 2,
                )
softplus = ActiFunc(0.807, 0.929, 0.500, "softplus",
                    lambda x: log(1 + exp(x)),
                    lambda x: 1 / (1 + exp(-x)),
                    )
"""af1
This is a specific activation class with gamma_1 = 0.
"""
af1 = ActiFunc(0, 2/3, 0, "$\gamma_{1}=0$",
               lambda x: -1 + np.sqrt(5) * exp(-2 * x ** 2),
               lambda x: -4 * np.sqrt(5) * x * exp(-2 * x ** 2),
               )
sigmoid = ActiFunc(0.5, 0.293, 0.206, "sigmoid",
                   lambda x: 1 / (1 + exp(-x)),
                   lambda x: exp(-x) / (1 + exp(-x)) ** 2,
                   )
xsigmoid = ActiFunc(0.208, 0.354, 0.501, "Swish",
                    lambda x: x / (1 + exp(-x)),
                    lambda x: (1 + exp(-x) - x * exp(-x)) / (1 + exp(-x)) ** 2,
                    )
list_acti_func = [ReLU, leaky_ReLU, xsigmoid, softplus, tanh, sigmoid, af1]
