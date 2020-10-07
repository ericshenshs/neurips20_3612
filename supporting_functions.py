""" A module that provides three supporting functions to help plot Figure 1
to Figure 3 in the paper.

"""

import numpy as np
import activation_functions as af


def square_length(psid, psip, lamb=10 ** -3, rho=2, af=af.ReLU):
    """ A function that returns the limiting squared length V and its limit R.

    The calculations of V and R are given in the supplementary pdf.

    Args:
        psid (float or str): The psi_d. If "+oo", the function returns
                             the limit R.
        psip (float): The psi_p. e.g., 1/3.
        lamb (float): The lambda. e.g., 10 ** -3.
        rho  (float): The rho. e.g., 2.
        af   (ActiFunc): The activation function to be used. e.g. af.ReLU.

    Returns:
        The value of V. If psid is "+oo", returns the value of R.

    Raises:
        ValueError: If the unique solution of the quartic equation
                    can not be found.

    """
    def helper(psip, lamb, rho, af):
        """ A helper function of "square_length" to calculate R.
        """
        varrho, thetastar = af.varrho, af.thetastar
        barlamb = lamb / (thetastar ** 2)
        psi2 = 1 / psip
        if varrho < 10 ** (-10):
            chi = - psi2 / (1 + psi2 * barlamb)
        else:
            b = - varrho ** (-2) + (psi2 - 1) / (1 + psi2 * barlamb)
            c = - psi2 / (1 + psi2 * barlamb) * varrho ** (-2)
            chi = (1/2) * (- b - (b ** 2 - 4 * c) ** (1/2))
        scrQ = - chi * barlamb
        scrA1 = rho / (1 + rho) * (- chi ** 2 * (chi * varrho ** 4 - chi * varrho ** 2 + psi2 * varrho ** 2 + varrho ** 2 - chi * psi2 * varrho ** 4 + 1))\
            + 1 / (1 + rho) * (chi ** 2 * (chi * varrho ** 2 - 1) * (chi ** 2 * varrho ** 4 - 2 * chi * varrho ** 2 + varrho ** 2 + 1))
        scrA0dpsi1 = (psi2 - 1) * chi ** 3 * varrho ** 6 + (1 - 3 * psi2) * chi ** 2 * varrho ** 4 + 3 * psi2 * chi * varrho ** 2 - psi2
        scrL = scrQ * ((rho / (1 + rho)) * (1 / (1 - chi * varrho ** 2)) + (1 / (1 + rho)))
        square_length = (scrL - barlamb * scrA1 / scrA0dpsi1) / (scrQ ** 2)
        return square_length
    if psid == "+oo":
        return helper(psip, lamb, rho, af)

    varrho, thetastar = af.varrho, af.thetastar
    psi1 = psid / psip
    psi2 = 1 / psip
    xibar = complex(0, (psi1 * psi2 * lamb) ** (1/2) / thetastar)
    c = (psi1 - psi2) / (- xibar)
    coef = [(- varrho**2),
            (- 2 * c * varrho ** 2 - varrho ** 2 * xibar),
            (varrho ** 2 + 1 - psi2 * varrho ** 2 - c * varrho ** 2 * (c + xibar)),
            (c * varrho ** 2 + c + xibar - psi2 * c * varrho ** 2),
            (psi2),
            ]
    raw_b = np.roots(coef)
    true_ab = []
    for b in raw_b:
        if b.imag > 0 and (b + c).imag > 0:
            true_ab.append((b+c, b))
    if len(true_ab) == 0:
        raise ValueError('Can not find solution.')
    if len(true_ab) > 1:
        raise ValueError('The solution is not unique')
    else:
        a, b = true_ab[0]
    chi = a * b
    scrQ = 1 + psip * (chi + chi * varrho ** 2 / (1 - chi * varrho ** 2))
    scrL = scrQ * ((rho / (1 + rho)) * (1 / (1 - chi * varrho ** 2)) + (1 / (1 + rho)))
    scrA1 = rho / (1 + rho) * (- chi ** 2 * (chi * varrho ** 4 - chi * varrho ** 2 + psi2 * varrho ** 2 + varrho ** 2 - chi * psi2 * varrho ** 4 + 1))\
        + 1 / (1 + rho) * (chi ** 2 * (chi * varrho ** 2 - 1) * (chi ** 2 * varrho ** 4 - 2 * chi * varrho ** 2 + varrho ** 2 + 1))
    scrA0 = - chi ** 5 * varrho ** 6 + 3 * chi ** 4 * varrho ** 4 + (psi1 * psi2 - psi2 - psi1 + 1) * chi ** 3 * varrho ** 6 - 2 * chi ** 3 * varrho ** 4 - 3 * chi ** 3 * varrho ** 2\
        + (psi1 + psi2 - 3 * psi1 * psi2 + 1) * chi ** 2 * varrho ** 4 + 2 * chi ** 2 * varrho ** 2 + chi ** 2 + 3 * psi1 * psi2 * chi * varrho ** 2 - psi1 * psi2
    scrA = scrA1 / scrA0
    square_length = (scrL - psid / psip * lamb / (thetastar ** 2) * scrA) / (scrQ ** 2)
    return square_length.real


def simu_square_length(psid, psip, lamb=10**(-3), rho=2, n_iter=20, n=300,
                       af=af.ReLU):
    """ A function that generates a list of squared length nL^2 from a random
    quadratic model.

    The noise level of the non-linear part is the same as the noise
    level of the noise epsilon. The total signal-noise level is
    fixed to be 1, that is, sigma_beta^2 + sigma_ep^2 + sigma_NL^2 = 1.
    There are no interception in the simulation.

    Args:
        psid (float): d/n.
        psip (float): p/n.
        lamb (float): lambda.
        rho  (float): rho.
        n_iter (int): the length of the returning list.
        n      (int): the dimension n.
        af(ActiFunc): The activation (from module acti) to be used.

    Returns:
        A list of values nL^2.

    """
    rho = 2
    sigma_beta2 = 1 / (rho + 1) * rho
    sigma_ep2 = 1 / (rho + 1) * 1/2
    sigma_NL2 = 1 / (rho + 1) * 1/2
    d = int(np.floor(n * psid))
    p = int(np.floor(n * psip))
    sigma = af.sigma
    square_length_l = []
    for iterator in range(n_iter):
        beta = np.random.normal(size=p)
        beta = beta / np.linalg.norm(beta) * ((sigma_beta2) ** (1/2))
        eps = np.random.normal(size=n, scale=sigma_ep2 ** (1/2))
        X = np.random.normal(size=(n, p))
        W = np.random.normal(size=(d, p), scale=(1/p) ** (1/2))
        G = np.random.normal(size=(p, p))
        tG = np.trace(G)
        nonlinear = sigma_NL2 ** (1/2) / p * np.asarray([X[i].T @ G @ X[i] - tG for i in range(n)])
        y = X @ beta + nonlinear + eps
        A = sigma(X @ W.T)
        alpha = np.linalg.inv(n * lamb * d / p * np.eye(d) + A.T @ A) @ A.T @ y
        H = A @ np.linalg.inv(n * lamb * d / p * np.eye(d) + A.T @ A) @ A.T
        square_length_l.append(1 / n * np.linalg.norm(y - A @ alpha) ** 2 / ((1 - 1 / n * np.trace(H)) ** 2))
    return square_length_l


def sample_qqplot(psid, psip, lamb=10**(-3), rho=2, n_iter=300, n=300,
                  af=af.ReLU):
    """ A function that generates a sample of zeta/||y - A\alpha||_^2 from the
    random quadratic model.

    The noise level of the non-linear part is the same as the noise level of
    the noise epsilon. The total signal-noise level is fixed to be 1,
    that is, sigma_beta^2 + sigma_ep^2 + sigma_NL^2 = 1.
    There are no interception in the simulation.

    Args:
        psid (float): d/n.
        psip (float): p/n.
        lamb (float): lambda.
        rho  (float): rho.
        n_iter (int): the length of the returning list (sample size).
        n      (int): the dimension n.
        af(ActiFunc): The activation (from module acti) to be used.

    Returns:
        A list of the asymptotic normal quantities.

    """
    rho = 2
    sigma_beta2 = 1 / (rho + 1) * rho
    sigma_ep2 = 1 / (rho + 1) * 1/2
    sigma_NL2 = 1 / (rho + 1) * 1/2
    d = int(np.floor(n * psid))
    p = int(np.floor(n * psip))
    sigma, sigma_prime = af.sigma, af.sigma_prime
    sample = []
    for iterator in range(n_iter):
        beta = np.random.normal(size=p)
        beta = beta / np.linalg.norm(beta) * ((sigma_beta2) ** (1/2))
        eps = np.random.normal(size=n, scale=sigma_ep2 ** (1/2))
        X = np.random.normal(size=(n, p))
        W = np.random.normal(size=(d, p), scale=(1/p) ** (1/2))
        u0 = np.asarray([int(i == 0) for i in range(p)])
        G = np.random.normal(size=(p, p))
        tG = np.trace(G)
        nonlinear = sigma_NL2 ** (1/2) / p * np.asarray([X[i].T @ G @ X[i] - tG for i in range(n)])
        y = X @ beta + nonlinear + eps
        z0 = X @ u0
        A = sigma(X @ W.T)
        alpha = np.linalg.inv(n * lamb * d / p * np.eye(d) + A.T @ A) @ A.T @ y
        f = y - A @ alpha
        In = np.eye(n)
        H = A @ np.linalg.inv(n * lamb * d / p * np.eye(d) + A.T @ A) @ A.T
        T0 = - (In - H) * np.diag(sigma_prime(X @ W.T) @ np.diag(alpha) @ W @ u0)
        T1 = - A @ np.linalg.inv(n * lamb * d / p * np.eye(d) + A.T @ A) @ np.diag(W @ u0) @ sigma_prime(W @ X.T) @ np.diag(f)
        TL = (In - H) * (u0.T @ beta)
        xi = z0.T @ f - np.trace(T0 + T1 + TL)
        Z = xi / np.linalg.norm(f)
        sample.append(Z)
    return sample
