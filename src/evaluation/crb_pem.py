# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import minimize_scalar

def get_AG(th, T0):
    """
    Constructs the discrete-time state transition matrix A, state transition noise
    scaling matrix G, and process noise covariance matrix D for a given unitless
    Larmor frequency th (theta = omega_L * T2) and unitless sampling interval T0 (T0 = dt / T2).
    """
    c = np.cos(th * T0)
    s = np.sin(th * T0)
    A = np.exp(-T0) * np.array([[c, s], [-s, c]])
    G = np.sqrt(1.0 - np.exp(-2.0 * T0)) * np.eye(2)
    D = G @ G.T
    D = 0.5 * (D + D.T)
    return A, G, D


def simulate_discrete_system(nsamples, th0, T0, sig_v, x_init):
    """
    Simulates a trajectory of the discrete spin dynamics and corresponding noisy
    measurements for a fixed unitless Larmor frequency th0.
    """
    A, G, _ = get_AG(th0, T0)
    y = np.zeros(nsamples)
    x_history = np.zeros((nsamples, 2))
    x = np.array(x_init, dtype=float).copy()
    
    for k in range(nsamples):
        x_history[k, :] = x
        y[k] = x[1] + sig_v * np.random.randn()
        x = A @ x + G @ np.random.randn(2)
        
    return y, x_history


class PredictionErrorMethod:
    """
    Object-oriented Prediction Error Method (PEM) parameter estimator
    based on Kalman Filter prediction error likelihood.
    """
    def __init__(self, T0, sig_v, m_init, S_init):
        self.T0 = T0
        self.sig_v = sig_v
        self.m_init = np.array(m_init, dtype=float)
        self.S_init = np.array(S_init, dtype=float)

    def get_likelihood(self, th, y):
        """
        Evaluates the Kalman Filter prediction error likelihood function V_N(theta) (34)
        for a candidate Larmor frequency parameter th and measurement data y.
        """
        sv2 = self.sig_v**2
        A, _, D = get_AG(th, self.T0)
        q = 0.0
        ns = len(y)
        
        m = self.m_init.copy()
        S = self.S_init.copy()
        
        for k in range(ns):
            e = y[k] - m[1]  # prediction error (C = [0, 1])
            W = sv2 + S[1, 1]
            
            q = (k * q + (e**2 / W + np.log(W))) / (k + 1)
            
            # Symmetrized covariance update
            S = S - np.outer(S[:, 1], S[1, :]) / W
            
            # Mean update
            m = m + S[:, 1] * e / sv2
            
            # Prediction
            m = A @ m
            S = A @ S @ A.T + D
            
        return 0.5 * q

    def estimate(self, y, th_range):
        """
        Estimates the Larmor frequency theta by minimizing the likelihood function V_N(theta)
        over the search bounds th_range.
        """
        res = minimize_scalar(
            self.get_likelihood,
            bounds=th_range,
            method='bounded',
            args=(y,),
            options={'xatol': 1e-16}
        )
        return res.x, res.fun


class CramerRaoBounds:
    """
    Object-oriented interface for calculating analytical asymptotic and numerical
    Monte Carlo Cramér-Rao Bounds (CRB) for Larmor frequency estimation.
    """
    def __init__(self, configurator):
        self.config = configurator

    def calculate_asymptotic(self):
        """
        Calculates the steady-state asymptotic CRB in mHz.
        """
        T2 = self.config.T2
        Sph = self.config.Sph
        g_D = self.config.g_D
        N = self.config.N
        w0 = self.config.w0
        th0 = w0 * T2
        
        pre_factor = 1e3 * np.sqrt(8.0 * Sph) / (np.pi * (T2**1.5) * g_D * N)
        as_crb = pre_factor * ((th0**2 + 1.0)**1.5) / (th0 * np.sqrt(th0**4 + 3.0 * th0**2 + 6.0))
        return as_crb

    def calculate_monte_carlo(self, nsamples, ntries, ep=1e-6):
        """
        Calculates the Monte Carlo CRB in mHz by averaging likelihood gradients.
        """
        T2 = self.config.T2
        N = self.config.N
        T0 = self.config.meas_probing_rate_unitless
        sig_v = self.config.sig_v
        xc = self.config.xc
        
        th0 = self.config.w01  # True unitless Larmor frequency
        xini = np.array([0.0, 0.5 * xc * N])
        m0_m = xini.copy()
        S0_m = np.zeros((2, 2))
        
        # Instantiate internal PEM estimator
        pem = PredictionErrorMethod(T0, sig_v, m0_m, S0_m)
        ca = 1.0 / (2.0 * np.pi * T2)
        dLe = 0.0
        
        for j in range(int(ntries)):
            y, _ = simulate_discrete_system(nsamples, th0, T0, sig_v, xini)
            
            # Compute likelihood values for finite differences
            q_plus = pem.get_likelihood(th0 + ep, y)
            q_minus = pem.get_likelihood(th0 - ep, y)
            
            dL = 0.5 * nsamples * (q_plus - q_minus) / ep
            dLe = (j * dLe + dL**2) / (j + 1)
            
        crb = 1e3 * ca / np.sqrt(dLe)
        return crb
