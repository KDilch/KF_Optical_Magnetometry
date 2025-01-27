
import numpy as np
def data_transform(T2, N, q, g_D, Sph, T0):
    Nq = N * q
    xcoef = np.sqrt(2 / Nq)
    ycoef = np.sqrt(2 / (g_D**2 * Nq))
    sig_v = np.sqrt(2 * Sph / (g_D**2 * Nq * T0))
    T01 = T0 / T2
    return xcoef, ycoef, sig_v, T01