import typing as tp
import numpy as np
from math import atan, pi, degrees
from numpy import e as e_const
from numba import jit


@jit(nopython=True)
def direct_task_1d(
    num_freq: int,
    first_period: float,
    geometric_step: float,
    layer_resistance: np.ndarray,
    layer_power: np.ndarray,
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Calculate one-dimensional direct task of MTZ.

    Args:
        num_freq (int): Number of periods / frequencies (NT).
        first_period (float): First period (T)
        geometric_step (float): Geometric step by periods (Q)
        layer_resistance (np.ndarray): electrical resistance by layer in Ohm * m (P)
        layer_power (np.ndarray): Power of layers in m (H)

    Returns:
        rho (np.ndarray): Rho field over model.
        phi (np.ndarray): Phi field over model.
    """
    mu_zero = 4 * pi * 1.0e-7

    num_layers = layer_resistance.shape[0]

    periods = [first_period * geometric_step ** i for i in range(num_freq)]

    # Vectorize as much as possible
    indexes = np.arange(num_layers - 1, 0, -1)
    indexes_ = indexes - 1
    k_array = np.sqrt(-1j * mu_zero / layer_resistance[indexes_])
    a_array = np.sqrt(layer_resistance[indexes_] / layer_resistance[indexes])

    rho_t, phi_t = np.empty(num_freq), np.empty(num_freq)
    for i, period in enumerate(periods):
        r = 1.0
        omega = (2 * pi / period) ** 0.5
        for m in range(num_layers - 1, 0, -1):
            k = k_array[num_layers - m - 1]
            a = a_array[num_layers - m - 1]
            b = e_const ** (-2 * k * omega * layer_power[m - 1]) * (r - a) / (r + a)
            r = (1 + b) / (1 - b)
        rho_t[i] = layer_resistance[0] * abs(r) ** 2
        phi_t[i] = degrees(atan(r.imag / r.real))

    # Additional vectorization out of loop
    phi_t = phi_t - 45

    return rho_t, phi_t
