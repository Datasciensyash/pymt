import typing as tp
import numpy as np
from math import atan, pi, degrees
from numpy import e as e_const

from numba import jit, prange


@jit(nopython=True)
def direct_task_1d(
        periods: np.ndarray,
        layer_resistance: np.ndarray,
        layer_power: np.ndarray,
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Calculate one-dimensional direct task of MT.

    Args:
        periods (np.ndarray): Array of periods, e.g. [0.01, 0.005, ...]
        layer_resistance (np.ndarray): electrical resistance by layer in Ohm * m
        layer_power (np.ndarray): Power of layers in m

    Returns:
        rho (np.ndarray): Rho field over model.
        phi (np.ndarray): Phi field over model.
    """
    mu_zero_j = (-1j * (4 * pi * 1.0e-7))

    num_layers = layer_resistance.shape[0]

    # Vectorize as much as possible
    indexes = np.arange(num_layers - 1, 0, -1)
    k_array = np.sqrt(mu_zero_j / layer_resistance[indexes - 1])
    a_array = np.sqrt(layer_resistance[indexes - 1] / layer_resistance[indexes])

    rho_t, phi_t = np.empty(periods.shape[0]), np.empty(periods.shape[0])
    for i in range(periods.shape[0]):
        r = 1.0
        omega = (2 * pi / periods[i]) ** 0.5
        for m in range(num_layers - 1, 0, -1):
            k = k_array[num_layers - m - 1]
            a = a_array[num_layers - m - 1]
            b = e_const ** (-2 * k * omega * layer_power[m - 1]) * (r - a) / (r + a)
            r = (1 + b) / (1 - b)
        rho_t[i] = layer_resistance[0] * np.abs(r) ** 2
        phi_t[i] = degrees(atan(r.imag / r.real))

    # Additional vectorization out of loop
    phi_t = phi_t - 45

    return rho_t, phi_t


@jit(nopython=True, parallel=True)
def direct_task_2d(
        periods: np.ndarray,
        layer_resistance: np.ndarray,
        layer_power: np.ndarray
):
    """
    Calculate two-dimensional direct task of MT.

    Args:
        periods (np.ndarray):
            Array of periods, e.g. [0.01, 0.005, ...]
        layer_resistance (np.ndarray):
            Electrical resistance by layer in Ohm * m, with shape (Width, LayerNum + 1)
        layer_power (np.ndarray):
            Power of layers in m, with shape (Width, LayerNum)

    Returns:
        rho (np.ndarray): Rho field over model with shape (Width, PeriodNum)
        phi (np.ndarray): Phi field over model with shape (Width, PeriodNum)
    """
    rho_t = np.empty((layer_resistance.shape[0], periods.shape[0]))
    phi_t = np.empty((layer_resistance.shape[0], periods.shape[0]))

    for i in prange(layer_resistance.shape[0]):
        rho, phi = direct_task_1d(periods, layer_resistance[i, :], layer_power[i, :])
        rho_t[i, :] = rho
        phi_t[i, :] = phi

    return rho_t, phi_t


@jit(nopython=True, parallel=True)
def direct_task_3d(
        periods: np.ndarray,
        layer_resistance: np.ndarray,
        layer_power: np.ndarray
):
    """
    Calculate two-dimensional direct task of MT.

    Args:
        periods (np.ndarray):
            Array of periods, e.g. [0.01, 0.005, ...]
        layer_resistance (np.ndarray):
            Electrical resistance by layer in Ohm * m, with shape (WidthX, WidthY, LayerNum + 1)
        layer_power (np.ndarray):
            Power of layers in m, with shape (WidthX, WidthY, LayerNum)

    Returns:
        rho (np.ndarray): Rho field over model with shape (WidthX, WidthY, PeriodNum)
        phi (np.ndarray): Phi field over model with shape (WidthX, WidthY, PeriodNum)
    """
    rho_t = np.empty((layer_resistance.shape[0], layer_resistance.shape[1], periods.shape[0]))
    phi_t = np.empty((layer_resistance.shape[0], layer_resistance.shape[1], periods.shape[0]))

    for i in prange(layer_resistance.shape[0]):
        rho, phi = direct_task_2d(periods, layer_resistance[i, :, :], layer_power[i, :, :])
        rho_t[i, :, :] = rho
        phi_t[i, :, :] = phi

    return rho_t, phi_t
