import numpy as np
import pytest

from pymt.direct_task import direct_task_1d


def test_direct_task():
    rho, phi = direct_task_1d(
        num_freq=27,
        first_period=0.01,
        geometric_step=2,
        layer_resistivity=np.array([1000, 1]),
        layer_power=np.array([5000]),
    )

    phi_real = np.array(
        [
            -45.0,
            -43.78,
            -45.0,
            -54.3,
            -67.08,
            -76.49,
            -81.54,
            -83.67,
            -84.04,
            -83.26,
            -81.62,
            -79.22,
            -76.15,
            -72.53,
            -68.55,
            -64.48,
            -60.6,
            -57.14,
            -54.22,
            -51.87,
            -50.05,
            -48.67,
            -47.65,
            -46.9,
            -46.36,
            -45.97,
            -45.69,
        ]
    )

    rho_real = np.array(
        [
            993.01,
            1011.83,
            1176.27,
            1278.06,
            1000.0,
            599.27,
            323.14,
            169.9,
            89.61,
            48.01,
            26.36,
            14.98,
            8.9,
            5.59,
            3.75,
            2.7,
            2.08,
            1.7,
            1.46,
            1.31,
            1.21,
            1.15,
            1.1,
            1.07,
            1.05,
            1.03,
            1.02,
        ]
    )

    rho, phi = np.round(rho, 2), np.round(phi, 2)

    assert np.allclose(phi, phi_real)
    assert np.allclose(rho, rho_real)
