import typing as tp

import numpy as np

import pymt.direct_task as direct_tasks


class ResistivityMicrogrid:
    """
    N-Dimensional Grid class for MT data.

    Args:
        resistivity: Resistivity microgrid, in Ohm * m.
            Last dimension is z (/depth), used for direct task computation.
            Example for depth = 100 microgrid points:
                1D - np.random.randint(1, 1000, (100))
                2D - np.random.randint(1, 1000, (32, 100))
                3D - np.random.randint(1, 1000, (32, 72, 100))

        grid_pixel_size: Pixel (one point in microgrid) size in m.
        apparent_resistivity: Modulus of apparent resistivity, in Ohm * m.
        impedance_phase: Phase of impedance, in degrees.

    """

    __slots__ = (
        "resistivity",
        "layer_power",
        "grid_element_size",
        "_periods",
        "_apparent_resistivity",
        "_impedance_phase",
    )

    def __init__(
        self,
        resistivity: np.ndarray,
        grid_pixel_size: float,
        periods: tp.Optional[np.ndarray] = None,
        apparent_resistivity: tp.Optional[np.ndarray] = None,
        impedance_phase: tp.Optional[np.ndarray] = None,
    ):
        self.resistivity = resistivity
        self.layer_power = np.full_like(resistivity, grid_pixel_size)
        self.grid_element_size = grid_pixel_size

        self._periods = periods
        self._apparent_resistivity = apparent_resistivity
        self._impedance_phase = impedance_phase

    @property
    def apparent_resistivity(self) -> np.ndarray:
        """
        Apparent resistivity in Ohm * m.
        """
        if self._apparent_resistivity is None:
            raise AttributeError(
                f"Apparent resistivity does not exist. Compute it first with the compute_direct_task method."
            )
        return self._apparent_resistivity

    @property
    def impedance_phase(self) -> np.ndarray:
        """
        Impedance phase in degrees.
        """
        if self._impedance_phase is None:
            raise AttributeError(
                f"Impedance phase does not exist. Compute it first with the compute_direct_task method."
            )
        return self._impedance_phase

    @property
    def periods(self) -> np.ndarray:
        """
        Periods (frequencies), used to compute impedance_phase and apparent_resistivity.
        """
        if self._periods is None:
            raise AttributeError(
                f"Periods does not exist. Set it first with the compute_direct_task method."
            )
        return self._periods

    @apparent_resistivity.setter
    def apparent_resistivity(self, value: np.ndarray):
        self._apparent_resistivity = value

    @impedance_phase.setter
    def impedance_phase(self, value: np.ndarray):
        self._impedance_phase = value

    @periods.setter
    def periods(self, value: np.ndarray):
        self._periods = value

    def compute_direct_task(self, periods: np.ndarray):
        number_of_dims = len(self.resistivity.shape)
        if number_of_dims > 3:
            raise ValueError(
                "compute_direct_task is implemented for only 1D, 2D and 3D resistivity microgrids.",
                f"Got {number_of_dims}D array with shape {self.resistivity.shape}",
            )

        direct_task_fn = getattr(direct_tasks, f"direct_task_{number_of_dims}d")

        rho, phi = direct_task_fn(periods, self.resistivity, self.layer_power)

        self.periods = periods
        self.apparent_resistivity = rho
        self.impedance_phase = phi

        return None
