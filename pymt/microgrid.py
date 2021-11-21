import typing as tp
import numpy as np

from pymt.direct_task import direct_task_1d


class ResistivityMicrogrid:
    """
    1D Grid class for magnetotellurics data.

    Args:
        resistivity: Resistivity microgrid, in Ohm * m.
        pixel_size: Pixel (one point in microgrid) size in m.
        apparent_resistivity: Modulus of apparent resistivity, in Ohm * m.
        impedance_phase: Phase of impedance, in degrees.

    """

    def __init__(
        self,
        resistivity: np.ndarray,
        pixel_size: float,
        num_freq: tp.Optional[int] = None,
        apparent_resistivity: tp.Optional[np.ndarray] = None,
        impedance_phase: tp.Optional[np.ndarray] = None,
    ):
        self.resistivity = resistivity
        self.layer_power = np.full_like(resistivity, pixel_size)
        self.pixel_size = pixel_size

        self._num_freq = num_freq
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
        if self.impedance_phase is None:
            raise AttributeError(
                f"Impedance phase does not exist. Compute it first with the compute_direct_task method."
            )
        return self._impedance_phase

    @property
    def num_freq(self) -> int:
        """
        Number of frequencies in impedance_phase and apparent_resistivity.
        """
        if self.num_freq is None:
            raise AttributeError(
                f"Num freq does not exist. Set it first with the compute_direct_task method."
            )
        return self._num_freq

    @apparent_resistivity.setter
    def apparent_resistivity(self, value: np.ndarray):
        self._apparent_resistivity = value

    @impedance_phase.setter
    def impedance_phase(self, value: np.ndarray):
        self._impedance_phase = value

    @num_freq.setter
    def num_freq(self, value: int):
        self._num_freq = value

    def compute_direct_task(
        self,
        num_freq: int,
        first_period: float = 0.01,
        geometric_step: float = 2.0,
    ):
        rho, phi = direct_task_1d(
            num_freq, first_period, geometric_step, self.resistivity, self.layer_power
        )

        self.num_freq = num_freq
        self.apparent_resistivity = rho
        self.impedance_phase = phi

        return None
