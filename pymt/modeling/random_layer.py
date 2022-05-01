import typing as tp

import numpy as np

from pymt.microgrid import ResistivityMicrogrid
from pymt.modeling.abstract import ResistivityModel
from pymt.modeling.functional import generate_random_layers_2d

__all__ = ["RandomLayerModel"]


class RandomLayerModel(ResistivityModel):
    def __init__(
        self,
        alpha_range: tp.Tuple[float, float] = (0.01, 0.04),
        resistivity_range: tp.Tuple[float, float] = (1., 20000.),
        powers_range: tp.Tuple[float, float] = (20., 1000.),
        num_layers_range: tp.Tuple[int, int] = (3, 6)
    ):
        """
        Random "Layered" model generator class.

        Args:
            alpha_range:
                Tuple of min & max values for alpha param, e.g (0.1, 0.5)
                Lesser alpha - flatten layers.
            resistivity_range:
                Tuple of min & max values for resistivity, e.g (1., 20000.)
            powers_range:
                Tuple of min & max values for pixel powers, e.g (20., 1000.)
        """
        self._alpha_range = alpha_range
        self._resistivity_range = resistivity_range
        self._powers_range = powers_range
        self._num_layers_range = num_layers_range

        if any([i < 0 for i in self._resistivity_range]):
            raise ValueError(
                f"Resistivity must be in range [0, +inf), got: {self._resistivity_range}"
            )

        if any([i < 0 for i in self._resistivity_range]):
            raise ValueError(
                f"Powers must be in range (0, +inf), got: {self._powers_range}"
            )

    def to_microgrid(
        self,
        size: tp.Tuple[int, int],
    ) -> ResistivityMicrogrid:

        resistivity = generate_random_layers_2d(
            **size,
            num_layers=np.random.randint(*self._num_layers_range),
            resistivity_range=self._resistivity_range,
            alpha=np.random.uniform(*self._alpha_range),
        )

        layer_powers = np.random.randint(
            *self._powers_range, size=resistivity.shape
        )

        return ResistivityMicrogrid(
            resistivity, layer_power=layer_powers
        )
