import typing as tp

import numpy as np

from pymt.microgrid import ResistivityMicrogrid
from pymt.modeling.abstract import ResistivityModel
from pymt.modeling.functional import generate_random_layers_2d, generate_random_layers_3d

__all__ = ["RandomLayerModel"]


class RandomLayerModel(ResistivityModel):
    def __init__(
        self,
        layer_power_max: tp.List[float],
        layer_power_min: tp.List[float],
        layer_resistivity_max: tp.List[float],
        layer_resistivity_min: tp.List[float],
        layer_exist_probability: tp.Optional[tp.List[float]] = None,
    ):
        """
        Random "Layered" model generator class.

        Args:
            layer_power_max:
                List of max power for each layer in meters, e.g [100, 2000, 100, ...]
            layer_power_min:
                List of min power for each layer in meters, e.g [50, 1000, 50, ...]
            layer_exist_probability:
                List of layer existence probability in each point, e.g. [1.0, 1.0, 0.9, ...]
            layer_resistivity_min:
                List of min layer resistivity in Ohm * m, e.g. [2000, 1500, 8000, ...]
            layer_resistivity_max:
                List of max layer resistivity in Ohm * m, e.g. [3000, 6500, 12000, ...]
        """
        self.layer_power_max = np.array(layer_power_max)
        self.layer_power_min = np.array(layer_power_min)
        self.layer_resistivity_min = np.array(layer_resistivity_min)
        self.layer_resistivity_max = np.array(layer_resistivity_max)

        self.layer_exist_probability = layer_exist_probability
        if self.layer_exist_probability is None:
            self.layer_exist_probability = np.ones_like(layer_resistivity_max)

        if np.any(self.layer_resistivity_min < 0):
            raise ValueError(
                "All elements in layer_resistivity_min must be greater than 0, but "
                f"got min. element of layer_resistivity_min={self.layer_resistivity_min.min()}. "
                f"Indexes of all elements less than zero: "
                f"{np.where(self.layer_resistivity_min < 0)[0].tolist()}."
            )

        if np.any(self.layer_resistivity_max < 0):
            raise ValueError(
                "All elements in layer_resistivity_max must be greater or equal than 0, but "
                f"got min. element of layer_resistivity_max={self.layer_resistivity_max.min()}. "
                f"Indexes of all elements less than zero: "
                f"{np.where(self.layer_resistivity_max < 0)[0].tolist()}."
            )

        if np.any((self.layer_resistivity_max - self.layer_resistivity_min) < 0):
            raise ValueError(
                "All elements of layer_resistivity_max must be greater than elements in "
                "layer_resistivity_min. But that is false for elements with indexes: "
                f"{np.where((self.layer_resistivity_max - self.layer_resistivity_min) < 0)[0].tolist()}"
            )

        if np.any(self.layer_exist_probability < 0) or np.any(self.layer_exist_probability > 1):
            raise ValueError(f"All elements in layer_exist_probability must be in range [0, 1].")

    def to_microgrid(
        self,
        size: tp.Union[int, tp.Tuple[int, int]],
        grid_pixel_size: float,
        default_resistivity: tp.Optional[float] = None,
        random_layer_powers: tp.Optional[tp.Tuple[int, int]] = None,
    ) -> ResistivityMicrogrid:

        layer_resistivity = np.clip(
            np.random.random(self.layer_resistivity_max.shape) * self.layer_resistivity_max,
            self.layer_resistivity_min,
            np.inf,
        )

        if default_resistivity is None:
            default_resistivity = layer_resistivity[-1]

        if isinstance(size, int):
            resistivity = generate_random_layers_2d(
                size,
                grid_pixel_size,
                default_resistivity,
                self.layer_power_max,
                self.layer_power_min,
                self.layer_exist_probability,
                layer_resistivity,
            )
        elif isinstance(size, tuple):
            resistivity = generate_random_layers_3d(
                size[0],
                size[1],
                grid_pixel_size,
                default_resistivity,
                self.layer_power_max,
                self.layer_power_min,
                self.layer_exist_probability,
                layer_resistivity,
            )
        else:
            raise ValueError(f"size must be tuple or int, got {type(size)}.")

        layer_powers = None
        if random_layer_powers is not None:
            layer_powers = np.random.randint(
                *random_layer_powers, size=resistivity.shape
            )

        return ResistivityMicrogrid(
            resistivity, grid_pixel_size=grid_pixel_size, layer_power=layer_powers
        )
