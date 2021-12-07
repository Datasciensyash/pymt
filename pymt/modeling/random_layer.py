import typing as tp

import numpy as np

from pymt.microgrid import ResistivityMicrogrid
from pymt.modeling.abstract import ResistivityModel
from pymt.modeling.functional import (generate_random_layers_2d,
                                      generate_random_layers_3d)

__all__ = ["RandomLayerModel"]


class RandomLayerModel(ResistivityModel):
    def __init__(
        self,
        layer_power_max: tp.List[float],
        layer_power_min: tp.List[float],
        layer_exist_probability: tp.List[float],
        layer_resistivity: tp.List[float],
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
            layer_resistivity:
                List of layer resistivity in Ohm * m, e.g. [2000, 1500, 8000, ...]

        """
        self.layer_power_max = layer_power_max
        self.layer_power_min = layer_power_min
        self.layer_exist_probability = layer_exist_probability
        self.layer_resistivity = layer_resistivity

    def to_microgrid(
        self,
        size: tp.Union[int, tp.Tuple[int], tp.Tuple[int, int]],
        grid_pixel_size: float,
        default_resistivity: float = 0.0,
    ) -> ResistivityMicrogrid:
        if isinstance(size, int):
            resistivity = generate_random_layers_2d(
                size,
                grid_pixel_size,
                default_resistivity,
                np.array(self.layer_power_max),
                np.array(self.layer_power_min),
                np.array(self.layer_exist_probability),
                np.array(self.layer_resistivity),
            )
        elif isinstance(size, tuple):
            resistivity = generate_random_layers_3d(
                size[0],
                size[1],
                grid_pixel_size,
                default_resistivity,
                np.array(self.layer_power_max),
                np.array(self.layer_power_min),
                np.array(self.layer_exist_probability),
                np.array(self.layer_resistivity),
            )
        else:
            raise ValueError(f"size must be tuple or int, got {type(size)}.")

        return ResistivityMicrogrid(resistivity, grid_pixel_size=grid_pixel_size)
