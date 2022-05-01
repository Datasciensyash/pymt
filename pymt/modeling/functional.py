import numpy as np
from numba import jit, prange
import typing as tp

__all__ = ["generate_random_layers_2d"]


@jit(nopython=True)
def clip(value: np.ndarray, min: float, max: float) -> np.ndarray:
    value = np.where(value < min, min, value)
    value = np.where(value > max, max, value)
    return value


@jit(nopython=True)
def generate_random_layers_2d(
        width: int,
        depth: int,
        num_layers: int = 4,
        resistivity_range: tp.Tuple[float, float] = (1, 20000),
        alpha: float = 0.01,
) -> np.ndarray:
    num_layers = num_layers + 1
    resistivity = np.zeros((width, depth))
    layer_depths = np.random.random((num_layers,))
    layer_values = np.random.randint(*resistivity_range, (num_layers,))
    for i in range(width):
        layer_depths = layer_depths + (np.random.random(layer_depths.shape) - 0.5) * alpha
        layer_depths = layer_depths / np.sum(layer_depths)
        layer_depths = clip(layer_depths, 0., 1.)
        depths = (np.cumsum(layer_depths) * depth).astype(np.int32)
        depths[0], depths[-1] = 0, depth
        for j in range(1, num_layers):
            resistivity[i, depths[j - 1]:depths[j]] = layer_values[j - 1]
    return resistivity
