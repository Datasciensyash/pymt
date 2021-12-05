import numpy as np
from numba import jit, prange


@jit(nopython=True)
def generate_random_layers_2d(
    size: int,
    step_z: float,
    default_resistivity: float,
    layer_power_max: np.ndarray,
    layer_power_min: np.ndarray,
    layer_exist_probability: np.ndarray,
    layer_resistivity: np.ndarray,
) -> np.ndarray:
    num_layers = len(layer_resistivity)
    resistivity_grid = np.full(
        (size, np.sum(layer_power_max) // step_z), default_resistivity
    )
    for x in prange(size):
        offset = 0
        for i in range(num_layers):
            power = 0
            if np.random.random() <= layer_exist_probability[i]:
                power = layer_power_min[i] + (
                    np.random.random() * (layer_power_max[i] - layer_power_min[i])
                )
            num_blocks = int(power // step_z)
            resistivity_grid[x, offset : offset + num_blocks] = layer_resistivity[i]
            offset += num_blocks
    return resistivity_grid


@jit(nopython=True)
def generate_random_layers_3d(
    size_x: int,
    size_y: int,
    step_z: float,
    default_resistivity: float,
    layer_power_max: np.ndarray,
    layer_power_min: np.ndarray,
    layer_exist_probability: np.ndarray,
    layer_resistivity: np.ndarray,
) -> np.ndarray:
    num_layers = len(layer_resistivity)
    resistivity_grid = np.full(
        (size_x, size_y, np.sum(layer_power_max) // step_z), default_resistivity
    )
    for x in prange(size_x):
        for y in prange(size_y):
            offset = 0
            for i in range(num_layers):
                power = 0
                if np.random.random() <= layer_exist_probability[i]:
                    power = layer_power_min[i] + (
                        np.random.random() * (layer_power_max[i] - layer_power_min[i])
                    )
                num_blocks = int(power // step_z)
                resistivity_grid[
                    x, y, offset : offset + num_blocks
                ] = layer_resistivity[i]
                offset += num_blocks
    return resistivity_grid
