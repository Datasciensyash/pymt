import typing as tp
import numpy as np
import abc

from pymt.microgrid import ResistivityMicrogrid


class ResistivityModel(abc.ABC):
    @abc.abstractmethod
    def to_microgrid(self, pixel_size: float) -> ResistivityMicrogrid:
        pass


class ResistivityMacroGrid(ResistivityModel):
    def __init__(
        self,
        block_height: tp.List[float],
        resistivity: tp.List[float],
    ):
        """
        Arbitrary MacroGrid class for magnetotellurics data.

        Args:
            block_height: Size of each block in m.
            resistivity: Resistivity microgrid, in Ohm * m.
        """
        self.block_height = block_height
        self.resistivity = resistivity

    def to_microgrid(self, pixel_size: float) -> ResistivityMicrogrid:
        resistivity_data = []
        for resistivity, block_height in zip(self.resistivity, self.block_height):
            num_pixels = int(block_height // pixel_size)
            resistivity_data.extend([resistivity] * num_pixels)
        resistivity_data = np.array(resistivity_data)
        return ResistivityMicrogrid(resistivity_data, pixel_size)


class ResistivityLayerModel(ResistivityModel):
    def __init__(
        self,
        layer_power: tp.List[float],
        resistivity: tp.List[float],
    ):
        """
        Arbitrary MacroGrid class for magnetotellurics data.

        Args:
            layer_power: Power of each layer in m.
            resistivity: Resistivity microgrid, in Ohm * m.
        """
        self.layer_power = layer_power
        self.resistivity = resistivity

    def to_microgrid(self, pixel_size: float) -> ResistivityMicrogrid:
        resistivity_data = []
        for resistivity, block_height in zip(self.resistivity, self.layer_power):
            num_pixels = int(block_height // pixel_size)
            resistivity_data.extend([resistivity] * num_pixels)
        resistivity_data = np.array(resistivity_data)
        return ResistivityMicrogrid(resistivity_data, pixel_size)


if __name__ == "__main__":
    rmg = ResistivityMacroGrid([2, 4, 8, 16], [10, 300, 500, 2500])
    g = rmg.to_microgrid(pixel_size=2)
    g.compute_direct_task(np.array([0.01 * 2 ** i for i in range(13)]))
