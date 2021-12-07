import abc

from pymt.microgrid import ResistivityMicrogrid

__all__ = ["ResistivityModel"]


class ResistivityModel(abc.ABC):
    @abc.abstractmethod
    def to_microgrid(self, *args, **kwargs) -> ResistivityMicrogrid:
        """
        Generate ResistivityMicrogrid from parameters.

        Returns:
            ResistivityMicrogrid - Generated ResistivityMicrogrid.

        """
        pass
