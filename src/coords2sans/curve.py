"""A calculated SANS curve: q, I and a label."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes


@dataclass
class Curve:
    """A SANS curve: the intensity at each scattering vector length.

    Attributes
    ----------
    q : np.ndarray
        The scattering vector lengths, in Å⁻¹.
    intensity : np.ndarray
        I(q) in cm⁻¹, one value per q.
    label : str, default=''
        A name for the curve, shown in plots.
    """

    q: np.ndarray
    intensity: np.ndarray
    label: str = ''

    def save(self, path: Path) -> None:
        """Write the curve as a ``.dat`` that SasView reads as it is.

        Two columns, q in Å⁻¹ and I in cm⁻¹, separated by ``", "``,
        under a ``# q (1/A), I (1/cm)`` header. There is no third
        column: SasView's ASCII reader would take it for error bars.

        Parameters
        ----------
        path : Path
            The file to write.
        """
        np.savetxt(path, np.c_[self.q, self.intensity], delimiter=', ', header='q (1/A), I (1/cm)')

    @classmethod
    def load(cls, path: Path, label: str | None = None) -> Curve:
        """Read a curve that ``save`` wrote.

        Parameters
        ----------
        path : Path
            The ``.dat`` file.
        label : str | None, default=None
            The curve's label. None takes the file name without its
            suffix.

        Returns
        -------
        Curve
            The curve in the file.
        """
        q, intensity = np.loadtxt(path, delimiter=',', unpack=True)
        return cls(q, intensity, path.stem if label is None else label)

    def plot(self, ax: Axes | None = None) -> Axes:
        """Draw the curve as log-log I(q).

        matplotlib is an optional dependency: install the ``plot``
        extra.

        Parameters
        ----------
        ax : Axes | None, default=None
            The axes to draw on, to overlay curves. None draws on new
            ones.

        Returns
        -------
        Axes
            The axes drawn on.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        ax.loglog(self.q, self.intensity, label=self.label)
        ax.set_xlabel('q (Å⁻¹)')
        ax.set_ylabel('I(q) (cm⁻¹)')
        if self.label:
            ax.legend()
        return ax
