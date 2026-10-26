"""The SANS calculation of a structure: settings in, a Curve out.

``SansCalculator`` computes in this process. ``SansTask`` carries a
structure file and a calculator, so it can run in another process.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from coords2sans.curve import Curve
from coords2sans.structure import Structure

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence
    from pathlib import Path


class SansCalculator:
    """The settings of a SANS calculation, and the calculation itself.

    One size is computed as SasView's ``GenSAS`` does with
    ``is_avg = True``: every atom is a point with the amplitude
    (SLD − solvent SLD) · volume, at its distance r from the structure's
    centre, and F(q) = Σ amplitude · sin(q r) / (q r). With a
    ``polydispersity``, blown-up and shrunk copies of the structure are
    added up as sasmodels (SasView) does it: sizes from a Gaussian, each
    counting by its share, ``scale · Σ n·F² / Σ n·V · 1e8 +
    background``.

    Attributes
    ----------
    SIZE_RANGE : float, default=3.0
        How far the computed sizes reach on each side of the mean, in
        standard deviations: 99.7 % of the micelles.
    """

    SIZE_RANGE: float = 3.0

    def __init__(
        self,
        solvent_sld: float,
        *,
        deuterated: Sequence[str] = (),
        q_min: float = 0.002,
        q_max: float = 0.2,
        q_points: int = 256,
        q_log: bool = True,
        scale: float = 1.0,
        background: float = 0.0,
        polydispersity: float = 0.0,
        size_points: int = 35,
    ) -> None:
        """Keep the settings; ``calculate`` uses them.

        Parameters
        ----------
        solvent_sld : float
            The SLD of the solvent, in Å⁻²: -0.56e-6 for H2O, 6.36e-6
            for D2O.
        deuterated : Sequence[str], default=()
            The residue names whose hydrogens are deuterium.
        q_min : float, default=0.002
            The smallest q, in Å⁻¹.
        q_max : float, default=0.2
            The largest q, in Å⁻¹.
        q_points : int, default=256
            The number of q values.
        q_log : bool, default=True
            Space the q values evenly on a log scale, instead of a
            linear one.
        scale : float, default=1.0
            The volume fraction of the aggregates.
        background : float, default=0.0
            A flat background, in cm⁻¹.
        polydispersity : float, default=0.0
            The spread of the sizes: the relative standard deviation of
            the size factor. 0 computes one size.
        size_points : int, default=35
            How many sizes stand in for the spread. An odd number puts
            one exactly at the mean.
        """
        self.solvent_sld = solvent_sld
        self.deuterated = tuple(deuterated)
        self.q_min = q_min
        self.q_max = q_max
        self.q_points = q_points
        self.q_log = q_log
        self.scale = scale
        self.background = background
        self.polydispersity = polydispersity
        self.size_points = size_points

    @property
    def q(self) -> np.ndarray:
        """The q values the curve is computed at.

        Returns
        -------
        np.ndarray
            ``q_points`` values from ``q_min`` to ``q_max``, in Å⁻¹.
        """
        spacing = np.geomspace if self.q_log else np.linspace
        return spacing(self.q_min, self.q_max, self.q_points)

    def calculate(self, structure: Structure, label: str = '') -> Curve:
        """Compute the SANS curve of a structure.

        Parameters
        ----------
        structure : Structure
            The aggregate, in one piece, with its elements.
        label : str, default=''
            The curve's label.

        Returns
        -------
        Curve
            I(q) in cm⁻¹ at ``q``.
        """
        q = self.q
        volume = structure.volume
        amplitudes = (structure.sld(self.deuterated) - self.solvent_sld) * volume
        distances = np.linalg.norm(structure.positions - structure.centre, axis=1)
        factors, shares = self._sizes()
        squares = sum(
            share * self._form(q, distances * factor, amplitudes * factor**3) ** 2
            for factor, share in zip(factors, shares, strict=True)
        )
        # the shares' mean volume; 1e8 turns Å⁻¹ into cm⁻¹
        mean_volume = volume.sum() * (shares * factors**3).sum()
        intensity = self.scale * squares / mean_volume * 1e8 + self.background
        return Curve(q, intensity, label)

    def _sizes(self) -> tuple[np.ndarray, np.ndarray]:
        """The size factors to compute, and their shares.

        Without ``polydispersity``, one size: factor 1, share 1.
        Otherwise ``size_points`` factors evenly from
        1 − ``SIZE_RANGE`` σ to 1 + ``SIZE_RANGE`` σ, factors ≤ 0
        dropped, each with the Gaussian's height there as its share; the
        shares add up to 1. This is sasmodels' ``GaussianDispersion``.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The size factors, and their shares.

        Raises
        ------
        ValueError
            If ``polydispersity`` is negative, or ``size_points`` is
            below 1.
        """
        sigma = self.polydispersity
        if sigma < 0 or self.size_points < 1:
            err_msg = (
                'polydispersity must be 0 or more and size_points 1 or more, got '
                f'{sigma} and {self.size_points}'
            )
            raise ValueError(err_msg)
        if sigma == 0 or self.size_points == 1:
            return np.array([1.0]), np.array([1.0])
        reach = self.SIZE_RANGE * sigma
        factors = np.linspace(1 - reach, 1 + reach, self.size_points)
        factors = factors[factors > 0]
        shares = np.exp(-((factors - 1) ** 2) / (2 * sigma**2))
        return factors, shares / shares.sum()

    @staticmethod
    def _form(q: np.ndarray, distances: np.ndarray, amplitudes: np.ndarray) -> np.ndarray:
        """The spherically averaged amplitude F(q) of one size.

        Parameters
        ----------
        q : np.ndarray
            The scattering vector lengths, in Å⁻¹.
        distances : np.ndarray
            Each atom's distance from the centre, in Å.
        amplitudes : np.ndarray
            Each atom's (SLD − solvent SLD) · volume, in Å.

        Returns
        -------
        np.ndarray
            F(q), one value per q.
        """
        # np.sinc(x) is sin(pi x) / (pi x)
        return np.array([amplitudes @ np.sinc(q_value * distances / np.pi) for q_value in q])


@dataclass
class SansTask:
    """One curve from one structure file, as plain picklable data.

    ``run`` reads the file and computes its curve. The task holds a
    path and settings, not arrays, so sending it to another process is
    cheap: ``ProcessPoolExecutor.map(SansTask.run, tasks)``. It knows
    nothing about process pools; ``task.run()`` also works in this
    process.

    Attributes
    ----------
    structure_file : Path
        The ``.gro`` or ``.pdb`` file.
    calculator : SansCalculator
        The settings of the calculation.
    residues : Sequence[str] | None, default=None
        The residue names of the aggregate. None keeps every atom.
    masses : Mapping[str, Sequence[float]] | None, default=None
        By residue name, the masses in u of its atoms in topology
        order.
    elements : Mapping[str, Sequence[str]] | None, default=None
        By residue name, the element symbols of its atoms in order.
    label : str, default=''
        The curve's label.
    """

    structure_file: Path
    calculator: SansCalculator
    residues: Sequence[str] | None = None
    masses: Mapping[str, Sequence[float]] | None = None
    elements: Mapping[str, Sequence[str]] | None = None
    label: str = ''

    def run(self) -> Curve:
        """Read the structure file and compute its curve.

        Returns
        -------
        Curve
            The calculator's curve of the file's aggregate.
        """
        structure = Structure.read(
            self.structure_file,
            residues=self.residues,
            masses=self.masses,
            elements=self.elements,
        )
        return self.calculator.calculate(structure, self.label)
