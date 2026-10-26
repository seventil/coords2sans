"""The atoms of an aggregate, read from a ``.gro`` or ``.pdb`` file."""

from __future__ import annotations

import string
import warnings
from typing import TYPE_CHECKING

import numpy as np
import periodictable
from periodictable import formula
from periodictable import nsf

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence
    from pathlib import Path

# H to U (Z = 1 to 92) and deuterium. Any two are 0.0196 u apart or
# more; heavier elements have whole-number masses that tie, such as Cm
# and Bk at 247 u.
_TABLE = [*[e for e in periodictable.elements if 0 < e.number <= 92], periodictable.D]
_SYMBOLS = np.array([element.symbol for element in _TABLE])
_MASSES = np.array([element.mass for element in _TABLE])


class Structure:
    """The atoms of an aggregate, prepared for a scattering calculation.

    Make it with ``read``. It holds, per atom, the attributes
    ``resids``, ``resnames``, ``names``, ``positions`` (Å) and
    ``elements``, and per residue name ``element_sources``:
    ``"masses"``, ``"elements"``, ``"pdb"`` or ``"names"`` (guessed).
    After ``read`` it is only read from, except ``elements``, which may
    be corrected.

    Attributes
    ----------
    ELEMENT_SYMBOLS : np.ndarray, default=_SYMBOLS
        The elements a topology mass is matched against: H to U, and D.
    ELEMENT_MASSES : np.ndarray, default=_MASSES
        Their masses in u, in the same order.
    MASS_TOLERANCE : float, default=0.01
        The largest gap in u between a topology mass and its element.
        Force fields round masses by less than 0.005 u; a united-atom
        CH2 (14.027 u) lies 0.02 u from N and must not pass.
    AVOGADRO : float, default=6.02214129e23
        SasView's value, kept so that atom volumes match SasView's.
    """

    ELEMENT_SYMBOLS: np.ndarray = _SYMBOLS
    ELEMENT_MASSES: np.ndarray = _MASSES
    MASS_TOLERANCE: float = 0.01
    AVOGADRO: float = 6.02214129e23

    def __init__(
        self,
        resids: np.ndarray,
        resnames: np.ndarray,
        names: np.ndarray,
        positions: np.ndarray,
        elements: Sequence[str],
        element_sources: Mapping[str, str] | None = None,
    ) -> None:
        """Hold atoms that are already prepared.

        ``read`` prepares a file; this takes atoms from any source, for
        example another library, as they are.

        Parameters
        ----------
        resids : np.ndarray
            The residue number of each atom.
        resnames : np.ndarray
            The residue name of each atom.
        names : np.ndarray
            The name of each atom.
        positions : np.ndarray
            The atom positions in Å, with shape (atoms, 3), the
            aggregate in one piece.
        elements : Sequence[str]
            The element symbol of each atom.
        element_sources : Mapping[str, str] | None, default=None
            Where each residue name's elements came from. None records
            ``"elements"`` for every residue.
        """
        self.resids = np.asarray(resids)
        self.resnames = np.asarray(resnames)
        self.names = np.asarray(names)
        self.positions = np.asarray(positions, dtype=float)
        self.elements = elements
        self.element_sources = dict(
            element_sources or dict.fromkeys(np.unique(self.resnames).tolist(), 'elements')
        )

    @property
    def elements(self) -> np.ndarray:
        """The element symbol of each atom.

        Assign a new array to correct it; the assignment is checked.
        ``sld`` and ``volume`` are computed from it on every read.

        Returns
        -------
        np.ndarray
            One symbol per atom, such as ``"C"`` or ``"D"``.
        """
        return self._elements

    @elements.setter
    def elements(self, value: Sequence[str]) -> None:
        """Set the element symbol of each atom.

        Parameters
        ----------
        value : Sequence[str]
            One symbol per atom.

        Raises
        ------
        ValueError
            If there is not one symbol per atom, or a symbol is not in
            ``ELEMENT_SYMBOLS``.
        """
        symbols = np.asarray(value, dtype=str)
        if symbols.shape != self.resids.shape:
            err_msg = f'{len(symbols)} elements given for {len(self.resids)} atoms'
            raise ValueError(err_msg)
        unknown = sorted(set(symbols.tolist()) - set(self.ELEMENT_SYMBOLS.tolist()))
        if unknown:
            err_msg = f'Unknown elements: {", ".join(unknown)}'
            raise ValueError(err_msg)
        self._elements = symbols

    @property
    def centre(self) -> np.ndarray:
        """The middle of the atoms' bounding box, as SasView takes it.

        Returns
        -------
        np.ndarray
            The centre in Å, shape (3,).
        """
        return (self.positions.min(axis=0) + self.positions.max(axis=0)) / 2

    @property
    def volume(self) -> np.ndarray:
        """The volume of each atom: its mass at its element's density.

        Computed as SasView's ``PDBReader`` does. D has the volume of H.

        Returns
        -------
        np.ndarray
            One volume per atom, in Å³.
        """
        symbols, index = np.unique(self.elements, return_inverse=True)
        atoms = [formula(symbol) for symbol in symbols]
        volumes = np.array([1e24 * atom.mass / atom.density / self.AVOGADRO for atom in atoms])
        return volumes[index]

    def sld(self, deuterated: Sequence[str] = ()) -> np.ndarray:
        """The neutron scattering length density (SLD) of each atom.

        The coherent SLD of each element at its bulk density, as
        SasView's ``PDBReader`` gives it. The hydrogens of the
        ``deuterated`` residues count as D.

        Parameters
        ----------
        deuterated : Sequence[str], default=()
            Residue names whose hydrogens are deuterium.

        Returns
        -------
        np.ndarray
            One SLD per atom, in Å⁻².

        Raises
        ------
        ValueError
            If a deuterated residue name is not in the structure.
        """
        unknown = sorted(set(deuterated) - set(self.resnames.tolist()))
        if unknown:
            err_msg = f'Deuterated residues not in the structure: {", ".join(unknown)}'
            raise ValueError(err_msg)
        elements = self.elements.copy()
        elements[(elements == 'H') & np.isin(self.resnames, list(deuterated))] = 'D'
        symbols, index = np.unique(elements, return_inverse=True)
        slds = np.array([nsf.neutron_sld(symbol)[0] for symbol in symbols]) * 1e-6
        return slds[index]

    @classmethod
    def read(
        cls,
        path: Path,
        residues: Sequence[str] | None = None,
        masses: Mapping[str, Sequence[float]] | None = None,
        elements: Mapping[str, Sequence[str]] | None = None,
    ) -> Structure:
        """Read a ``.gro`` or ``.pdb`` file and prepare its aggregate.

        Only the atoms of ``residues`` are kept: water and ions are the
        solvent. A ``.gro`` has its periodic wrapping fixed. Then each
        atom gets its element, per residue name from the first source
        that has it: ``masses``, ``elements``, the PDB element column.
        Residues none of them covers are guessed from their atom names,
        with a warning.

        Parameters
        ----------
        path : Path
            The structure file; its suffix picks the reader.
        residues : Sequence[str] | None, default=None
            The residue names of the aggregate. None keeps every atom.
        masses : Mapping[str, Sequence[float]] | None, default=None
            By residue name, the masses in u of its atoms in topology
            order, as a topology file lists them.
        elements : Mapping[str, Sequence[str]] | None, default=None
            By residue name, the element symbols of its atoms in order.

        Returns
        -------
        Structure
            The aggregate, in one piece, with its elements.

        Raises
        ------
        ValueError
            If the file is not a ``.gro`` or ``.pdb``, or holds no atom
            of ``residues``.
        """
        readers = {'.gro': cls._from_gro, '.pdb': cls._from_pdb}
        reader = readers.get(path.suffix.lower())
        if reader is None:
            err_msg = f"Can't read {path.name}: give a .gro or a .pdb file"
            raise ValueError(err_msg)
        resids, resnames, names, positions, box, file_elements = reader(path)
        kept = np.ones(len(resids), bool) if residues is None else np.isin(resnames, residues)
        if not kept.any():
            err_msg = f'{path.name} holds no residue named {", ".join(residues or ())}'
            raise ValueError(err_msg)
        resids, resnames, names, positions = (
            array[kept] for array in (resids, resnames, names, positions)
        )
        if box is not None:
            positions = cls._fix_periodic_wrapping(resids, positions, box)
        if file_elements is not None:
            file_elements = file_elements[kept]
        symbols, sources = cls._assign_elements(
            resids, resnames, names, masses or {}, elements or {}, file_elements
        )
        return cls(resids, resnames, names, positions, symbols, sources)

    @staticmethod
    def _from_gro(path: Path) -> tuple:
        """Read the atoms and the box of a GROMACS ``.gro`` file.

        Fields are cut at their fixed columns, since names may touch.
        As GROMACS defines it, the width of a position field is the
        distance between the decimal points of the first atom's x and
        y.

        Parameters
        ----------
        path : Path
            The ``.gro`` file.

        Returns
        -------
        tuple
            Residue numbers, residue names, atom names, positions in Å,
            the box lengths in Å, and None (no element column).

        Raises
        ------
        ValueError
            If the box is not rectangular.
        """
        lines = path.read_text(encoding='utf-8').splitlines()
        atoms = lines[2 : 2 + int(lines[1])]
        dot = atoms[0].index('.', 20)
        width = atoms[0].index('.', dot + 1) - dot
        columns = [slice(20 + k * width, 20 + (k + 1) * width) for k in range(3)]
        resids = np.array([int(line[0:5]) for line in atoms])
        resnames = np.array([line[5:10].strip() for line in atoms])
        names = np.array([line[10:15].strip() for line in atoms])
        nm = np.array([[line[column] for column in columns] for line in atoms], dtype=float)
        box = np.array(lines[2 + len(atoms)].split(), dtype=float)
        if np.any(box[3:]):
            err_msg = f'{path.name} has a triclinic box; only rectangular ones are supported'
            raise ValueError(err_msg)
        return resids, resnames, names, nm * 10, box[:3] * 10, None

    @staticmethod
    def _from_pdb(path: Path) -> tuple:
        """Read the atoms of a PDB file, with their element column.

        The ``ATOM`` and ``HETATM`` records of the first model are cut
        at their fixed columns. The residue name takes columns 18 to 21,
        where GROMACS, CHARMM and shapespyer write four-letter names. A
        PDB gives no box: it is taken as it is, as SasView takes it,
        since many programs write a placeholder ``CRYST1`` of 1 Å.

        Parameters
        ----------
        path : Path
            The PDB file.

        Returns
        -------
        tuple
            Residue numbers, residue names, atom names, positions in Å,
            None (no box), and the element column (columns 77 to 78),
            with ``""`` where it is empty.
        """
        atoms = []
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.startswith('ENDMDL'):
                break
            if line.startswith(('ATOM  ', 'HETATM')):
                atoms.append(line)
        resids = np.array([int(line[22:26]) for line in atoms])
        resnames = np.array([line[17:21].strip() for line in atoms])
        names = np.array([line[12:16].strip() for line in atoms])
        positions = np.array([(line[30:38], line[38:46], line[46:54]) for line in atoms], float)
        file_elements = np.array([line[76:78].strip().capitalize() for line in atoms])
        return resids, resnames, names, positions, None, file_elements

    @staticmethod
    def _fix_periodic_wrapping(
        resids: np.ndarray, positions: np.ndarray, box: np.ndarray
    ) -> np.ndarray:
        """Put an aggregate back together across its periodic box faces.

        GROMACS writes every position back inside the box, which can
        split molecules and the aggregate. A molecule is a run of atoms
        with the same residue number. Each one is made whole first: its
        atoms move to their images nearest its first atom. The
        aggregate's centre is the circular mean of the molecule centres
        along each axis, which a split does not move. Then each molecule
        moves by whole box lengths to its image nearest that centre.

        Parameters
        ----------
        resids : np.ndarray
            The residue number of each atom.
        positions : np.ndarray
            The atom positions, with shape (atoms, 3).
        box : np.ndarray
            The box lengths, in the unit of ``positions``.

        Returns
        -------
        np.ndarray
            The positions, with the aggregate in one piece.
        """
        starts = np.flatnonzero(np.r_[True, resids[1:] != resids[:-1]])
        sizes = np.diff(np.r_[starts, len(positions)])
        first = np.repeat(positions[starts], sizes, axis=0)
        offset = positions - first
        positions = first + offset - box * np.round(offset / box)
        centres = np.add.reduceat(positions, starts) / sizes[:, None]
        angle = 2 * np.pi * centres / box
        mean = np.arctan2(np.sin(angle).mean(0), np.cos(angle).mean(0))
        # folded into the box, so an aggregate already whole stays put
        centre = (box / (2 * np.pi) * mean) % box
        shift = -box * np.round((centres - centre) / box)
        return positions + np.repeat(shift, sizes, axis=0)

    @classmethod
    def _assign_elements(
        cls,
        resids: np.ndarray,
        resnames: np.ndarray,
        names: np.ndarray,
        masses: Mapping[str, Sequence[float]],
        elements: Mapping[str, Sequence[str]],
        file_elements: np.ndarray | None,
    ) -> tuple[np.ndarray, dict[str, str]]:
        """Find each atom's element, per residue name.

        A residue is a run of atoms with the same residue number. Masses
        and given elements are matched by the atom's position in its
        residue, so the atom names of the structure and the topology may
        differ. A mass gives the element of nearest mass in
        ``ELEMENT_MASSES``. A residue no source covers is guessed from
        its atom names, leading digits left out: a name that is a
        two-letter element (``CL``, ``NA``) is that element, any other
        its first letter (``C21`` is C, ``HG1`` is H). An atom named
        exactly ``CA`` is then calcium, hence the warning.

        Parameters
        ----------
        resids : np.ndarray
            The residue number of each atom.
        resnames : np.ndarray
            The residue name of each atom.
        names : np.ndarray
            The name of each atom.
        masses : Mapping[str, Sequence[float]]
            By residue name, its atom masses in u, in order.
        elements : Mapping[str, Sequence[str]]
            By residue name, its element symbols, in order.
        file_elements : np.ndarray | None
            The PDB element column, ``""`` where empty; None for a
            ``.gro``.

        Returns
        -------
        tuple[np.ndarray, dict[str, str]]
            The element of each atom, and each residue name's source.

        Raises
        ------
        ValueError
            If a residue has another number of atoms than its masses or
            elements, a mass lies more than ``MASS_TOLERANCE`` from
            every element (hydrogen mass repartitioning, a united-atom
            group), or a guessed name gives no element.
        """
        starts = np.flatnonzero(np.r_[True, resids[1:] != resids[:-1]])
        sizes = np.diff(np.r_[starts, len(resids)])
        position = np.arange(len(resids)) - np.repeat(starts, sizes)
        size = np.repeat(sizes, sizes)
        known = set(cls.ELEMENT_SYMBOLS.tolist())
        result = np.empty(len(resids), dtype=cls.ELEMENT_SYMBOLS.dtype)
        sources = {}
        for resname in np.unique(resnames).tolist():
            atoms = resnames == resname
            given = masses.get(resname, elements.get(resname))
            if given is not None and np.any(size[atoms] != len(given)):
                err_msg = f'A {resname} residue has not the {len(given)} atoms given for it'
                raise ValueError(err_msg)
            if resname in masses:
                atom_masses = np.asarray(masses[resname], dtype=float)[position[atoms]]
                nearest = np.abs(atom_masses[:, None] - cls.ELEMENT_MASSES).argmin(axis=1)
                far = np.abs(cls.ELEMENT_MASSES[nearest] - atom_masses) > cls.MASS_TOLERANCE
                if far.any():
                    err_msg = f"The mass {atom_masses[far][0]} u in {resname} is no element's"
                    raise ValueError(err_msg)
                result[atoms], sources[resname] = cls.ELEMENT_SYMBOLS[nearest], 'masses'
            elif resname in elements:
                result[atoms] = np.asarray(elements[resname])[position[atoms]]
                sources[resname] = 'elements'
            elif file_elements is not None and np.all(file_elements[atoms] != ''):
                result[atoms], sources[resname] = file_elements[atoms], 'pdb'
            else:
                stems = [name.lstrip(string.digits) for name in names[atoms]]
                guesses = [
                    stem.capitalize()
                    if len(stem) == 2 and stem.capitalize() in known
                    else stem[:1].upper()
                    for stem in stems
                ]
                if not known.issuperset(guesses):
                    err_msg = f'No element for some atom names of {resname}; give its elements'
                    raise ValueError(err_msg)
                result[atoms], sources[resname] = guesses, 'names'
        guessed = [resname for resname, source in sources.items() if source == 'names']
        if guessed:
            warnings.warn(
                f'Elements of {", ".join(guessed)} guessed from atom names; check '
                'structure.elements',
                stacklevel=3,
            )
        return result, sources
