import numpy as np
import pytest

from coords2sans.structure import Structure

# Three atoms of a made-up residue SUR (C, H, H), named so that the
# names alone would make calcium of the carbon.
SUR_MASSES = {'SUR': [12.011, 1.008, 1.008]}


def write_gro(path, rows, box='   3.00000   3.00000   3.00000'):
    """Write a .gro of (resid, resname, name, x, y, z) rows, in nm."""
    lines = [
        f'{r:5d}{rn:<5s}{n:>5s}{k:5d}{x:8.3f}{y:8.3f}{z:8.3f}'
        for k, (r, rn, n, x, y, z) in enumerate(rows, 1)
    ]
    path.write_text('\n'.join(['title', f'{len(lines):5d}', *lines, box, '']))
    return path


SUR_ROWS = [
    (1, 'SUR', 'CA', 1.000, 1.000, 1.000),
    (1, 'SUR', 'HA1', 1.109, 1.000, 1.000),
    (1, 'SUR', 'HA2', 1.000, 1.109, 1.000),
    (2, 'TIP3', 'OH2', 0.300, 0.300, 0.300),
    (2, 'TIP3', 'H1', 0.396, 0.300, 0.300),
    (2, 'TIP3', 'H2', 0.300, 0.396, 0.300),
]


class TestReadGro:
    def test_cuts_fixed_columns(self, tmp_path):
        # The residue number touches the residue name and the atom name
        # touches the atom number; the second atom carries velocities.
        # Splitting on spaces would fail.
        path = tmp_path / 'touching.gro'
        path.write_text(
            'title\n'
            '    2\n'
            '10000POPC C311411037   1.144   1.144   1.185\n'
            '  179BRA     BR11038   2.085   2.381   2.363 -0.1062  0.2399 -0.3583\n'
            '   8.21135   8.21135  11.67869\n'
        )

        structure = Structure.read(path, elements={'POPC': ['C'], 'BRA': ['Br']})

        assert structure.resids.tolist() == [10000, 179]
        assert structure.resnames.tolist() == ['POPC', 'BRA']
        assert structure.names.tolist() == ['C3114', 'BR']
        np.testing.assert_allclose(
            structure.positions, [[11.44, 11.44, 11.85], [20.85, 23.81, 23.63]]
        )

    def test_field_width_follows_the_decimals(self, tmp_path):
        # Written with 4 decimals (gmx -ndec 4): each position field is
        # 9 wide.
        path = tmp_path / 'four_decimals.gro'
        path.write_text(
            'title\n'
            '    1\n'
            '    1CTA     N1    1   4.1441   4.1442   1.1853\n'
            '   8.21135   8.21135  11.67869\n'
        )

        structure = Structure.read(path, elements={'CTA': ['N']})

        np.testing.assert_allclose(structure.positions, [[41.441, 41.442, 11.853]])

    def test_rejects_a_triclinic_box(self, tmp_path):
        path = write_gro(
            tmp_path / 'triclinic.gro',
            SUR_ROWS[:3],
            box='   8.00000   8.00000   8.00000   0.00000   0.00000'
            '   4.00000   0.00000   0.00000   0.00000',
        )

        with pytest.raises(ValueError):
            Structure.read(path, masses=SUR_MASSES)


class TestReadPdb:
    def test_cuts_fixed_columns(self, tmp_path):
        # A four-letter residue name in columns 18-21, a four-letter
        # atom name, an ion as HETATM, and a second model that must not
        # be read. Positions stay in Å.
        path = tmp_path / 'bilayer.pdb'
        path.write_text(
            'CRYST1   50.000   50.000   70.395  90.00  90.00  90.00 P 1           1\n'
            'MODEL        1\n'
            'ATOM      1 H12A POPC    1     -24.464 -25.668  28.809  1.00  0.00      M\n'
            'HETATM    2  SOD SOD   201      10.000  20.000  30.000  1.00  0.00\n'
            'ENDMDL\n'
            'MODEL        2\n'
            'ATOM      1 H12A POPC    1       0.000   0.000   0.000  1.00  0.00      M\n'
            'ENDMDL\n'
        )

        structure = Structure.read(path, elements={'POPC': ['H'], 'SOD': ['Na']})

        assert structure.resids.tolist() == [1, 201]
        assert structure.resnames.tolist() == ['POPC', 'SOD']
        assert structure.names.tolist() == ['H12A', 'SOD']
        np.testing.assert_allclose(structure.positions, [[-24.464, -25.668, 28.809], [10, 20, 30]])

    def test_placeholder_box_is_not_used(self, tmp_path):
        # Many programs write a CRYST1 of 1 Å; a PDB gets no box fix.
        path = tmp_path / 'sur.pdb'
        path.write_text(
            'CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1\n'
            'ATOM      1   CA SUR     1      10.000  10.000  10.000\n'
            'ATOM      2  HA1 SUR     1      11.090  10.000  10.000\n'
            'ATOM      3  HA2 SUR     1      10.000  11.090  10.000\n'
        )

        structure = Structure.read(path, masses=SUR_MASSES)

        np.testing.assert_allclose(structure.positions[0], [10, 10, 10])


class TestSelection:
    def test_keeps_only_the_aggregate(self, tmp_path):
        path = write_gro(tmp_path / 'sur.gro', SUR_ROWS)

        structure = Structure.read(path, residues=['SUR'], masses=SUR_MASSES)

        assert structure.resnames.tolist() == ['SUR'] * 3

    def test_none_keeps_every_atom(self, tmp_path):
        path = write_gro(tmp_path / 'sur.gro', SUR_ROWS)

        structure = Structure.read(path, masses=SUR_MASSES, elements={'TIP3': ['O', 'H', 'H']})

        assert len(structure.resids) == 6

    @pytest.mark.parametrize(('name', 'residues'), [('model.xyz', ['SUR']), ('sur.gro', ['POPC'])])
    def test_rejects(self, tmp_path, name, residues):
        # A file of another type, or one without the aggregate's
        # residues.
        path = write_gro(tmp_path / name, SUR_ROWS)

        with pytest.raises(ValueError):
            Structure.read(path, residues=residues, masses=SUR_MASSES)


class TestPeriodicWrapping:
    @staticmethod
    def _aggregate():
        """Twenty three-atom molecules around a 6 nm box's middle.

        Returns the .gro rows and the positions in Å.
        """
        rng = np.random.default_rng(5)
        centres = 3 + rng.uniform(-1.5, 1.5, (20, 3))
        xyz = np.repeat(centres, 3, axis=0) + rng.uniform(-0.15, 0.15, (60, 3))
        rows = [
            (k // 3 + 1, 'SUR', ['CA', 'HA1', 'HA2'][k % 3], *position)
            for k, position in enumerate(np.round(xyz, 3))
        ]
        return rows, np.round(xyz, 3) * 10

    def test_rejoins_an_aggregate_split_by_the_box(self, tmp_path):
        # Shifted by 3 nm and wrapped into the box, as GROMACS writes
        # it: molecules and aggregate are split. Read back, the shape is
        # whole.
        rows, xyz = self._aggregate()
        wrapped = [(*row[:3], *((np.array(row[3:]) + 3) % 6)) for row in rows]
        path = write_gro(tmp_path / 'split.gro', wrapped, box='   6.00000   6.00000   6.00000')

        structure = Structure.read(path, masses=SUR_MASSES)

        shape = structure.positions - structure.positions.mean(0)
        np.testing.assert_allclose(shape, xyz - xyz.mean(0), atol=1e-9)

    def test_leaves_a_whole_aggregate_in_place(self, tmp_path):
        rows, xyz = self._aggregate()
        path = write_gro(tmp_path / 'whole.gro', rows, box='   6.00000   6.00000   6.00000')

        structure = Structure.read(path, masses=SUR_MASSES)

        np.testing.assert_allclose(structure.positions, xyz, atol=1e-9)


class TestElements:
    def test_masses_follow_the_position_in_the_residue(self, tmp_path):
        # Masses are matched by position, not by name: CA is carbon.
        path = write_gro(
            tmp_path / 'sur.gro', SUR_ROWS[:3] + [(2, *row[1:]) for row in SUR_ROWS[:3]]
        )

        structure = Structure.read(path, masses=SUR_MASSES)

        assert structure.elements.tolist() == ['C', 'H', 'H'] * 2
        assert structure.element_sources == {'SUR': 'masses'}

    def test_deuterium_by_its_mass(self, tmp_path):
        path = write_gro(tmp_path / 'sur.gro', SUR_ROWS[:3])

        structure = Structure.read(path, masses={'SUR': [12.011, 2.014, 2.014]})

        assert structure.elements.tolist() == ['C', 'D', 'D']

    def test_sources_in_order(self, tmp_path):
        # Masses come before given elements, given elements before the
        # PDB column, and the column before the names.
        path = tmp_path / 'mixed.pdb'
        path.write_text(
            'ATOM      1   CA SUR     1       0.000   0.000   0.000                       N\n'
            'ATOM      2   CA SUG     2       1.000   0.000   0.000                       N\n'
            'ATOM      3   CA SUP     3       2.000   0.000   0.000                       N\n'
            'ATOM      4   CA SUN     4       3.000   0.000   0.000\n'
        )

        with pytest.warns(UserWarning, match='SUN'):
            structure = Structure.read(
                path, masses={'SUR': [12.011]}, elements={'SUR': ['O'], 'SUG': ['O']}
            )

        assert structure.elements.tolist() == ['C', 'O', 'N', 'Ca']
        assert structure.element_sources == {
            'SUG': 'elements',
            'SUN': 'names',
            'SUP': 'pdb',
            'SUR': 'masses',
        }

    def test_name_fallback(self, tmp_path):
        # Only a name that is a two-letter element gives it; any other
        # gives its first letter, leading digits left out. HG1 is no
        # mercury and CO1 no cobalt.
        names = ['CL', 'NA', 'BR', 'C21', 'HG1', 'CO1', '1HB', 'D1']
        rows = [(k + 1, 'X', name, 0.1 * k, 0.0, 0.0) for k, name in enumerate(names)]
        path = write_gro(tmp_path / 'names.gro', rows)

        with pytest.warns(UserWarning, match='guessed'):
            structure = Structure.read(path)

        assert structure.elements.tolist() == ['Cl', 'Na', 'Br', 'C', 'H', 'C', 'H', 'D']

    @pytest.mark.parametrize(
        ('masses', 'elements'),
        [
            ({'SUR': [12.011, 3.024, 1.008]}, None),  # hydrogen mass repartitioning
            ({'SUR': [12.011, 14.027, 1.008]}, None),  # united-atom CH2, 0.02 u off N
            ({'SUR': [12.011, 1.008]}, None),  # residue bigger than its masses
            (None, {'SUR': ['C', 'H']}),  # residue bigger than its elements
        ],
    )
    def test_rejects_unmatched_atoms(self, tmp_path, masses, elements):
        path = write_gro(tmp_path / 'sur.gro', SUR_ROWS[:3])

        with pytest.raises(ValueError):
            Structure.read(path, masses=masses, elements=elements)

    def test_rejects_a_name_with_no_element(self, tmp_path):
        path = write_gro(tmp_path / 'q.gro', [(1, 'X', 'Q2', 0.0, 0.0, 0.0)])

        with pytest.raises(ValueError):
            Structure.read(path)


class TestEditingElements:
    def test_an_edit_changes_sld_and_volume(self, tmp_path):
        path = write_gro(tmp_path / 'sur.gro', SUR_ROWS[:3])
        structure = Structure.read(path, masses=SUR_MASSES)
        sld, volume = structure.sld(), structure.volume

        structure.elements = ['O', 'H', 'H']

        assert structure.sld()[0] != sld[0]
        assert structure.volume[0] != volume[0]

    @pytest.mark.parametrize('elements', [['C', 'H'], ['C', 'H', 'Xx']])
    def test_setter_rejects(self, tmp_path, elements):
        # Not one symbol per atom, or a symbol outside the table.
        path = write_gro(tmp_path / 'sur.gro', SUR_ROWS[:3])
        structure = Structure.read(path, masses=SUR_MASSES)

        with pytest.raises(ValueError):
            structure.elements = elements


class TestAtomProperties:
    def test_values_follow_the_atoms(self):
        # SasView's numbers: carbon 9.07 Å³; deuterium as big as
        # hydrogen, but its SLD is positive where hydrogen's is
        # negative.
        structure = Structure(
            np.arange(4), ['X'] * 4, ['A'] * 4, np.zeros((4, 3)), ['H', 'C', 'D', 'H']
        )

        assert structure.volume == pytest.approx([23.64, 9.066, 23.64, 23.64], rel=1e-3)
        assert structure.sld() == pytest.approx(
            [-1.582e-6, 7.332e-6, 2.820e-6, -1.582e-6], rel=1e-3
        )

    def test_deuterated_residues_change_only_their_hydrogens(self):
        structure = Structure(
            np.arange(4),
            ['SUR', 'SUR', 'TIP3', 'TIP3'],
            ['A'] * 4,
            np.zeros((4, 3)),
            ['C', 'H', 'O', 'H'],
        )

        sld = structure.sld(deuterated=['SUR'])

        assert sld[1] == pytest.approx(2.820e-6, rel=1e-3)
        np.testing.assert_array_equal(sld[[0, 2, 3]], structure.sld()[[0, 2, 3]])

    def test_unknown_deuterated_residue_raises(self):
        structure = Structure([1], ['SUR'], ['A'], np.zeros((1, 3)), ['C'])

        with pytest.raises(ValueError):
            structure.sld(deuterated=['DTA'])

    def test_centre_is_the_bounding_box_middle(self):
        # Not the mean position: atoms at x = 0, 1 and 10 Å give 5 Å.
        positions = [[0.0, 0, 0], [1, 0, 0], [10, 0, 0]]
        structure = Structure(np.arange(3), ['X'] * 3, ['A'] * 3, positions, ['C'] * 3)

        np.testing.assert_allclose(structure.centre, [5, 0, 0])
