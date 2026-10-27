from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import pytest

from coords2sans.calculator import SansCalculator
from coords2sans.calculator import SansTask
from coords2sans.structure import Structure

# One carbon, and a solvent that gives carbon a contrast of 1e-6 Å⁻²
CARBON = Structure([1], ['X'], ['C'], [[0.0, 0, 0]], ['C'])
CONTRAST = 1e-6
SOLVENT_SLD = CARBON.sld()[0] - CONTRAST

# 100 carbons and hydrogens of one residue, spread over about 40 Å
RNG = np.random.default_rng(7)
ELEMENTS = RNG.choice(['C', 'H'], 100)
RANDOM_ATOMS = Structure(
    [1] * 100, ['X'] * 100, ELEMENTS, RNG.normal(size=(100, 3)) * 20, ELEMENTS
)


class TestQ:
    @pytest.mark.parametrize(('q_log', 'step'), [(True, np.log), (False, np.asarray)])
    def test_spans_the_settings_in_even_steps(self, q_log, step):
        calculator = SansCalculator(0.0, q_min=0.01, q_max=0.3, q_points=7, q_log=q_log)

        q = calculator.q

        assert len(q) == 7
        assert q[[0, -1]] == pytest.approx([0.01, 0.3], rel=1e-12)
        np.testing.assert_allclose(np.diff(step(q)), np.diff(step(q)).mean(), rtol=1e-12)


class TestOneSize:
    @pytest.mark.parametrize('average_amplitude', [False, True])
    def test_forward_scattering(self, average_amplitude):
        # At q = 0 every atom adds its full amplitude:
        # I(0) = scale (Σ Δρ V)² 1e8 / Σ V + background.
        volume = RANDOM_ATOMS.volume
        contrast = RANDOM_ATOMS.sld() + 0.56e-6
        expected = 2 * (contrast @ volume) ** 2 * 1e8 / volume.sum() + 0.1
        calculator = SansCalculator(
            -0.56e-6,
            q_min=0,
            q_max=0,
            q_points=1,
            q_log=False,
            scale=2,
            background=0.1,
            average_amplitude=average_amplitude,
        )

        curve = calculator.calculate(RANDOM_ATOMS)

        assert curve.intensity == pytest.approx([expected], rel=1e-12)

    @pytest.mark.parametrize('count', [3, 1500])
    def test_debye_sum_over_all_pairs(self, count):
        # Atoms 1 Å apart on a line: each pairs with itself at 0 Å,
        # and there are 2 (count − d) pairs at d Å. 1500 atoms take
        # three blocks of pairs.
        positions = np.c_[np.arange(count), np.zeros((count, 2))]
        carbons = ['C'] * count
        structure = Structure(range(count), ['X'] * count, carbons, positions, carbons)
        volume = CARBON.volume[0]
        calculator = SansCalculator(SOLVENT_SLD, q_min=0.01, q_max=1, q_points=16)
        q = calculator.q
        d = np.arange(1, count)
        pairs = count + 2 * (count - d) @ (np.sin(np.outer(d, q)) / np.outer(d, q))
        expected = (CONTRAST * volume) ** 2 * pairs * 1e8 / (count * volume)

        curve = calculator.calculate(structure)

        np.testing.assert_allclose(curve.intensity, expected, rtol=1e-9)

    def test_debye_distances_rounded_to_the_bin_width(self):
        # The exact pair sum, without bins. With 100 atoms the rounding
        # errors of the pairs cancel less than in a micelle: 1.4e-4.
        amplitudes = (RANDOM_ATOMS.sld() + 0.56e-6) * RANDOM_ATOMS.volume
        distances = np.linalg.norm(
            RANDOM_ATOMS.positions[:, None] - RANDOM_ATOMS.positions, axis=-1
        )
        calculator = SansCalculator(-0.56e-6)
        pairs = [amplitudes @ np.sinc(q * distances / np.pi) @ amplitudes for q in calculator.q]
        expected = np.array(pairs) * 1e8 / RANDOM_ATOMS.volume.sum()

        curve = calculator.calculate(RANDOM_ATOMS)

        np.testing.assert_allclose(curve.intensity, expected, rtol=1e-3)

    def test_average_amplitude_from_the_bounding_box_middle(self):
        # As SasView, distances count from the middle of the bounding
        # box, not from the mean position: atoms at x = 0, 1 and 10 Å
        # lie 5, 4 and 5 Å from x = 5 Å.
        positions = [[0.0, 0, 0], [1, 0, 0], [10, 0, 0]]
        structure = Structure([1, 2, 3], ['X'] * 3, ['C'] * 3, positions, ['C'] * 3)
        volume = CARBON.volume[0]
        calculator = SansCalculator(
            SOLVENT_SLD, q_min=0.01, q_max=1, q_points=16, average_amplitude=True
        )
        q = calculator.q
        form = CONTRAST * volume * (2 * np.sin(5 * q) / (5 * q) + np.sin(4 * q) / (4 * q))

        curve = calculator.calculate(structure)

        np.testing.assert_allclose(curve.intensity, form**2 * 1e8 / (3 * volume), rtol=1e-12)

    def test_average_amplitude_of_uniform_sphere_matches_analytic_sphere(self):
        # The sphere of sasmodels' "sphere" model: I = Δρ² V P(qR)² 1e8.
        # Radii R u^(1/3) on a regular u grid fill it evenly; ± pairs
        # put the bounding-box middle at 0. R is set so that the
        # carbons' volumes add up to the sphere's. The averaged
        # amplitude sees only the radii; the Debye sum would add each
        # atom's pair with itself, which a smooth sphere does not have.
        count = 20000
        sphere_volume = 2 * count * CARBON.volume[0]
        radius = (3 * sphere_volume / (4 * np.pi)) ** (1 / 3)
        directions = np.random.default_rng(1).normal(size=(count, 3))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        radii = radius * ((np.arange(count) + 0.5) / count) ** (1 / 3)
        positions = np.r_[radii[:, None] * directions, -radii[:, None] * directions]
        carbons = ['C'] * 2 * count
        sphere = Structure([1] * 2 * count, ['X'] * 2 * count, carbons, positions, carbons)
        calculator = SansCalculator(SOLVENT_SLD, q_min=0.001, q_max=0.2, average_amplitude=True)
        x = calculator.q * radius
        expected = (
            CONTRAST**2 * sphere_volume * (3 * (np.sin(x) - x * np.cos(x)) / x**3) ** 2 * 1e8
        )
        far_from_zeros = expected > 1e-4 * expected[0]

        curve = calculator.calculate(sphere)

        np.testing.assert_allclose(
            curve.intensity[far_from_zeros], expected[far_from_zeros], rtol=1e-4
        )

    def test_deuterated_hydrogen_is_deuterium(self):
        names = ['C1', 'H11', 'H12']
        positions = [[0.0, 0, 0], [1.09, 0, 0], [0, 1.09, 0]]
        light = Structure([1] * 3, ['SUR'] * 3, names, positions, ['C', 'H', 'H'])
        heavy = Structure([1] * 3, ['SUR'] * 3, names, positions, ['C', 'D', 'D'])
        deuterating = SansCalculator(-0.56e-6, deuterated=['SUR'])

        result = deuterating.calculate(light)

        expected = SansCalculator(-0.56e-6).calculate(heavy)
        np.testing.assert_allclose(result.intensity, expected.intensity, rtol=1e-12)

    def test_curve_holds_q_and_label(self):
        calculator = SansCalculator(-0.56e-6, q_points=8)

        curve = calculator.calculate(RANDOM_ATOMS, 'eq5')

        np.testing.assert_array_equal(curve.q, calculator.q)
        assert curve.label == 'eq5'


class TestSizes:
    @pytest.mark.parametrize(('polydispersity', 'size_points'), [(0.0, 35), (0.1, 1)])
    def test_one_size(self, polydispersity, size_points):
        calculator = SansCalculator(0.0, polydispersity=polydispersity, size_points=size_points)

        factors, shares = calculator._sizes()

        assert factors.tolist() == [1.0]
        assert shares.tolist() == [1.0]

    def test_gaussian_over_three_sigma(self):
        # sasmodels' Gaussian: σ = 0.1, 5 sizes at -3, -1.5, 0, 1.5 and
        # 3 σ from the mean, each counting by the normal curve's height
        # there.
        calculator = SansCalculator(0.0, polydispersity=0.1, size_points=5)
        heights = np.exp(-np.array([9, 2.25, 0, 2.25, 9]) / 2)

        factors, shares = calculator._sizes()

        np.testing.assert_allclose(factors, [0.7, 0.85, 1, 1.15, 1.3], rtol=1e-12)
        np.testing.assert_allclose(shares, heights / heights.sum(), rtol=1e-12)

    def test_drops_sizes_at_or_below_zero(self):
        # σ = 0.5 reaches from -0.5 to 2.5: sizes -0.5 and 0 are
        # dropped.
        calculator = SansCalculator(0.0, polydispersity=0.5, size_points=7)

        factors, shares = calculator._sizes()

        np.testing.assert_allclose(factors, [0.5, 1, 1.5, 2, 2.5], rtol=1e-12)
        assert shares.sum() == pytest.approx(1, rel=1e-12)

    @pytest.mark.parametrize(('polydispersity', 'size_points'), [(-0.1, 35), (0.1, 0)])
    def test_calculate_rejects_bad_settings(self, polydispersity, size_points):
        # Settings may change after init, so calculate checks them.
        calculator = SansCalculator(0.0)
        calculator.polydispersity = polydispersity
        calculator.size_points = size_points

        with pytest.raises(ValueError, match='polydispersity'):
            calculator.calculate(RANDOM_ATOMS)


class TestPolydispersity:
    @pytest.mark.parametrize(
        ('average_amplitude', 'one_size'),
        [
            # Debye: two self pairs, and the pair 4 Å apart both ways
            (False, lambda x: 2 + 2 * np.sin(4 * x) / (4 * x)),
            # (Σ sinc)²: each carbon 2 Å from the middle
            (True, lambda x: (2 * np.sin(2 * x) / (2 * x)) ** 2),
        ],
    )
    def test_sizes_scale_positions_and_volumes(self, average_amplitude, one_size):
        # Two carbons 4 Å apart. A size f moves them to 4 f Å and makes
        # their volumes f³ as big: I_f = (Δρ V f³)² one_size(f q). The
        # shares n combine them: I = Σ n I_f 1e8 / Σ n 2 V f³.
        positions = [[-2.0, 0, 0], [2, 0, 0]]
        structure = Structure([1, 2], ['X'] * 2, ['C'] * 2, positions, ['C'] * 2)
        volume = CARBON.volume[0]
        calculator = SansCalculator(
            SOLVENT_SLD,
            q_min=0.01,
            q_max=1,
            q_points=16,
            polydispersity=0.4,
            size_points=5,
            average_amplitude=average_amplitude,
        )
        q = calculator.q
        factors, shares = calculator._sizes()
        sizes = [(CONTRAST * volume * f**3) ** 2 * one_size(f * q) for f in factors]
        expected = np.dot(shares, sizes) * 1e8 / (shares @ (2 * volume * factors**3))

        curve = calculator.calculate(structure)

        np.testing.assert_allclose(curve.intensity, expected, rtol=1e-12)

    def test_background_is_added_once(self):
        # polyJD adds it once per size; here it is added to the sum.
        without = SansCalculator(-0.56e-6, q_points=32, polydispersity=0.4)
        with_background = SansCalculator(-0.56e-6, q_points=32, polydispersity=0.4, background=0.1)

        difference = (
            with_background.calculate(RANDOM_ATOMS).intensity
            - without.calculate(RANDOM_ATOMS).intensity
        )

        np.testing.assert_allclose(difference, 0.1, rtol=1e-9)


@pytest.fixture
def sur_file(tmp_path):
    """A .gro of one SUR (C, H, H) beside one water.

    Its atom names alone would make calcium of the carbon (CA).
    """
    path = tmp_path / 'sur.gro'
    path.write_text(
        'SUR in water\n'
        '    6\n'
        '    1SUR     CA    1   1.000   1.000   1.000\n'
        '    1SUR    HA1    2   1.109   1.000   1.000\n'
        '    1SUR    HA2    3   1.000   1.109   1.000\n'
        '    2TIP3   OH2    4   0.300   0.300   0.300\n'
        '    2TIP3    H1    5   0.396   0.300   0.300\n'
        '    2TIP3    H2    6   0.300   0.396   0.300\n'
        '   3.00000   3.00000   3.00000\n'
    )
    return path


SUR_MASSES = {'SUR': [12.011, 1.008, 1.008]}


class TestSansTask:
    @pytest.mark.parametrize(
        ('masses', 'elements'), [(SUR_MASSES, None), (None, {'SUR': ['C', 'H', 'H']})]
    )
    def test_run_reads_the_file_and_calculates(self, sur_file, masses, elements):
        # The water is left out and the carbon is carbon, as when the
        # file is read and calculated by hand.
        calculator = SansCalculator(-0.56e-6, q_points=16)
        task = SansTask(sur_file, calculator, ['SUR'], masses, elements, 'SUR')
        structure = Structure.read(sur_file, residues=['SUR'], masses=masses, elements=elements)
        expected = calculator.calculate(structure, 'SUR')

        curve = task.run()

        np.testing.assert_array_equal(curve.intensity, expected.intensity)
        assert curve.label == 'SUR'

    def test_runs_in_another_process(self, sur_file):
        # The point of plain data: a spawn child, as on Windows and
        # macOS, unpickles the task and gives the same curve.
        calculator = SansCalculator(-0.56e-6, q_points=16)
        task = SansTask(sur_file, calculator, ['SUR'], SUR_MASSES, label='SUR')
        with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn')) as pool:
            (curve,) = pool.map(SansTask.run, [task])

        np.testing.assert_array_equal(curve.intensity, task.run().intensity)
        assert curve.label == 'SUR'
