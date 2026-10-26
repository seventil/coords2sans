import subprocess  # noqa: S404
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from coords2sans.curve import Curve

matplotlib.use('Agg')


@pytest.fixture
def curve():
    q = np.geomspace(0.002, 0.2, 8)
    return Curve(q, 37.5 / (1 + (q / 0.02) ** 2), 'eq5')


class TestSave:
    def test_writes_two_columns_under_a_header(self, curve, tmp_path):
        # SasView's ASCII reader takes columns as Q, I, dI, dQ: a
        # third column would become error bars, so there are exactly
        # two, split by ', '.
        path = tmp_path / 'eq5.dat'

        curve.save(path)

        lines = path.read_text().splitlines()
        assert lines[0] == '# q (1/A), I (1/cm)'
        assert len(lines) == 1 + len(curve.q)
        assert all(len(line.split(', ')) == 2 for line in lines[1:])


class TestLoad:
    def test_reads_back_what_save_wrote(self, curve, tmp_path):
        path = tmp_path / 'eq5.dat'
        curve.save(path)

        loaded = Curve.load(path)

        np.testing.assert_array_equal(loaded.q, curve.q)
        np.testing.assert_array_equal(loaded.intensity, curve.intensity)
        assert loaded.label == 'eq5'

    def test_label_given_overrides_the_file_name(self, curve, tmp_path):
        path = tmp_path / 'eq5.dat'
        curve.save(path)

        assert Curve.load(path, label='stage 5').label == 'stage 5'


class TestPlot:
    def test_draws_log_log_and_overlays(self, curve):
        ax = curve.plot()
        Curve(curve.q, 2 * curve.intensity, 'eq5 x2').plot(ax)

        assert len(ax.get_lines()) == 2
        assert (ax.get_xscale(), ax.get_yscale()) == ('log', 'log')
        assert [text.get_text() for text in ax.get_legend().get_texts()] == ['eq5', 'eq5 x2']
        plt.close(ax.figure)

    def test_no_legend_without_a_label(self, curve):
        ax = Curve(curve.q, curve.intensity).plot()

        assert ax.get_legend() is None
        plt.close(ax.figure)


def test_importing_curve_does_not_import_matplotlib():
    # matplotlib is the optional 'plot' extra: only Curve.plot may
    # import it.
    code = 'import sys, coords2sans.curve; assert "matplotlib" not in sys.modules'
    subprocess.run([sys.executable, '-c', code], check=True)  # noqa: S603
