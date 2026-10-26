"""Small-angle neutron scattering curves I(q) from atom coordinates.

``Structure`` reads the atoms of an aggregate, ``SansCalculator``
computes its ``Curve``, and ``SansTask`` does both from a file, in
this process or in another.
"""

from importlib.metadata import version

from coords2sans.calculator import SansCalculator
from coords2sans.calculator import SansTask
from coords2sans.curve import Curve
from coords2sans.structure import Structure

__version__ = version('coords2sans')
