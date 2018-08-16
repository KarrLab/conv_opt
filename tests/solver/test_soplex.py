""" Tests for the SoPlex solver

:Author: Jonathan Karr <jonrkarr@gmail.com>
:Date: 2018-08-14
:Copyright: 2018, Karr Lab
:License: MIT
"""

from ..test_core import SolverTestCase
import attrdict
import conv_opt
import numpy
import mock
import unittest


if conv_opt.Solver.soplex in conv_opt.ENABLED_SOLVERS:
    import soplex


@unittest.skipUnless(conv_opt.Solver.soplex in conv_opt.ENABLED_SOLVERS, 'SoPlex is not installed')
class SoplexTestCase(SolverTestCase):
    pass
