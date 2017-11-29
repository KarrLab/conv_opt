""" GLPK module

:Author: Jonathan Karr <jonrkarr@gmail.com>
:Date: 2017-11-22
:Copyright: 2017, Karr Lab
:License: MIT
"""

from __future__ import absolute_import
from .optlang import OptlangModel
from ..core import QuadraticTerm, ConvOptError
import optlang.glpk_interface
import swiglpk


class GlpkModel(OptlangModel):
    """ GLPK solver """

    INTERFACE = optlang.glpk_interface

    def get_stats(self):
        """ Get diagnostic information about the model

        Returns:
            :obj:`dict`: diagnostic information about the model
        """
        model = self._model

        return {
            'is_integer': model.is_integer,
            'bf_exists': swiglpk.glp_bf_exists(model.problem),
            'bf_updated': swiglpk.glp_bf_updated(model.problem),
            'warm_up': swiglpk.glp_warm_up(model.problem),
        }
