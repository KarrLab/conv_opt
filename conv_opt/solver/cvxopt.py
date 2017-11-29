""" CVXOPT module

:Author: Jonathan Karr <jonrkarr@gmail.com>
:Date: 2017-11-22
:Copyright: 2017, Karr Lab
:License: MIT
"""

from __future__ import absolute_import
from .cvxpy import CvxpyModel
from ..core import VariableType, ConvOptError
import cvxpy


class CvxoptModel(CvxpyModel):
    """ CVXOPT solver """
    SOLVER = cvxpy.CVXOPT

    def load(self, conv_opt_model):
        """ Load a model to CVXPY's data structure

        Args:
            conv_opt_model (:obj:`cvxpy.Problem`): model

        Returns:
            :obj:`dict`: the model in CVXPY's data structure

        Raises:
            :obj:`ConvOptError`: if a variable has an unsupported type
        """

        solver_model = super(CvxoptModel, self).load(conv_opt_model)

        for variable in conv_opt_model.variables:
            if variable.type != VariableType.continuous:
                raise ConvOptError('Unsupported variable of type "{}"'.format(variable.type))

        return solver_model
