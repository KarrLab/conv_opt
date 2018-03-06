Installation
============

Requirements
--------------------------

First, install `Python <https://www.python.org>`_ and `pip <https://pip.pypa.io>`_. The following command illustrates how to install Python and pip on Ubuntu Linux:

    .. code-block:: bash

        apt-get install python python-pip


Optional requirements
--------------------------

Second, optionally install additional solvers. Please see our detailed `instructions <http://docs.karrlab.org/intro_to_wc_modeling/latest/installation.html>`_.

* `Cbc <https://projects.coin-or.org/cbc>`_ and `CyLP <mpy.github.io/CyLPdoc/>`_
* `FICO XPRESS <http://www.fico.com/en/products/fico-xpress-optimization>`_
* `Gurobi <http://www.gurobi.com/products/gurobi-optimizer>`_
* `IBM CPLEX <https://www-01.ibm.com/software/commerce/optimization/cplex-optimizer>`_
* `MOSEK Optimizer <https://www.mosek.com>`_


Installing this package
---------------------------

Third, install this package. The latest release of this package can be installed from PyPI using this command:

    .. code-block:: bash

        pip install conv_opt

Alternatively, the latest version of this package can be installed from GitHub using this command:

    .. code-block:: bash

        pip install git+https://github.com/KarrLab/conv_opt.git#egg=conv_opt

Support for the optional solvers can be installed using the following options::

    pip install conv_opt[cbc,cplex,gurobi,mosek,xpress]
