PACO Implementation
===================
This package implements the algorithms developed by [Flasseur et. al 2018 [1]](https://www.aanda.org/articles/aa/abs/2018/10/aa32745-18/aa32745-18.html)

It is implemented as a [Pynpoint](https://pynpoint.readthedocs.io/en/latest/) module, providing convinient usage as part of the Pynpoint Pipeline. The source code is located in the "paco" directory.

Authors
-------
Polychronis Patapis, _ETH Zurich_

Evert Nasedkin, _ETH Zurich_

email: evertn@student.ethz.ch

Usage
-----
Currently, the Example, Pynpoint_Example and Data_from_gabriele notebooks provide the best overview of usage. 

To use as a Pynpoint module, call the PACOModule class constructor and add the module to the pipeline. The details of the constructors used are documented within the source files. As a Pynpoint module, Pynpoint will handle file IO.
.. code:: python
    from pynpoint import Pypeline
    from paco import PACOModule

    pipeline = Pypeline(...)
    module = PACOModule(...)
    pipeline.add_module(module)


To use PACO directly, import one of the the processing modules (Fast or Full PACO are currently available). Fast PACO is recommended, as the loss of accuracy in the SNR is small, while the computation time is much faster.

File Structure
-------------------
  - processing/pacoModule.py: Contains python modules that setup and run paco through the Pynpoint pipeline. Includes the standalone pacoModule, and the pacoContrastModule, which will compute a contrast curve using PACO.
  - util/paco.py: Contains the modules that implement the paco algorithms (Full PACO, Fast PACO and flux estimation.
  - util/limits.py: Contains the function to compute the contrast at a specified location using PACO.
  - util/pacomath.py: General utility and math functions for basic computations.


Algorithms
----------
fullpaco - Algorithm 1 from [1].
fastpaco - Algorithm 2 from [1]. Adds preprocessing of statistics to reduce computation time at expense of accuracy.
PACO.fluxEstimate - Algorithm 3 from [1] Unbiased estimation of source flux

