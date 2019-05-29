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


To use PACO directly, import one of the the processing modules (Fast or Full PACO are currently available). Fast PACO is recommended, as the loss of accuracy in the SNR is small, while the computation time is much, much lower.

Create an instance of the class, specifying the patch size in the number of pixels in a circular patch (recommend 13, **49** or 113, which are 3,4, and 5 pixel radius circular patches respectively):

.. code:: python
    import paco.processing.fastpaco as fastPACO
    fp = fastPACO.fastPACO(angles = [...],patch_size = 49)

Set the stack of frames to be processed:

.. code:: python
    fp.setImageSequence(image_sequence)

Supplying the list of rotation angles between frames, and the pixel scaling, run PACO:
.. code:: python
    a,b = fp.PACO(...)

This returns 2D maps for a and b, the the inverse variance and flux estimate respectively. The signal to noise can be computed as b/sqrt(a).

Directory Structure
-------------------
- paco: Contains python modules that implement the PACO algorithm and various IO and utility functions.
  - processing: Implementation of PACO algorithms
  - util: Utility functions, including rotations, coordinate transformations, distributions and models

- testData: Contains toy dataset used in testing.
- output: Location of output files/directories

Run the Example.ipynb notebook to use. Currently this notebook generates a toy dataset, and runs the FastPACO algorithm to produce a signal-to-noise (SNR) map of the data.

Algorithms
----------
fullpaco - Algorithm 1 from [1].
fastpaco - Algorithm 2 from [1]. Adds preprocessing of statistics to reduce computation time at expense of accuracy.
PACO.fluxEstimate - Algorithm 3 from [1] Unbiased estimation of source flux

Requirements
------------
- Python 3.0
- pipenv

Environment
-----------
This package includes a pipenv environment file to include the necessary packages (listed in the Pipfile).

