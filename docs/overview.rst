.. _overview:

Overview
========

Here you find a list of all available pipeline modules with a very short summary of what each module does. Reading modules import data into the database, writing modules export data from the database, and processing modules run a specific task for the data reduction and analysis. More details on the design of the pipeline can be found in the :ref:`architecture` section.

.. _readmodule:

Reading Modules
---------------

* :class:`FitsReadingModule`: Reading FITS files and adding the images and relevant header information to the database.
* :class:`Hdf5ReadingModule`: Reading datasets and attributes from HDF5 files that have been created by PynPoint.
* :class:`ParangReadingModule`: Reading a list of parallactic angles and adding them as attribute to a dataset.
* :class:`AttributeReadingModule`: Reading a list of values and adding them as attribute to a dataset

.. _writemodule:

Writing Modules
---------------

* :class:`FitsWritingModule`: Export a dataset from the database to a FITS file.
* :class:`Hdf5WritingModule`: Export part of the database to a new HDF5 file.
* :class:`TextWritingModule`: Export a dataset to an ASCII file.
* :class:`ParangWritingModule`: Export the parallactic angles of a dataset to an ASCII file.
* :class:`AttributeWritingModule`: Export a list of attribute values to an ASCII file.

.. _procmodule:

Processing Modules
------------------

Background Subtraction
~~~~~~~~~~~~~~~~~~~~~~

* :class:`SimpleBackgroundSubtractionModule`:
* :class:`MeanBackgroundSubtractionModule`:
* :class:`LineSubtractionModule`:
* :class:`NoddingBackgroundModule`:

Bad Pixel Cleaning
~~~~~~~~~~~~~~~~~~

* :class:`BadPixelSigmaFilterModule`:
* :class:`BadPixelInterpolationModule`:
* :class:`BadPixelMapModule`:
* :class:`BadPixelTimeFilterModule`:
* :class:`ReplaceBadPixelsModule`:

Basic Processing
~~~~~~~~~~~~~~~~

* :class:`SubtractImagesModule`:
* :class:`AddImagesModule`:
* :class:`RotateImagesModule`:

Centering
~~~~~~~~~

* :class:`StarExtractionModule`:
* :class:`StarAlignmentModule`:
* :class:`StarCenteringModule`:
* :class:`ShiftImagesModule`:
* :class:`WaffleCenteringModule`:

Dark and Flat Correction
~~~~~~~~~~~~~~~~~~~~~~~~

* :class:`DarkCalibrationModule`:
* :class:`FlatCalibrationModule`:

Denoising
~~~~~~~~~

* :class:`WaveletTimeDenoisingModule`:
* :class:`TimeNormalizationModule`:

Detection Limits
~~~~~~~~~~~~~~~~

* :class:`ContrastCurveModule`:

Flux and Position
~~~~~~~~~~~~~~~~~

* :class:`FakePlanetModule`:
* :class:`SimplexMinimizationModule`:
* :class:`FalsePositiveModule`:
* :class:`MCMCsamplingModule`:
* :class:`AperturePhotometryModule`:

Frame Selection
~~~~~~~~~~~~~~~

* :class:`RemoveFramesModule`:
* :class:`FrameSelectionModule`:
* :class:`RemoveLastFrameModule`:
* :class:`RemoveStartFramesModule`:

Image Resizing
~~~~~~~~~~~~~~

* :class:`CropImagesModule`:
* :class:`ScaleImagesModule`:
* :class:`AddLinesModule`:
* :class:`RemoveLinesModule`:

PCA Background Subtraction
~~~~~~~~~~~~~~~~~~~~~~~~~~

* :class:`PCABackgroundPreparationModule`:
* :class:`PCABackgroundSubtractionModule`:
* :class:`DitheringBackgroundModule`:

PSF Preparation
~~~~~~~~~~~~~~~

* :class:`PSFpreparationModule`:
* :class:`AngleInterpolationModule`:
* :class:`AngleCalculationModule`:
* :class:`SortParangModule`:
* :class:`SDIpreparationModule`:

PSF Subtraction
~~~~~~~~~~~~~~~

* :class:`PcaPsfSubtractionModule`:
* :class:`ClassicalADIModule`:


Stacking
~~~~~~~~

* :class:`StackAndSubsetModule`:
* :class:`MeanCubeModule`:
* :class:`DerotateAndStackModule`:
* :class:`CombineTagsModule`:
