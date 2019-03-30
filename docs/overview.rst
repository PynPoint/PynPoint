.. _overview:

Overview
========

Here you find a list of all available pipeline modules with a very short description of what each module does. Reading modules import data into the database, writing modules export data from the database, and processing modules run a specific task of the data reduction and analysis. More details on the design of the pipeline can be found in the :ref:`architecture` section.

.. _readmodule:

Reading Modules
---------------

* :class:`FitsReadingModule`: Imports FITS files and relevant header information into the database.
* :class:`Hdf5ReadingModule`: Imports datasets and attributes from an HDF5 file (as created by PynPoint).
* :class:`ParangReadingModule`: Imports a list of parallactic angles as dataset attribute.
* :class:`AttributeReadingModule`: Imports a list of values as dataset attribute.

.. _writemodule:

Writing Modules
---------------

* :class:`FitsWritingModule`: Exports a dataset from the database to a FITS file.
* :class:`Hdf5WritingModule`: Exports part of the database to a new HDF5 file.
* :class:`TextWritingModule`: Exports a dataset to an ASCII file.
* :class:`ParangWritingModule`: Exports the parallactic angles of a dataset to an ASCII file.
* :class:`AttributeWritingModule`: Exports a list of attribute values to an ASCII file.

.. _procmodule:

Processing Modules
------------------

Background Subtraction
~~~~~~~~~~~~~~~~~~~~~~

* :class:`SimpleBackgroundSubtractionModule`: Simple background subtraction for dithering datasets.
* :class:`MeanBackgroundSubtractionModule`: Mean background subtraction for dithering datasets.
* :class:`LineSubtractionModule`: Subtraction of striped detector artifacts.
* :class:`NoddingBackgroundModule`: Background subtraction for nodding datasets.

Bad Pixel Cleaning
~~~~~~~~~~~~~~~~~~

* :class:`BadPixelSigmaFilterModule`: Finds and replaces bad pixels with a sigma filter
* :class:`BadPixelInterpolationModule`: Interpolates bad pixels with a spectral deconvolution technique.
* :class:`BadPixelMapModule`: Creates a bad pixel map from dark and flat images.
* :class:`BadPixelTimeFilterModule`: Sigma clipping of bad pixels along the time dimension.
* :class:`ReplaceBadPixelsModule`: Replaces bad pixels based on a bad pixel map.

Basic Processing
~~~~~~~~~~~~~~~~

* :class:`SubtractImagesModule`: Subtracts two stacks of images.
* :class:`AddImagesModule`: Adds two stacks of images
* :class:`RotateImagesModule`: Rotates a stack of images.

Centering
~~~~~~~~~

* :class:`StarExtractionModule`: Locates the position of the star.
* :class:`StarAlignmentModule`: Aligns the images with a cross-correlation.
* :class:`StarCenteringModule`: Centers the images by fitting a 2D Gaussian or Moffat function.
* :class:`ShiftImagesModule`: Shifts a stack of images.
* :class:`WaffleCenteringModule`: Uses waffle spots to center the images.

Dark and Flat Correction
~~~~~~~~~~~~~~~~~~~~~~~~

* :class:`DarkCalibrationModule`: Dark frame subtraction.
* :class:`FlatCalibrationModule`: Flat field correction.

Denoising
~~~~~~~~~

* :class:`WaveletTimeDenoisingModule`: Wavelet-based denoising in the time domain.
* :class:`TimeNormalizationModule`: Normalizes the images.

Detection Limits
~~~~~~~~~~~~~~~~

* :class:`ContrastCurveModule`: Computes a contrast curve.

Flux and Position
~~~~~~~~~~~~~~~~~

* :class:`FakePlanetModule`: Injects an artificial planet in a dataset.
* :class:`SimplexMinimizationModule`: Determines the flux and position with a simplex minimization.
* :class:`FalsePositiveModule`: Computes the signal-to-noise ratio and false positive fraction.
* :class:`MCMCsamplingModule`: Estimates the flux and position of a planet with MCMC sampling.
* :class:`AperturePhotometryModule`: Measures the integrated flux at a position.

Frame Selection
~~~~~~~~~~~~~~~

* :class:`RemoveFramesModule`: Removes images by their index number.
* :class:`FrameSelectionModule`: Frame selection to remove low-quality image.
* :class:`RemoveLastFrameModule`: Removes the last image of a VLT/NACO dataset.
* :class:`RemoveStartFramesModule`: Removes images at the beginning of each original data cube.

Image Resizing
~~~~~~~~~~~~~~

* :class:`CropImagesModule`: Crops the images.
* :class:`ScaleImagesModule`: Resamples the images (spatially and/or in flux).
* :class:`AddLinesModule`: Adds pixel lines on the sides of the images.
* :class:`RemoveLinesModule`: Resmoves pixel lines from the sides of the images.

PCA Background Subtraction
~~~~~~~~~~~~~~~~~~~~~~~~~~

* :class:`PCABackgroundPreparationModule`: Preparation for the PCA-based background subtraction.
* :class:`PCABackgroundSubtractionModule`: PCA-based background subtraction.
* :class:`DitheringBackgroundModule`: Wrapper for background subtraction of dithering datasets.

PSF Preparation
~~~~~~~~~~~~~~~

* :class:`PSFpreparationModule`: Masks the images before the PSF subtraction.
* :class:`AngleInterpolationModule`: Interpolates the parallactic angles between the start and end values.
* :class:`AngleCalculationModule`: Calculates the parallactic angles.
* :class:`SortParangModule`: Sorts the images by parallactic angle.
* :class:`SDIpreparationModule`: Prepares the images for SDI.

PSF Subtraction
~~~~~~~~~~~~~~~

* :class:`PcaPsfSubtractionModule`: PSF subtraction with PCA.
* :class:`ClassicalADIModule`: PSF subtraction with classical ADI.


Stacking
~~~~~~~~

* :class:`StackAndSubsetModule`: Stacks and/or selects a random subset of the images.
* :class:`MeanCubeModule`: Computes the mean of each original data cube.
* :class:`DerotateAndStackModule`: Derotates and/or stacks the images.
* :class:`CombineTagsModule`: Combines multiple database tags into a single dataset.
