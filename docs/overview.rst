.. _overview:

Overview
========

Here you find a list of all available pipeline modules with a very short summary of what each module does. Reading modules import data into the database, writing modules export data from the database, and processing modules run a specific task for the data reduction and analysis. More details on the design of the pipeline can be found in the :ref:`architecture` section.

.. _readmodule:

Reading Modules
---------------

* :class:`FitsReadingModule`: Importing FITS files and relevant header information into the database.
* :class:`Hdf5ReadingModule`: Importing datasets and attributes from an HDF5 file (created by PynPoint).
* :class:`ParangReadingModule`: Importing a list of parallactic angles as dataset attribute.
* :class:`AttributeReadingModule`: Importing a list of values as dataset attribute.

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

* :class:`SimpleBackgroundSubtractionModule`: Simple background subtraction for dithering datasets.
* :class:`MeanBackgroundSubtractionModule`: Mean background subtraction for dithering datasets.
* :class:`LineSubtractionModule`: Subtraction of striped artifacts.
* :class:`NoddingBackgroundModule`: Background subtraction for nodding datasets.

Bad Pixel Cleaning
~~~~~~~~~~~~~~~~~~

* :class:`BadPixelSigmaFilterModule`: Sigma filter which finds and replaces bad pixels.
* :class:`BadPixelInterpolationModule`: Interpolate bad pixels with a spectral deconvolution technique.
* :class:`BadPixelMapModule`: Create a bad pixel map from dark and flat images.
* :class:`BadPixelTimeFilterModule`: Sigma clipping of bad pixels along the time dimension.
* :class:`ReplaceBadPixelsModule`: Replace bad pixels that are selected from a bad pixel map.

Basic Processing
~~~~~~~~~~~~~~~~

* :class:`SubtractImagesModule`: Subtract two stacks of images.
* :class:`AddImagesModule`: Add two stacks of images
* :class:`RotateImagesModule`: Rotate a stack of images.

Centering
~~~~~~~~~

* :class:`StarExtractionModule`: Locate the position of the star.
* :class:`StarAlignmentModule`: Align the images with a cross-correlation.
* :class:`StarCenteringModule`: Center the images by fitting a 2D Gaussian of Moffat function.
* :class:`ShiftImagesModule`: Shift a stack of images.
* :class:`WaffleCenteringModule`: Use waffle spots to center the images.

Dark and Flat Correction
~~~~~~~~~~~~~~~~~~~~~~~~

* :class:`DarkCalibrationModule`: Dark frame subtraction.
* :class:`FlatCalibrationModule`: Flat field correction.

Denoising
~~~~~~~~~

* :class:`WaveletTimeDenoisingModule`: Wavelet-based denoising in the time domain.
* :class:`TimeNormalizationModule`: Normalize the images.

Detection Limits
~~~~~~~~~~~~~~~~

* :class:`ContrastCurveModule`: Calculate a contrast curve.

Flux and Position
~~~~~~~~~~~~~~~~~

* :class:`FakePlanetModule`: Inject an artificial planet in a dataset.
* :class:`SimplexMinimizationModule`: Determine the flux and position with a simplex minimization.
* :class:`FalsePositiveModule`: Compute the signal-to-noise ratio and false positive fraction.
* :class:`MCMCsamplingModule`: Estimate the flux and position of a planet with MCMC sampling.
* :class:`AperturePhotometryModule`: Measure the integrated flux at a position.

Frame Selection
~~~~~~~~~~~~~~~

* :class:`RemoveFramesModule`: Remove images by there index number.
* :class:`FrameSelectionModule`: Frame selection to remove low-quality image.
* :class:`RemoveLastFrameModule`: Remove the last image of a VLT/NACO dataset.
* :class:`RemoveStartFramesModule`: Remove images at the beginning of each original data cube.

Image Resizing
~~~~~~~~~~~~~~

* :class:`CropImagesModule`: Crop the images.
* :class:`ScaleImagesModule`: Resample the images (spatially and/or in flux).
* :class:`AddLinesModule`: Add pixel lines on the sides of the images.
* :class:`RemoveLinesModule`: Remove pixel lines from the sides of the images.

PCA Background Subtraction
~~~~~~~~~~~~~~~~~~~~~~~~~~

* :class:`PCABackgroundPreparationModule`: Preparation for the PCA-based background subtraction.
* :class:`PCABackgroundSubtractionModule`: PCA-based background subtraction.
* :class:`DitheringBackgroundModule`: Wrapper for background subtraction of dithering datasets.

PSF Preparation
~~~~~~~~~~~~~~~

* :class:`PSFpreparationModule`: Mask the images before the PSF subtraction.
* :class:`AngleInterpolationModule`: Interpolate the parallactic angles between the start and end values.
* :class:`AngleCalculationModule`: Calculate the parallactic angles.
* :class:`SortParangModule`: Sort the images by parallactic angle.
* :class:`SDIpreparationModule`: Prepare the images for SDI with a rescaling.

PSF Subtraction
~~~~~~~~~~~~~~~

* :class:`PcaPsfSubtractionModule`: PSF subtraction with PCA.
* :class:`ClassicalADIModule`: PSF subtraction with classical ADI.


Stacking
~~~~~~~~

* :class:`StackAndSubsetModule`: Stack and/or select a random subset of the images.
* :class:`MeanCubeModule`: Compute the mean of each original data cube.
* :class:`DerotateAndStackModule`: Derotate and/or stack the images.
* :class:`CombineTagsModule`: Combine multiple database tags into a single dataset.
