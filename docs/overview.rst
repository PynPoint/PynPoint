.. _overview:

Overview
========

Here you find a list of all available pipeline modules with a very short description of what each module does. Reading modules import data into the database, writing modules export data from the database, and processing modules run a specific task of the data reduction and analysis. More details on the design of the pipeline can be found in the :ref:`architecture` section.

.. note::
   All PynPoint classes ending with ``Module`` in their name (e.g. :class:`~pynpoint.readwrite.fitsreading.FitsReadingModule`) are pipeline modules that can be added to an instance of :class:`~pynpoint.core.pypeline.Pypeline` (see :ref:`pypeline` section).

.. _readmodule:

Reading Modules
---------------

* :class:`~pynpoint.readwrite.fitsreading.FitsReadingModule`: Import FITS files and relevant header information into the database.
* :class:`~pynpoint.readwrite.hdf5reading.Hdf5ReadingModule`: Import datasets and attributes from an HDF5 file (as created by PynPoint).
* :class:`~pynpoint.readwrite.textreading.ParangReadingModule`: Import a list of parallactic angles as dataset attribute.
* :class:`~pynpoint.readwrite.textreading.AttributeReadingModule`: Import a list of values as dataset attribute.

.. _writemodule:

Writing Modules
---------------

* :class:`~pynpoint.readwrite.fitswriting.FitsWritingModule`: Export a dataset from the database to a FITS file.
* :class:`~pynpoint.readwrite.hdf5writing.Hdf5WritingModule`: Export part of the database to a new HDF5 file.
* :class:`~pynpoint.readwrite.textwriting.TextWritingModule`: Export a dataset to an ASCII file.
* :class:`~pynpoint.readwrite.textwriting.ParangWritingModule`: Export the parallactic angles of a dataset to an ASCII file.
* :class:`~pynpoint.readwrite.textwriting.AttributeWritingModule`: Export a list of attribute values to an ASCII file.

.. _procmodule:

Processing Modules
------------------

Background Subtraction
~~~~~~~~~~~~~~~~~~~~~~

* :class:`~pynpoint.processing.background.SimpleBackgroundSubtractionModule`: Simple background subtraction for dithering datasets.
* :class:`~pynpoint.processing.background.MeanBackgroundSubtractionModule`: Mean background subtraction for dithering datasets.
* :class:`~pynpoint.processing.background.LineSubtractionModule`: Subtraction of striped detector artifacts.
* :class:`~pynpoint.processing.background.NoddingBackgroundModule`: Background subtraction for nodding datasets.

Bad Pixel Cleaning
~~~~~~~~~~~~~~~~~~

* :class:`~pynpoint.processing.badpixel.BadPixelSigmaFilterModule`: Find and replace bad pixels with a sigma filter.
* :class:`~pynpoint.processing.badpixel.BadPixelInterpolationModule`: Interpolate bad pixels with a spectral deconvolution technique.
* :class:`~pynpoint.processing.badpixel.BadPixelMapModule`: Create a bad pixel map from dark and flat images.
* :class:`~pynpoint.processing.badpixel.BadPixelTimeFilterModule`: Sigma clipping of bad pixels along the time dimension.
* :class:`~pynpoint.processing.badpixel.ReplaceBadPixelsModule`: Replace bad pixels based on a bad pixel map.

Basic Processing
~~~~~~~~~~~~~~~~

* :class:`~pynpoint.processing.basic.SubtractImagesModule`: Subtract two stacks of images.
* :class:`~pynpoint.processing.basic.AddImagesModule`: Add two stacks of images
* :class:`~pynpoint.processing.basic.RotateImagesModule`: Rotate a stack of images.

Centering
~~~~~~~~~

* :class:`~pynpoint.processing.centering.StarExtractionModule`: Locate the position of the star.
* :class:`~pynpoint.processing.centering.StarAlignmentModule`: Align the images with a cross-correlation.
* :class:`~pynpoint.processing.centering.StarCenteringModule`: Center the images by fitting a 2D Gaussian or Moffat function.
* :class:`~pynpoint.processing.centering.ShiftImagesModule`: Shift a stack of images.
* :class:`~pynpoint.processing.centering.WaffleCenteringModule`: Use the waffle spots to center the images.

Dark and Flat Correction
~~~~~~~~~~~~~~~~~~~~~~~~

* :class:`~pynpoint.processing.darkflat.DarkCalibrationModule`: Dark frame subtraction.
* :class:`~pynpoint.processing.darkflat.FlatCalibrationModule`: Flat field correction.

Denoising
~~~~~~~~~

* :class:`~pynpoint.processing.timedenoising.WaveletTimeDenoisingModule`: Wavelet-based denoising in the time domain.
* :class:`~pynpoint.processing.timedenoising.TimeNormalizationModule`: Normalize a stack of images.

Detection Limits
~~~~~~~~~~~~~~~~

* :class:`~pynpoint.processing.limits.ContrastCurveModule`: Compute a contrast curve.

Flux and Position
~~~~~~~~~~~~~~~~~

* :class:`~pynpoint.processing.fluxposition.FakePlanetModule`: Inject an artificial planet in a dataset.
* :class:`~pynpoint.processing.fluxposition.SimplexMinimizationModule`: Determine the flux and position with a simplex minimization.
* :class:`~pynpoint.processing.fluxposition.FalsePositiveModule`: Compute the signal-to-noise ratio and false positive fraction.
* :class:`~pynpoint.processing.fluxposition.MCMCsamplingModule`: Estimate the flux and position of a planet with MCMC sampling.
* :class:`~pynpoint.processing.fluxposition.AperturePhotometryModule`: Compute the integrated flux at a position.

Frame Selection
~~~~~~~~~~~~~~~

* :class:`~pynpoint.processing.frameselection.RemoveFramesModule`: Remove images by their index number.
* :class:`~pynpoint.processing.frameselection.FrameSelectionModule`: Frame selection to remove low-quality image.
* :class:`~pynpoint.processing.frameselection.RemoveLastFrameModule`: Remove the last image of a VLT/NACO dataset.
* :class:`~pynpoint.processing.frameselection.RemoveStartFramesModule`: Remove images at the beginning of each original data cube.

Image Resizing
~~~~~~~~~~~~~~

* :class:`~pynpoint.processing.resizing.CropImagesModule`: Crop the images.
* :class:`~pynpoint.processing.resizing.ScaleImagesModule`: Resample the images (spatially and/or in flux).
* :class:`~pynpoint.processing.resizing.AddLinesModule`: Add pixel lines on the sides of the images.
* :class:`~pynpoint.processing.resizing.RemoveLinesModule`: Remove pixel lines from the sides of the images.

PCA Background Subtraction
~~~~~~~~~~~~~~~~~~~~~~~~~~

* :class:`~pynpoint.processing.pcabackground.PCABackgroundPreparationModule`: Preparation for the PCA-based background subtraction.
* :class:`~pynpoint.processing.pcabackground.PCABackgroundSubtractionModule`: PCA-based background subtraction.
* :class:`~pynpoint.processing.pcabackground.DitheringBackgroundModule`: Wrapper for background subtraction of dithering datasets.

PSF Preparation
~~~~~~~~~~~~~~~

* :class:`~pynpoint.processing.psfpreparation.PSFpreparationModule`: Mask the images before the PSF subtraction.
* :class:`~pynpoint.processing.psfpreparation.AngleInterpolationModule`: Interpolate the parallactic angles between the start and end values.
* :class:`~pynpoint.processing.psfpreparation.AngleCalculationModule`: Calculate the parallactic angles.
* :class:`~pynpoint.processing.psfpreparation.SortParangModule`: Sort the images by parallactic angle.
* :class:`~pynpoint.processing.psfpreparation.SDIpreparationModule`: Prepare the images for SDI.

PSF Subtraction
~~~~~~~~~~~~~~~

* :class:`~pynpoint.processing.psfsubtraction.PcaPsfSubtractionModule`: PSF subtraction with PCA.
* :class:`~pynpoint.processing.psfsubtraction.ClassicalADIModule`: PSF subtraction with classical ADI.

Stacking
~~~~~~~~

* :class:`~pynpoint.processing.stacksubset.StackAndSubsetModule`: Stack and/or select a random subset of the images.
* :class:`~pynpoint.processing.stacksubset.StackCubesModule`: Collapse each original data cube separately.
* :class:`~pynpoint.processing.stacksubset.DerotateAndStackModule`: Derotate and/or stack the images.
* :class:`~pynpoint.processing.stacksubset.CombineTagsModule`: Combine multiple database tags into a single dataset.