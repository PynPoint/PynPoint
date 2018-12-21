.. _tutorial:

Tutorial
========

.. _introduction:

Introduction
------------

The architecture and the user interfaces of PynPoint have changed since version 0.3.0 to enable end-to-end data reduction. Some old PynPoint features such as the workflow are not implemented in the new architecture design. At the moment, the easiest way to run the PynPoint pipeline is with either a Python script or the interactive mode of Python. See the :ref:`running` section for a quick start example including direct imaging data of beta Pic.

The pipeline works with two different components:

1. Pipeline modules which read, write, and process data:

	1.1 :class:`pynpoint.core.processing.ReadingModule` - Reading of the data and relevant header information.

	1.2 :class:`pynpoint.core.processing.WritingModule` - Exporting of results from the database.

	1.3 :class:`pynpoint.core.processing.ProcessingModule` - Processing and analysis of the data.

2. The actual pipeline :class:`pynpoint.core.pypeline` which capsules a list of pipeline modules.

.. _data-types:

Data Types
----------

PynPoint currently works with three types of input and output data:

* FITS files
* HDF5 files
* Text files

PynPoint creates an HDF5 database called ``PynPoin_database.hdf5`` in the ``working_place_in`` of the pipeline. This is the central data storage in which the results of the processing steps are saved. The advantage of the HDF5 data format is that reading of data is much faster compared to the FITS data format and it is possible to quickly read subsets from very large datasets.

Input data is read into the central database with a :class:`pynpoint.core.processing.ReadingModule`. By default, PynPoint will read data from the ``input_place_in`` but setting a manual folder is possible to read data to separate database tags (e.g., dark frames, flat fields, and science data). Here we show an example of how to read FITS files and a list of parallactic angles.

First, we need to create an instance of :class:`pynpoint.core.pypeline.Pypeline`::

    pipeline = Pypeline(working_place_in="/path/to/working_place",
                        input_place_in="/path/to/input_place",
                        output_place_in="/path/to/output_place")

Next, we read the science data from the the default input location::

    module = FitsReadingModule(name_in="read_science",
                               input_dir=None,
                               image_tag="science")

    pipeline.add_module(module)

And we read the flat fields from a separate location::

    module = FitsReadingModule(name_in="read_flat",
                               input_dir="/path/to/flat",
                               image_tag="flat")

    pipeline.add_module(module)

The parallactic angles are read from a text file in the default input folder and attached as attribute to the science data::

    module = ParangReadingModule(file_name,
                                 name_in="parang",
                                 input_dir=None,
                                 data_tag="science")

    pipeline.add_module(module)

Finally, we run all pipeline modules::

    pipeline.run()

Alternatively, it is also possible to run the modules individually by their ``name_in`` value::

    pipeline.run_module("read_science")
    pipeline.run_module("read_flat")
    pipeline.run_module("parang")

The FITS files of the science data and flat fields are read and stored into the central HDF5 database. The data is labelled with a tag which is used by other pipeline module to access data from the database.

Restoring data from an already existing pipeline database can be done by creating an instance of :class:`pynpoint.core.pypeline.Pypeline` with the ``working_place_in`` pointing to the path of the ``PynPoint_database.hdf5`` file.

PynPoint can also handle the HDF5 format as input and output data. Data and corresponding attributes can be exported as HDF5 file with the  :class:`pynpoint.readwrite.hdf5writing` module. This data format can be imported into the central database with the :class:`pynpoint.readwrite.hdf5reading` module. Have a look at the :ref:`pynpoint-package` section for more information.

.. _hdf5-files:

HDF5 Files
----------

There are several options to access data from the central HDF5 database:

	* Use :class:`pynpoint.readwrite.fitswriting.FitsWritingModule` to export data to a FITS file, as shown in the :ref:`end-to-end` section.
	* Use the easy access functions of the :class:`pynpoint.core.pypeline` class to retrieve data and attributes from the database:

		* ``pipeline.get_data(tag='tag_name')``

		* ``pipeline.get_attribute(data_tag='tag_name', attr_name='attr_name')``

	* Use an external tool such as |HDFCompass| or |HDFView| to read, inspect, and visualize data and attributes in the HDF5 database. We recommend using HDFCompass because it is easy to use and has a basic plotting functionality, allowing the user to quickly inspect images from a particular database tag. In HDFCompass, the static attributes can be opened with the `Reopen as HDF5 Attributes` option.

.. |HDFCompass| raw:: html

   <a href="https://support.hdfgroup.org/projects/compass/download.html" target="_blank">HDFCompass</a>

.. |HDFView| raw:: html

   <a href="https://support.hdfgroup.org/downloads/index.html" target="_blank">HDFView</a>

.. _end-to-end:

End-To-End Examples
-------------------

VLT/SPHERE H-alpha data
~~~~~~~~~~~~~~~~~~~~~~~

An end-to-end example of a `SPHERE/ZIMPOL <https://www.eso.org/sci/facilities/paranal/instruments/sphere.html>`_ H-alpha data set of the accreting M dwarf companion of HD 142527 (see `Cugno et al. 2019 <https://arxiv.org/abs/1812.06993>`_) can be downloaded `here <https://people.phys.ethz.ch/~stolkert/hd142527_zimpol_h-alpha.tgz>`_.

VLT/NACO M' dithering data
~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we show an end-to-end processing example of a pupil-stabilized data set of beta Pic from `Stolker et al. (2018) <http://adsabs.harvard.edu/abs/2018arXiv181103336S>`_ (see also :ref:`running`). This archival data set was obtained with `VLT/NACO <https://www.eso.org/sci/facilities/paranal/instruments/naco.html>`_ in the M' band. A dithering pattern was applied to sample the sky background.

First we need to import the Pypeline, as well as the I/O and processing modules::

    from PynPoint import *

Next, we create an instance of :class:`pynpoint.core.pypeline` with the ``working_place_in`` pointing to a path where PynPoint has enough space to create its database, ``input_place_in`` pointing to the path with the raw FITS files, and ``output_place_in`` a folder for the output::

    pipeline = Pypeline(working_place_in="/path/to/working_place",
                        input_place_in="/path/to/input_place",
                        output_place_in"/path/to/output_place")

The FWHM of the PSF is defined for simplicity::

    fwhm = 0.134 # [arcsec]

Now we are ready to add all the pipeline modules that we need. Have a look at the documentation in the :ref:`pynpoint-package` section for a detailed description of the individual modules and their parameters.

1. Import the raw science, flat, and dark images into the database::

    module = FitsReadingModule(name_in="read1",
                               input_dir="/path/to/science/",
                               image_tag="science",
                               overwrite=True,
                               check=True)

    pipeline.add_module(module)

    module = FitsReadingModule(name_in="read2",
                               input_dir="/path/to/flat/",
                               image_tag="flat",
                               overwrite=True,
                               check=False)

    pipeline.add_module(module)

    module = FitsReadingModule(name_in="read4",
                               input_dir="/path/to/dark/",
                               image_tag="dark",
                               overwrite=True,
                               check=False)

    pipeline.add_module(module)

2. Import the image with the (separately processed) unsaturated PSF of the star::

    module = Hdf5ReadingModule(name_in="read4",
                               input_filename="flux.hdf5",
                               input_dir="/path/to/flux/",
                               tag_dictionary={"flux": "flux"})

    pipeline.add_module(module)

3. Remove NDIT+1 frames which contain the average of the FITS cube (NACO specific)::

    module = RemoveLastFrameModule(name_in="last",
                                   image_in_tag="science",
                                   image_out_tag="last")

    pipeline.add_module(module)

4. Calculate the parallactic angles which each image::

    module = AngleCalculationModule(name_in="angle",
                                    data_tag="last",
                                    instrument="NACO")

    pipeline.add_module(module)

5. Remove the top two lines to make the images square::

    module = RemoveLinesModule(lines=(0,0,0,2),
                               name_in="cut",
                               image_in_tag="last",
                               image_out_tag="cut")

    pipeline.add_module(module)

6. Subtract the dark current from the flat field::

    module = DarkCalibrationModule(name_in="dark",
                                   image_in_tag="flat",
                                   dark_in_tag="dark",
                                   image_out_tag="flat_cal")

    pipeline.add_module(module)

7. Divide the science data by the master flat::

    module = FlatCalibrationModule(name_in="flat",
                                   image_in_tag="science",
                                   flat_in_tag="flat_cal",
                                   image_out_tag="science_cal")

    pipeline.add_module(module)

8. Remove the first 5 frames from each FITS cube because of the systematically higher background emission::

    module = RemoveStartFramesModule(frames=5,
                                     name_in="first",
                                     image_in_tag="science_cal",
                                     image_out_tag="first")

    pipeline.add_module(module)

9. PCA based background subtraction::

    module = DitheringBackgroundModule(name_in="background",
                                       image_in_tag="first",
                                       image_out_tag="background",
                                       center=((263.,263.), (116.,263.), (116.,116.), (263,116.)),
                                       cubes=None,
                                       size=3.5,
                                       gaussian=fwhm,
                                       subframe=10.*fwhm,
                                       pca_number=60,
                                       mask_star=4.*fwhm,
                                       mask_planet=None,
                                       subtract_mean=True,
                                       bad_pixel=(9, 5., 3),
                                       crop=True,
                                       prepare=True,
                                       pca_background=True,
                                       combine="pca")

    pipeline.add_module(module)

10. Bad pixel correction::

	module = BadPixelSigmaFilterModule(name_in="bad",
                                           image_in_tag="background",
                                           image_out_tag="bad",
                                           map_out_tag="bpmap",
                                           box=9,
                                           sigma=5.,
                                           iterate=3)

	pipeline.add_module(module)

11. Frame selection::

	module = FrameSelectionModule(name_in="select",
                                      image_in_tag="bad",
                                      selected_out_tag="selected",
                                      removed_out_tag="removed",
                                      index_out_tag=None,
                                      method="median",
                                      threshold=2.,
                                      fwhm=fwhm,
                                      aperture=("circular", fwhm),
                                      position=(None, None, 4.*fwhm))

	pipeline.add_module(module)

12. Extract the star position and center with pixel precision::

	module = StarExtractionModule(name_in="extract",
                                      image_in_tag="selected",
                                      image_out_tag="extract",
                                      index_out_tag="index",
                                      image_size=3.,
                                      fwhm_star=fwhm,
                                      position=(None, None, 4.*fwhm))

	pipeline.add_module(module)

13. Align the images with a cross-correlation of the central 800 mas::

	module = StarAlignmentModule(name_in="align",
                                     image_in_tag="odd",
                                     ref_image_in_tag=None,
                                     image_out_tag="align",
                                     interpolation="spline",
                                     accuracy=10,
                                     resize=None,
                                     num_references=10,
                                     subframe=0.8)

	pipeline.add_module(module)

14. Center the images with subpixel precision by applying a constant shift::

	module = StarCenteringModule(name_in="center",
                                     image_in_tag="align",
                                     image_out_tag="center",
                                     mask_out_tag=None,
                                     fit_out_tag="fit",
                                     method="mean",
                                     interpolation="spline",
                                     radius=5.*fwhm,
                                     sign="positive",
                                     guess=(0., 0., 1., 1., 100., 0.))

	pipeline.add_module(module)

15. Stack by 100 images::

	module = StackAndSubsetModule(name_in="stack",
                                      image_in_tag="center",
                                      image_out_tag="stack",
                                      random=None,
                                      stacking=100)

	pipeline.add_module(stack)

16. Prepare the data for PSF subtraction::

	module = PSFpreparationModule(name_in="prep",
                                      image_in_tag="stack",
                                      image_out_tag="prep",
                                      mask_out_tag=None,
                                      norm=False,
                                      resize=None,
                                      cent_size=fwhm,
                                      edge_size=1.)

	pipeline.add_module(module)

17. PSF subtraction with PCA::

	module = PcaPsfSubtractionModule(pca_numbers=np.arange(1, 51, 1),
                                         name_in="pca",
                                         images_in_tag="prep",
                                         reference_in_tag="prep",
                                         res_mean_tag="pca_mean",
                                         res_median_tag="pca_median",
                                         res_weighted_tag=None,
                                         res_arr_out_tag=None,
                                         res_rot_mean_clip_tag=None,
                                         extra_rot=0.)

	pipeline.add_module(module)

18. Measure the signal-to-noise ratio and false positive fraction::

	module = FalsePositiveModule(position=(50.5, 26.5),
                                    aperture=fwhm/2.,
                                    ignore=True,
                                    name_in="fpf",
                                    image_in_tag="pca_median",
                                    snr_out_tag="fpf")

	pipeline.add_module(module)

19. Write the median residuals to a FITS file::

	module = FitsWritingModule(name_in="write",
                                    file_name="residuals.fits",
                                    output_dir=None,
                                    data_tag="pca_median",
                                    data_range=None)

	pipeline.add_module(module)

20. And finally, run the pipeline::

	pipeline.run()

21. Or, to run a specific pipeline module individually::

	pipeline.run_module("pca")
