.. _examples:

Examples
--------

VLT/SPHERE H-alpha data
~~~~~~~~~~~~~~~~~~~~~~~

An end-to-end example of a `SPHERE/ZIMPOL <https://www.eso.org/sci/facilities/paranal/instruments/sphere.html>`_ H-alpha data set of the accreting M dwarf companion of HD 142527 (see `Cugno et al. 2019 <http://adsabs.harvard.edu/abs/2019A%26A...622A.156C>`_) can be downloaded `here <https://people.phys.ethz.ch/~stolkert/pynpoint/hd142527_zimpol_h-alpha.tgz>`_.

VLT/NACO M' dithering data
~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we show an end-to-end processing example of a pupil-stabilized data set of beta Pic from `Stolker et al. (2019) <http://adsabs.harvard.edu/abs/2019A%26A...622A.156C>`_ (see also :ref:`running`). This archival data set was obtained with `VLT/NACO <https://www.eso.org/sci/facilities/paranal/instruments/naco.html>`_ in the M' band. A dithering pattern was applied to sample the sky background.

First we need to import the Pypeline, as well as the I/O and processing modules. These can be directly imported from the package, for example::

    from pynpoint import Pypeline, FitsReadingModule

Next, we create an instance of :class:`~pynpoint.core.pypeline.Pypeline` with the ``working_place_in`` pointing to a path where PynPoint has enough space to create its database, ``input_place_in`` pointing to the path with the raw FITS files, and ``output_place_in`` to a folder for the output::

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

    module = RemoveLinesModule(lines=(0, 0, 0, 2),
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
                                       center=((263, 263), (116, 263), (116, 116), (263, 116)),
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
                                     guess=(0., 0., 1., 1., 100., 0., 0.))

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

	module = PcaPsfSubtractionModule(pca_numbers=range(1, 51),
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
