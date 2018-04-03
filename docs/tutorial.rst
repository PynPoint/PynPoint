.. _tutorial:

Tutorial
========

.. _introduction:

Introduction
------------

The architecture and the user interfaces of PynPoint have changed since version 0.3.0 to enable end-to-end data reduction. Some old PynPoint features such as the workflow are not implemented in the new architecture design. At the moment, the easiest way to run the PynPoint pipeline is with either a Python script or the interactive mode of Python. See the :ref:`running` section for a quick start example including direct imaging data of beta Pic.

The pipeline works with two different components:

1. Pipeline modules which read, write, and process data:

    1.1 :class:`PynPoint.Core.Processing.ReadingModule` - Read data and relevant header information.
    1.2 :class:`PynPoint.Core.Processing.WritingModule` - Export results of processing modules.
    1.3 :class:`PynPoint.Core.Processing.ProcessingModule` - Process the data (e.g., background subtraction, frame selection).

2. The actual pipeline :class:`PynPoint.Core.Pypeline` which capsules a list of pipeline modules.

.. _data-types:

Data Types
----------

PynPoint currently works with three types of input and output data:

* FITS files
* HDF5 files
* Text files

PynPoint creates an HDF5 database called ``PynPoin_database.hdf5`` in the ``working_place_in`` of the pipeline. This is the central data storage in which the results of the processing steps are saved. The advantage of the HDF5 data format is that reading of data is much faster compared to the FITS data format and it is possible to quickly read subsets from very large datasets.

Input data is read into the central database with a :class:`PynPoint.Core.Processing.ReadingModule`. By default, PynPoint will read data from the ``input_place_in`` but setting a manual folder is possible to read data to separate database tags (e.g., dark frames, flat fields, and science data). Here we show an example of how to read FITS files and a list of parallactic angles.

First, we need to create an instance of :class:`PynPoint.Core.Pypeline.Pypeline`: ::

	pipeline = Pypeline(working_place_in="/path/to/working_place",
                            input_place_in="/path/to/input_place",
                            output_place_in="/path/to/output_place")

Next, we read the science data from the the default input location: ::

	read_science = FitsReadingModule(name_in="read_science",
                                         input_dir=None,
                                         image_tag="science")

	pipeline.add_module(read_science)

And we read the flat fields from a separate location: ::

	read_flat = FitsReadingModule(name_in="read_flat",
                                      input_dir="/path/to/flat",
                                      image_tag="flat")

	pipeline.add_module(read_flat)

The parallactic angles are read from a text file in the default input folder and attached as attribute to the science data: ::

    parang = ParangReadingModule(file_name,
                                 name_in="parang",
                                 input_dir=None,
                                 data_tag="science")

    pipeline.add_module(parang)

Finally, we run all pipeline modules: ::

	pipeline.run()

The FITS files of the science data and flat fields are read and stored into the central HDF5 database. The data is labelled with a tag which is used by other pipeline module to access data from the database.

Restoring data from an already existing pipeline database can be done by creating an instance of :class:`PynPoint.Core.Pypeline.Pypeline` with the ``working_place_in`` pointing to the path of the ``PynPoint_database.hdf5`` file.

PynPoint can also handle the HDF5 format as input and output data. Data and corresponding attributes can be exported as HDF5 file with the  :class:`PynPoint.IOmodules.Hdf5Writing` module. This data format can be imported into the central database with the :class:`PynPoint.IOmodules.Hdf5Reading` module. Have a look at the :ref:`pynpoint-package` section for more information.

.. _hdf5-files:

HDF5 Files
----------

There are several options to access data from the central HDF5 database:

	* Use :class:`PynPoint.IOmodules.FitsWriting.FitsWritingModule` to export data to a FITS file, as shown in the :ref:`end-to-end` section.
	* Use the easy access functions of the :class:`PynPoint.Core.Pypeline` class to retrieve data and attributes from the database:

		* ``pipeline.get_data(tag='tag_name')``

		* ``pipeline.get_attribute(data_tag='tag_name', attr_name='attr_name')``

	* Use an external tool such as |HDFCompass| or |HDFView| to read, inspect, and visualize data and attributes in the HDF5 database. We recommend using HDFCompass because it is easy to use and has a basic plotting functionality, allowing the user to quickly inspect images from a particular database tag. In HDFCompass, the static attributes can be opened with the `Reopen as HDF5 Attributes` option.

.. |HDFCompass| raw:: html

   <a href="https://support.hdfgroup.org/projects/compass/download.html" target="_blank">HDFCompass</a>

.. |HDFView| raw:: html

   <a href="https://support.hdfgroup.org/downloads/index.html" target="_blank">HDFView</a>

.. _end-to-end:

End-To-End Example
------------------

Here we show an end-to-end data reduction example of an ADI data set of beta Pic as presented in Stolker et al. in prep. (see also :ref:`running`). This archival data set was obtained with VLT/NACO in the M' band. A dithering pattern was applied to sample the sky background.

First we need to import the ``Pypeline`` module: ::

	from PynPoint import Pypeline

the pipeline modules for reading and writing FITS, HDF5, and text files: ::

	from PynPoint.IOmodules import FitsReadingModule, FitsWritingModule, \
	                               Hdf5ReadingModule, TextWritingModule

and all the processing modules that we want to run: ::

	from PynPoint.ProcessingModules import RemoveLastFrameModule, AngleInterpolationModule, \
                     RemoveLinesModule, RemoveStartFramesModule, DitheringBackgroundModule, \
                     FrameSelectionModule, BadPixelSigmaFilterModule, StarExtractionModule, \
                     StarAlignmentModule, StarCenteringModule, StackAndSubsetModule, \
                     PSFpreparationModule, FastPCAModule, SimplexMinimizationModule, \
                     MCMCsamplingModule

Next, we create an instance of :class:`PynPoint.Core.Pypeline` with the ``working_place_in`` pointing to a path where PynPoint has enough space to create its database, ``input_place_in`` pointing to the path with the raw FITS files, and ``output_place_in`` a folder for the output: ::

	pipeline = Pypeline(working_place_in="/path/to/working_place",
                            input_place_in="/path/to/input_place",
                            output_place_in"/path/to/output_place")

Now we are ready to add the different pipeline steps. Have a look at the documentation in the :ref:`pynpoint-package` section for a detailed description of the individual modules and their parameters.

1. Read the raw science data: ::

	read = FitsReadingModule(name_in="read",
                                 input_dir=None,
                                 image_tag="im_arr")

	pipeline.add_module(read)

2. Read the frames with the unsaturated PSF of the star (which has been processed separately): ::

    flux = Hdf5ReadingModule(name_in="flux",
                             input_filename="flux.hdf5",
                             input_dir="/path/to/flux",
                             tag_dictionary={"flux": "flux"})

    pipeline.add_module(flux)

3. Remove the last (NDIT+1) frame of each FITS cube (NACO specific): ::

    last = RemoveLastFrameModule(name_in="last",
                                 image_in_tag="science",
                                 image_out_tag="last")

    pipeline.add_module(last)

4. Calculate the parallactic angles for each image with a linear interpolation: ::

    angle = AngleInterpolationModule(name_in="angle",
                                     data_tag="last")

    pipeline.add_module(angle)

5. Remove the top two lines to make the images square: ::

    cut = RemoveLinesModule(lines=(0,0,0,2),
                            name_in="cut",
                            image_in_tag="last",
                            image_out_tag="cut")

    pipeline.add_module(cut)

6. Remove the first three frames of each FITS cube because the background is significantly higher: ::

    first = RemoveStartFramesModule(frames=3,
                                    name_in="first",
                                    image_in_tag="cut",
                                    image_out_tag="first")

    pipeline.add_module(first)

7. Combined mean and PCA-based background subtraction: ::

    background = DitheringBackgroundModule(name_in="background",
                                           image_in_tag="first",
                                           image_out_tag="background",
                                           center=((263., 263.),
                                                   (116., 263.),
                                                   (116., 116.),
                                                   (263., 116.)),
                                           cubes=None,
                                           size=3.5,
                                           gaussian=0.15,
                                           subframe=20,
                                           pca_number=60,
                                           mask=0.7,
                                           crop=True,
                                           prepare=True,
                                           pca_background=True,
                                           combine="pca")

    pipeline.add_module(background)

8. Frame selection: ::

    select = FrameSelectionModule(name_in="select",
                                  image_in_tag="background",
                                  selected_out_tag="selected",
                                  removed_out_tag="removed",
                                  method="median",
                                  threshold=2.,
                                  fwhm=0.2,
                                  aperture=0.1,
                                  position=(None, None, 20))

    pipeline.add_module(select)

8. Bad pixel cleaning: ::

    bad = BadPixelSigmaFilterModule(name_in="bad",
                                    image_in_tag="selected",
                                    image_out_tag="bad",
                                    box=9,
                                    sigma=5,
                                    iterate=2)

    pipeline.add_module(bad)

9. Extract the star position and center with pixel precision: ::

    extract = StarExtractionModule(name_in="extract",
                                   image_in_tag="bad",
                                   image_out_tag="extract",
                                   image_size=3.,
                                   fwhm_star=0.2,
                                   position=(None, None, 20.))

    pipeline.add_module(extract)

10. Align the images with a cross correlation: ::

	align = StarAlignmentModule(name_in="align",
                                    image_in_tag="extract",
                                    ref_image_in_tag=None,
                                    image_out_tag="align",
                                    interpolation="spline",
                                    accuracy=10,
                                    resize=None,
                                    num_references=10)

	pipeline.add_module(align)

11. Center the frames with a constant shift: ::

	center = StarCenteringModule(name_in="center",
                                     image_in_tag="align",
                                     image_out_tag="center",
                                     fit_out_tag="fit",
                                     method="mean",
                                     interpolation="spline",
                                     radius=None)

	pipeline.add_module(center)

12. Stack by 100 images: ::

	stack = StackAndSubsetModule(name_in="stack",
                                     image_in_tag="center",
                                     image_out_tag="stack",
                                     random=None,
                                     stacking=100)

	pipeline.add_module(stack)

13. Prepare the data for PSF subtraction: ::

	prep = PSFpreparationModule(name_in="prep",
                                    image_in_tag="stack",
                                    image_out_tag="prep",
                                    image_mask_out_tag=None,
                                    mask_out_tag=None,
                                    norm=True,
                                    resize=None,
                                    cent_size=0.15,
                                    edge_size=1.5)

	pipeline.add_module(prep)

14. Subtract the stellar PSF with PCA: ::

	pca = FastPCAModule(pca_numbers=np.arange(1, 51, 1),
                            name_in="pca",
                            images_in_tag="prep",
                            reference_in_tag="prep",
                            res_mean_tag="res_mean",
                            res_median_tag=None,
                            res_arr_out_tag=None,
                            res_rot_mean_clip_tag=None,
                            extra_rot=0.)

	pipeline.add_module(pca)

15. Obtain the position and contrast of the planet: ::

	simplex = SimplexMinimizationModule(position=(64., 40.),
                                            magnitude=8.,
                                            psf_scaling=-0.65/0.0233,
                                            name_in="simplex",
                                            image_in_tag="stack",
                                            psf_in_tag="flux",
                                            res_out_tag="simplex",
                                            flux_position_tag="fluxpos",
                                            merit="hessian",
                                            aperture=0.1,
                                            sigma=0.027,
                                            tolerance=0.01,
                                            pca_number=30,
                                            mask=0.15,
                                            extra_rot=0.)

	pipeline.add_module(simplex)

16. Use MCMC sampling to explore the posterior distribution functions: ::

	mcmc = MCMCsamplingModule(param=(0.457, 210., 7.6),
                                  bounds=((0.400, 0.500), (205., 215.), (7., 9.)),
                                  name_in="mcmc",
                                  image_in_tag="stack",
                                  psf_in_tag="flux",
                                  chain_out_tag="mcmc",
                                  nwalkers=200,
                                  nsteps=1000,
                                  psf_scaling=-0.65/0.0233,
                                  pca_number=30,
                                  aperture=0.1,
                                  mask=0.15,
                                  extra_rot=0.,
                                  scale=2.,
                                  sigma=(1e-5, 1e-3, 1e-3))

	pipeline.add_module(mcmc)

17. Write the residuals of the PSF subtraction to a FITS file: ::

	write = FitsWritingModule(name_in="write",
                                  file_name="residuals.fits",
                                  output_dir=None,
                                  data_tag="res_mean",
                                  data_range=None)

	pipeline.add_module(write)

18. Write the results from the simplex minimization to a text file: ::

	text = TextWritingModule(name_in="text",
                                 file_name="simplex.dat",
                                 data_tag="fluxpos",
                                 output_dir=None,
                                 header="Position x [pix] - Position y [pix] - Separation [arcsec] - Angle [deg] - Contrast [mag] - Merit")

	pipeline.add_module(text)

19. And finally, run the pipeline: ::

	pipeline.run()
