========
Usage
========

PynPoint can be used in a number of ways. Since the PynPoint architecture has change completely in version 0.3.0 some old user interfaces are not ported to the new version, yet. The easiest way to use the new PynPoint Pipeline is to write a small python script or use the interactive mode of python. We provide test data to help you get started.

PynPoint works through two different components:

1. Pipeline modules which read and process the raw data and finally write out the results. Three different module types exist:

	1.1 :class:`PynPoint.core.Processing.ReadingModule` - read in raw data and their header information

	1.2 :class:`PynPoint.core.Processing.ProcessingModule` - process the data (e.g. dark-/flat-/background-/PSF-subtraction)

	1.3 :class:`PynPoint.core.Processing.WritingModule` - write out the results of ProcessingModules

2. The actual pipeline :class:`PynPoint.core.Pypeline` - capsules a list of pipeline modules

Interactive
-----------

To analyse data, in the examples below, we assume a directory (`input_place_in`) that contains a set of .fits files (raw data), a directory (`working_place_in`) where PynPoint has enough space to create its database and a directory (`output_place_in`) for the results. First you need to enter the Python command line: ::

	$ ipython 

Next we need to import the PynPoint Pypeline module, ::

	from PynPoint import Pypeline

the pipeline modules for reading and writing .fits data ::

	from PynPoint.io_modules import ReadFitsCubesDirectory, WriteAsSingleFitsFile

and all pipeline modules (pipeline steps) we what to execute: ::

	from PynPoint.processing_modules import BadPixelCleaningSigmaFilterModule, \
	DarkSubtractionModule, FlatSubtractionModule, CutTopTwoLinesModule, \
	AngleCalculationModule, SimpleBackgroundSubtractionModule, \ 
	StarExtractionModule, StarAlignmentModule, PSFSubtractionModule, \
	StackAndSubsetModule

First step we need to create an instance of the :class:`PynPoint.core.Pypeline` ::

	pipeline = Pypeline(working_place_in,
				input_place_in,
				output_place_in)

Now we are ready to add the different pipeline steps. For an explanation about the different modules check out their documentation in the :ref:`pynpoint-package` documentation. Input- and output-tags/-ports will be explained in :ref:`pipeline-architecture`. 

reading the raw data: ::

	reading_data = ReadFitsCubesDirectory(name_in="Fits_reading",
	                                    image_tag="im_arr")
	pipeline.add_module(reading_data)

reading the dark from the directory `dark_dir`: ::

	reading_dark = ReadFitsCubesDirectory(name_in="Dark_reading",
                                      	  input_dir= dark_dir,
                                      	  image_tag="dark_arr")
	pipeline.add_module(reading_dark)

reading the flat from the directory `flat_dir`: ::

	reading_flat = ReadFitsCubesDirectory(name_in="Flat_reading",
                                      	  input_dir= flat_dir,
                                      	  image_tag=“flat_arr")
	pipeline.add_module(reading_flat)

cutting the top two lines of the input frames (Needed for NACO Data): ::

	cutting = CutTopTwoLinesModule(name_in="NACO_cutting",
	                                image_in_tag="im_arr",
	                                image_out_tag="im_arr_cut")
	pipeline.add_module(cutting)

dark and flat subtraction: ::

	dark_sub = DarkSubtractionModule(name_in="dark_subtraction",
                           		image_in_tag="im_arr_cut",
                           		dark_in_tag="dark_arr",
                           		image_out_tag="dark_sub_arr")

	flat_sub = FlatSubtractionModule(name_in="flat_subtraction",
                           		image_in_tag="dark_sub_arr",
                           		flat_in_tag="flat_arr",
                           		image_out_tag="flat_sub_arr")

	pipeline.add_module(dark_sub)
	pipeline.add_module(flat_sub)

bad pixel cleaning: ::

	bp_cleaning = BadPixelCleaningSigmaFilterModule(name_in="sigma_filtering",
	                                                image_in_tag="flat_sub_arr",
	                                                image_out_tag="im_arr_bp_clean")
	pipeline.add_module(bp_cleaning)

background subtraction: ::

	bg_subtraction = SimpleBackgroundSubtractionModule(name_in="background_subtraction",
							star_pos_shift=602,
                                                   	image_in_tag="im_arr_bp_clean",
                                                   	image_out_tag="bg_cleaned_arr")
	pipeline.add_module(bg_subtraction)

star extraction and alignment: ::

	extraction = StarExtractionModule(name_in="star_cutting",
	                                  image_in_tag="bg_cleaned_arr",
	                                  image_out_tag="im_arr_cut",
	                                  psf_size=4,
	                                  fwhm_star=7)

	alignment = StarAlignmentModule(name_in="star_align",
	                                image_in_tag="im_arr_cut",
	                                image_out_tag="im_arr_aligned",
	                                accuracy=100,
	                                resize=2)
	pipeline.add_module(extraction)
	pipeline.add_module(alignment)

calculating the parallactic angle: ::

	angle_calc = AngleCalculationModule(name_in="angle_calculation",
	                                    data_tag="im_arr_aligned")
	pipeline.add_module(angle_calc)

subsampling the data by stacking: ::

	subset = StackAndSubsetModule(name_in="stacking_subset",
	                              image_in_tag="im_arr_aligned",
	                              image_out_tag="im_stacked",
	                              random_subset=None,
	                              stacking=20)
	pipeline.add_module(subset)

subtract the stars PSF using PCA: ::

	psf_sub = PSFSubtractionModule(pca_number=10,
	                               name_in="PSF_subtraction",
	                               images_in_tag="im_stacked",
	                               reference_in_tag="im_stacked",
	                               res_mean_tag="res_mean")
	pipeline.add_module(psf_sub)

writing out the result of the last step: ::

	writing = WriteAsSingleFitsFile(name_in="Fits_writing",
	                                file_name="test.fits",
	                                data_tag="res_mean")
	pipeline.add_module(writing)

**And finally run the pipeline:** ::

	pipeline.run()

You should see the process of the pipeline.
	
In the example above, the star is modelled using the first 10 principal components and the stack is averaged using the mean. 

All of the functions above have a number of keywords that can also be passed to them. More details of these keyword options are discussed in the :ref:`pynpoint-package` section.
	
Workflow
--------
The workflow is not supported in version 0.3.0.

Command line interface
----------------------
No command line interface supported in version 0.3.0

Data types
----------

PynPoint currently works with three input data types:

* fits files

* hdfs files

* save/restore files



The first time you use fits files as inputs, PynPoint will create a HDF5 of the data inside the same directory as the fits files. This is because the HDF5 file is much faster to read than several thousands of small fits files. To use fits inputs, you will need to put all the fits files in one directory and then pass this directory to the appropriate PynPoint call. The PynPoint method will then look for all *.fits files in that folder. In 'interactive' mode, this can be done by::

	images = PynPoint.images.create_wdir(dir_in)
	
When using the workflow make sure that ``intype`` is set to ``dir`` in the config file:: 

	intype = dir

HDF5 files, such as those created after you process a directory of fits files, can also be passed directly::

	images = PynPoint.images.create_whdf5input("filename.hdf5")
	
Alternatively, is can set in the workflow using::

	intype = hdf5
	
The main PynPoint instances also include a save and restore feature. To save the state of an instance::

	images.save("images_savefile.hdf5")
	
Later, an instance can be restored::

	images = PynPoint.images.create_restore("images_savefile.hdf5")


Data
----

To help you get started quickly and easily, we provide access to data. As part of the distribution, we provide data that has been stacked by averaging over 500 images at a time. See the install section for instructions on how to process this data. 

The path to the data can be retrieved by running::

	import PynPoint
	print(PynPoint.get_data_dir())

We also make available `the full data <http://www.phys.ethz.ch/~amaraa/Data_betapic_L_Band_PynPoint_conv.hdf5>`_  (without stacking). This is the data that we used to develop PynPoint and is discussed in more detail in our papers. It consists of the high-contrast imaging data-set used to confirm the existence of a massive exoplanet planet orbiting the nearby A-type star beta Pictoris (Lagrange et al. 2010). 

The data-set was taken on 2009 December 26 at the Very Large Telescope with the high-resolution, adaptive optics assisted, near-infrared camera NACO in the L' filter (central wavelengths 3.8 micron) in Angular Differential Imaging (ADI) mode. It consists of 80 data cubes, each containing 300 individual exposures with an individual exposure time of 0.2 s. The total field rotation of the full data-set amounted to ~44 degrees  on the sky. The raw data are publicly available from the |ESO_Archive| (Program ID: 084.C-0739(A)). 

For the test data, basic data reduction steps (sky subtraction, bad pixel correction and alignment of images) were already done as explained in Quanz et al. (2011). The final postage stamp size of the individual images is 73 x 73 pixels in the original image size. For PynPoint, we doubled the resolution, resulting in 146 x 146 pixels for the test data images. The same test data was also used in |Amara_Quanz|, where we introduced the PynPoint algorithm.


.. |Amara_Quanz| raw:: html

   <a href="http://adsabs.harvard.edu/abs/2012MNRAS.427..948A" target="_blank">Amara & Quanz (2012)</a>

.. |ESO_Archive| raw:: html

   <a href="http://archive.eso.org/cms/eso-data.html" target="_blank"> European Southern Observatory (ESO) archive </a>


