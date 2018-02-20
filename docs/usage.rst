========
Usage
========

PynPoint can be used in a number of ways. Since PynPoint version 0.3.0 the architecture and the user interfaces have change completely in order to enable raw data processing. Due to this redesign some old features like the workflow are missing in the new architecture. At the moment the easiest way to use the PynPoint Pipeline is to write a small python script or use the interactive mode of python. We provide test data to help you get started.

The PynPoint Pipeline works through two different components:

1. Pipeline modules which read and process the raw data and finally write out the results. Three different module types exist:

	1.1 :class:`PynPoint.core.Processing.ReadingModule` - read in raw data and their header information

	1.2 :class:`PynPoint.core.Processing.ProcessingModule` - process the data (e.g. dark-/flat-/background-/PSF-subtraction)

	1.3 :class:`PynPoint.core.Processing.WritingModule` - exports or displays the results of previous ProcessingModules

2. The actual pipeline :class:`PynPoint.Core.Pypeline` - capsules a list of pipeline modules


.. _interactive:

Interactive
-----------

To analyse data, in the examples below, we assume a directory (`input_place_in`) that contains a set of .fits files (raw data), a directory (`working_place_in`) where PynPoint has enough space to create its database and a directory (`output_place_in`) for the results. First you need to enter the Python command line: ::

	$ ipython 

Next we need to import the PynPoint Pypeline module, ::

	from PynPoint import Pypeline

the pipeline modules for reading and writing .fits data ::

	from PynPoint.IOmodules import FitsReadingModule, FitsWritingModule

and all pipeline modules (pipeline steps) we want to execute: ::

	from PynPoint.ProcessingModules import BadPixelCleaningSigmaFilterModule, \
	DarkSubtractionModule, FlatSubtractionModule, CutTopLinesModule, \
	AngleCalculationModule, MeanBackgroundSubtractionModule, \ 
	StarExtractionModule, StarAlignmentModule, PSFSubtractionModule, \
	StackAndSubsetModule, RemoveLastFrameModule

In order to be able to handle the different processing steps we need to create an instance of the :class:`PynPoint.Core.Pypeline` ::

	pipeline = Pypeline(working_place_in,
                        input_place_in,
                        output_place_in)

Now we are ready to add the different pipeline steps. For an explanation about the individual modules check out their documentation in the :ref:`pynpoint-package`. Input- and output-tags/-ports will be explained in :ref:`architecture`. According to |Amara_Quanz2| the following processing steps need to be added for a simple end to end ADI data processing pipeline:

1. Read the raw data: ::

	reading_data = ReadFitsCubesDirectory(name_in="Fits_reading",
                                          image_tag="im_arr")

	pipeline.add_module(reading_data)

2. Import the dark current from the directory `dark_dir`: ::

	reading_dark = ReadFitsCubesDirectory(name_in="Dark_reading",
                                      	  input_dir=dark_dir,
                                      	  image_tag="dark_arr")

	pipeline.add_module(reading_dark)

3. Read the flat-field exposure from the directory `flat_dir`: ::

	reading_flat = ReadFitsCubesDirectory(name_in="Flat_reading",
                                      	  input_dir=flat_dir,
                                      	  image_tag="flat_arr")

	pipeline.add_module(reading_flat)

4. Remove the last (NDIT+1) frame from each cube: ::

    remove_last = RemoveLastFrameModule(name_in="last_frame",
                                        image_in_tag="im_arr",
                                        image_out_tag="im_arr_last")

    pipeline.add_module(remove_last)

5. Cut the top two lines of the input frames (Needed for NACO Data): ::

	cutting = CutTopLinesModule(name_in="NACO_cutting",
                                image_in_tag="im_arr",
                                image_out_tag="im_arr_cut",
                                num_lines=2)

	pipeline.add_module(cutting)

6. Dark- and flat-subtraction: ::

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

7. Background subtraction: ::

    bg_subtraction = MeanBackgroundSubtractionModule(star_pos_shift=None,
                                                     cubes_per_position=1,
                                                     name_in="background_subtraction",
                                                     image_in_tag="flat_sub_arr",
                                                     image_out_tag="bg_cleaned_arr")

    pipeline.add_module(bg_subtraction)

8. Bad pixel cleaning: ::

	bp_cleaning = BadPixelCleaningSigmaFilterModule(name_in="sigma_filtering",
	                                                image_in_tag="flat_sub_arr",
	                                                image_out_tag="bp_cleaned_arr")

	pipeline.add_module(bp_cleaning)

9. Star extraction and alignment: ::

	extraction = StarExtractionModule(name_in="star_cutting",
	                                  image_in_tag="bg_cleaned_arr",
	                                  image_out_tag="im_arr_extract",
	                                  image_size=1.0,
	                                  fwhm_star=0.1)

	alignment = StarAlignmentModule(name_in="star_align",
	                                image_in_tag="im_arr_extract",
	                                image_out_tag="im_arr_aligned",
	                                accuracy=100,
	                                resize=2)

	pipeline.add_module(extraction)
	pipeline.add_module(alignment)

10. Calculate the parallactic angle: ::

	angle_calc = AngleCalculationModule(name_in="angle_calculation",
	                                    data_tag="im_arr_aligned")

	pipeline.add_module(angle_calc)

101. Subsample the data using stacking: ::

	subset = StackAndSubsetModule(name_in="stacking_subset",
	                              image_in_tag="im_arr_aligned",
	                              image_out_tag="im_arr_stacked",
	                              random_subset=None,
	                              stacking=4)

	pipeline.add_module(subset)

12. Subtract the stars PSF using PCA: ::

	psf_sub = PSFSubtractionModule(pca_number=10,
	                               name_in="PSF_subtraction",
	                               images_in_tag="im_arr_stacked",
	                               reference_in_tag="im_arr_stacked",
	                               res_mean_tag="res_mean")

	pipeline.add_module(psf_sub)

13. Write out the result of the last step: ::

	writing = WriteAsSingleFitsFile(name_in="Fits_writing",
	                                file_name="test.fits",
	                                data_tag="res_mean")

	pipeline.add_module(writing)

**And finally run the pipeline:** ::

	pipeline.run()

You should see the process of the pipeline.
	
In the example above, the star is modelled using the first 10 principal components and the stack is averaged using the mean. 

All of the functions above have a number of keywords that can also be passed to them. More details of these keyword options are discussed in the :ref:`pynpoint-package` section.
	
Python Skript
-------------
Another way of using the PynPoint pipeline is to create a python script and run it. Just copy the same lines of code from the :ref:`interactive` section into an empty .py file an run it using: ::

$ python test_file.py

Data types
----------

PynPoint works with two types of input data:

* FITS files

* HDF5 files

The first time you use FITS files as inputs, PynPoint will create an HDF5 database in the *working_place_in* of the Pypeline. This is because the HDF5 file is much faster to read than small FITS files and it provides the possibility to read subsets of huge datasets. To read FITS files as input, you will need to put all the FITS files in one directory and then pass this directory to the appropriate PynPoint Pypeline (*input_place_in*). Next you need to add a FitsReadingModule. If you do not define an own input directory for this ReadingModule it will look for data in the Pypeline default location *input_place_in*. Setting a own directory makes it possible to to read for example dark frames or flat field exposures from different directories. If you run the PynPoint Pypeline, the FitsReadingModule will look for all FITS files in the given folder and imports them into the Pypeline HDF5 database. In *interactive* mode, this can be done by::

	pipeline = Pypeline(working_place_in,
                            input_place_in,
                            output_place_in)

	# takes the default location
	reading_data = ReadFitsCubesDirectory(name_in="Fits_reading",
	                                      image_tag="im_arr")
	pipeline.add_module(reading_data)

	# uses own location 
	reading_flat = ReadFitsCubesDirectory(name_in="Flat_reading",
                                       	      input_dir=some/own/location,
                                              image_tag="flat_arr")
	pipeline.add_module(reading_flat)
	
	pipeline.run()

The code above will read all FITS files form the *input_place_in* and *some/own/location* and stores them into the Pypeline HDF5 database. The chosen tags are important for other Pypeline steps in order to let them access data directly from this database.

If you want to restore data from a Pypeline database which is located in a folder *some/folder/on/drive* you just need to create a Pypeline instance with a *working_place_in*=*some/folder/on/drive* like: ::

	pipeline = Pypeline(some/folder/on/drive,
                            input_place_in,
                            output_place_in)

HDF5 files can be an input as well. Using a :class:`PynPoint.IOmodules.Hdf5Writing` module you can export data from a Pypeline database. This data can be imported using a :class:`PynPoint.IOmodules.Hdf5Reading` module later. For more information have a look at the package documentation.

Workflow
--------
The workflow is not supported in version 0.3.0.

Command line interface
----------------------
No command line interface supported in version 0.3.0

Data
----

To help you get started quickly and easily, we provide access to data. As part of the distribution, we provide data that has been stacked by averaging over 500 images at a time. See the install section for instructions on how to process this data. 

The path to the data can be retrieved by running::

	import PynPoint
	print(PynPoint.get_data_dir())

We also make available `the full data <http://www.phys.ethz.ch/~amaraa/Data_betapic_L_Band_PynPoint_conv.hdf5>`_  (without stacking). This is the data that we used to develop PynPoint and is discussed in more detail in our papers. It consists of the high-contrast imaging data-set used to confirm the existence of a massive exoplanet planet orbiting the nearby A-type star beta Pictoris (Lagrange et al. 2010). 

The data-set was taken on 2009 December 26 at the Very Large Telescope with the high-resolution, adaptive optics assisted, near-infrared camera NACO in the L' filter (central wavelengths 3.8 micron) in Angular Differential Imaging (ADI) mode. It consists of 80 data cubes, each containing 300 individual exposures with an individual exposure time of 0.2 s. The total field rotation of the full data-set amounted to ~44 degrees  on the sky. The raw data are publicly available from the |ESO_Archive| (Program ID: 084.C-0739(A)). 

For the test data, basic data reduction steps (sky subtraction, bad pixel correction and alignment of images) were already done as explained in Quanz et al. (2011) using the other pipeline modules introduced in the :ref:`interactive` section. The final postage stamp size of the individual images is 73 x 73 pixels in the original image size. For PynPoint, we doubled the resolution, resulting in 146 x 146 pixels for the test data images. The same test data was also used in |Amara_Quanz2|, where we introduced the PynPoint algorithm.


.. |Amara_Quanz2| raw:: html

   <a href="http://www.sciencedirect.com/science/article/pii/S2213133715000049" target="_blank">Amara, A., Quanz, S. P. and Akeret J., Astronomy and Computing vol. 10 (2015)</a>

.. |ESO_Archive| raw:: html

   <a href="http://archive.eso.org/cms/eso-data.html" target="_blank"> European Southern Observatory (ESO) archive </a>

.. _dataaccess:

Looking inside HDF5 files
-------------------------

In order to access data from the HDF5 PynPoint database you have three options:

	* Use the FitsWritingModule to export data to a FITS file, as done in the :ref:`interactive` section.
	* Use the easy access functions of the :class:`PynPoint.Core.Pypeline` class to retrieve data and attributes from the database:

		* pipeline.get_data(...)

		* pipeline.get_attribute(..., ...)

	* Use an external tool such as |HDFCompass| or |HDFView| to read, inspect, and visualize data and attributes in the HDF5 database. We recommend using HDFCompass because it is easy to use and has a basic plotting functionality allowing the user to quickly inspect images from a particular database tags. In HDFCompass, the static attributes can be opened with the 'Reopen as HDF5 Attributes' option.

.. |HDFCompass| raw:: html

   <a href="https://support.hdfgroup.org/projects/compass/download.html" target="_blank">HDFCompass</a>

.. |HDFView| raw:: html

   <a href="https://support.hdfgroup.org/downloads/index.html" target="_blank">HDFView</a>
