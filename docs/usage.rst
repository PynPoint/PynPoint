========
Usage
========

PynPoint can be used in a number of ways. Below we outline two methods; (i) interactive; and (ii) workflow. We also provide test data to help you get started.

PynPoint works through three main classes: :py:class:`images`, :class:`basis` and :class:`residuals`.

* :class:`images` - contains the information to do with the data to be analysed.
* :class:`basis` - contains information on the basis set to be used.
* :class:`residuals` - manages the modelling, fitting and correction of the PSF.

Interactive
-----------

To analyse data, in the examples below we assume that this is a directory (`dir_in`) that contains a set of fits files, you first need to enter the python environment: ::

	$ ipython 

Then to use PynPoint, execute: ::

	import PynPoint
	images = PynPoint.images.create_wdir(dir_in,stackave=50,cent_size=0.05)
	basis = PynPoint.basis.create_wdir(dir_in,stackave=50,cent_size=0.05)
	res = PynPoint.residuals.create_winstances(images, basis)

The results in res can be viewed using its plot method: ::

	from PynPoint import PynPlot
	PynPlot.plt_res(res,5,imtype='mean')
	
In the example above, the star is modelled using the first 5 principal components and the stack is averaged using the mean. 

All of the functions above have a number of keywords that can also be passed to them. More details of these keyword options are discussed in the PynPoint package section.
	
Workflow
--------

The easiest way to run PynPoint is using the inbuilt :class:`workflow`. This relies on a config file that contains the information about which functions to execute and what options to use. To run a calculation in this way::

	import PynPoint
	ws = PynPoint.run(config_file)
	
As well as returning an instance (ws) containing the run results, data is also stored in the ``work_space`` directory defined in the ``config_file``. If this directory exists, then you will receive an error message. If you would like to force the calculation to overwrite an existing directory, then use the option ``force_replace=True``. For instance::
	
	ws = PynPoint.run(config_file,force_replace=True)
	

The workspace data can be restored later by passing the work_space directory::
	 
	 import PynPoint
	 ws = PynPoint.restore(work_space_dir)

Data can be retrieved from the ws instance using the get method. The available instances in the ws can be listed::

	ws.get_available()
	
To recover the residual instance (as in the interactive example) using the config example below::

	res = ws.get('residuals_module3')
	
This can then be used in the same way as the earlier residuals instance.

Command line interface
----------------------

We have also included a feature so that a config file can be passed to PynPoint and processed through the workflow engine using the command line::

	$ PynPoint <configfilename> True
	
In the above example, the keyword 'True' has been set to set force_replace = True. Again, the default value is False, so if the target output directory already exists, an error message will be returned if force_replace is not True.


Config example
--------------

In the example config file below, a workspace directory will be created called 'workspace_betapic_stk5' that will be used to store the results of the a calculation. The config file then calls for the running os three modules. The first two modules will use data stored in the directory ../data/Data_betapic_L_band/. The options used by these two modules are listed in the section [options1]. ::

	[workspace]
	workdir = ../data/baselinerun_paper/workspace_betapic_stk5/
	datadir = ../data/

	[module1]
	mod_type = images
	input = Data_betapic_L_Band/
	intype = dir
	options = options1

	[module2]
	mod_type = basis
	input = Data_betapic_L_Band/
	intype = dir
	options = options1

	[module3]
	mod_type = residuals
	intype = instances
	images_input = module1
	basis_input = module2

	[options1]
	cent_remove = True
	cent_size = 0.05
	edge_size = 1.0
	resize = True
	F_final = 2
	recent = False
	ran_sub = False
	para_sort = True
	inner_pix = False
	stackave = 5





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

	images = PynPoint.images.restore("images_savefile.hdf5")


Data
----

To help you get started quickly and easily, we provide access to data. As part of the distribution, we provide data that has been stacked by averaging over 500 images at a time. See the install section for instructions on how to process this data. 

The path to the data can be retrieved by running::

	import PynPoint
	print(PynPoint.get_data_dir())

We also make available `the full data <http://www.phys.ethz.ch/~amaraa/Data_betapic_L_Band_PynPoint_conv.hdf5>`_  (without stacking). This is the data that we used to develop PynPoint and is discussed in more detail in our papers. It consists of the high-contrast imaging data-set used to confirm the existence of a massive exoplanet planet orbiting the nearby A-type star beta Pictoris (Lagrange et al. 2010). 

The data-set was taken on 2009 December 26 at the Very Large Telescope with the high-resolution, adaptive optics assisted, near-infrared camera NACO in the L' filter (central wavelengths 3.8 micron) in Angular Differential Imaging (ADI) mode. It consists of 80 data cubes, each containing 300 individual exposures with an individual exposure time of 0.2 s. The total field rotation of the full data-set amounted to ~44 degrees  on the sky. The raw data are publicly available from the |ESO_Archive| (Program ID: 084.C-0739(A)). 

For the test data, basic data reduction steps (sky subtraction, bad pixel correction and alignment of images) were already done as explained in Quanz et al. (2011). The final postage stamp size of the individual images is 73 x 73 pixels in the original image size. For PynPoint, we doubled the resolution, resulting in 146 x 146 pixels for the test data images. The same test data was also used in |Amara_Quanz|, where we introduced the PynPoint algorithm.

 and 

.. |Amara_Quanz| raw:: html

   <a href="http://adsabs.harvard.edu/abs/2012MNRAS.427..948A" target="_blank">Amara & Quanz (2012)</a>

.. |ESO_Archive| raw:: html

   <a href="http://archive.eso.org/cms/eso-data.html" target="_blank"> European Southern Observatory (ESO) archive </a>


