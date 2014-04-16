========
Usage
========

PynPoint can be used in a number of ways. Below we outline two methods (i) interactive and (ii) workflow. We also provide test data to help you get started.

PynPoint works through three main classes: :py:class:`images`, :class:`basis` and :class:`residuals`.

* :class:`images` - contains the information to do with the data to be analysed
* :class:`basis` - contains information on the basis set to be used
* :class:`residuals` - manages the modeling, fitting and correction of the psf

Interactive
-----------

Assuming that you start with a directory (`dir_in`) containing a list of fits files, where each fits file contains an image of the star-planet system. To execute this example run::

	$ ipython --pylab

Then, to use PynPoint execute::

	import PynPoint
	images = PynPoint.images.create_wdir(dir_in,stackave=50,cent_size=0.05)
	basis = PynPoint.basis.create_wdir(dir_in,stackave=50,cent_size=0.05)
	res = PynPoint.residuals.create_winstances(images, basis)

The results in res can be viewed using its plot method::

	from PynPoint import pynplot
	pynplot.plt_res(res,5,imtype='mean')
	
In the example above the star is modeled using the first 5 principal component and that the stack is averaged using the mean. 

All of the function above have a number of keywords that can also be passed. More details of these keyword options are discussed in the PynPoint packages section.
	
Workflow
--------

The easiest way to run PynPoint is using the inbuilt :class:`workflow`. This relies on a config file that contains the information about which functions to execute and what options to use. To run a calculation in this way::

	import PynPoint
	ws = PynPoint.run(config_file)
	
As well as returning an instance (ws) containing the run results, data is also stored in the work_space directory defined in the config_file. If this directory exists then you will recieve an error message stating this. If you would like to force the calculation to overwrite an exsiting directory then use the option force_replace=True. For instance::
	
	ws = PynPoint.run(config_file,force_replace=True)
	

The workspace data can be restored later by passing the work_space directory::
	 
	 import PynPoint
	 ws = PynPoint.restore(work_space_dir)

Data can be retrieved from the ws instance using the get method. The available instances in the ws can be listed::

	ws.get_available()
	
To recover the residual instance (as in the interactive example) using the config example below::

	res = ws.get('residuals_module3')
	
This can then be used in the same way as the residual instance earlier.

Shell Script
------------

We have also included a feature so that a config file can be passed to PynPoint and processed through the workflow engine through the command line::

	$ PynPoint <configfilename> True
	
In the above example the keyword 'True' has been set to set force_replace = True. Again, the default value is False, so the if the target output directory already exists, an error message will be returned if force_replace is not True.


Config Example
--------------

In the example config file below, a workspace will be set up called 'workspace_betapic_stk5' will be used to store the results of the a calculation. The config file then calls to run three modules. The first two modules will use data stored in the directory ../data/Data_betapic_L_band/. The options used by these two modules are listed in the section [options1]. ::

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





Data Types
----------

PynPoint currently work with three input data types:

* fits files

* hdfs files

* save/restore files 



The first time that you use fits files as inputs, PynPoint will create an hdf5 of the data inside the same directory as the fits files. This is because the hdf5 file is much faster to read than several thousand small fits files. To use fits inputs, you need to put all the fits files in one directory and then pass this directory to the appropriate PynPoint call. The PynPoint method will then look for all *.fits files in that folder. In 'interactive' mode this can be done by::

	images = PynPoint.images.create_wdir(dir_in)
	
When using the workflow make sure that intype is set to dir in the config file:: 

	intype = dir

HDF5 files, such as those created after you process a directory of fits files, can also be passed directly::

	images = PynPoint.images.create_whdf5input(filename)
	
or for the workflow by setting::

	intype = hdf5
	
The main PynPoint instances also include a save and restore feature. To save the state of an instance::

	images.save(file_to_save_to)
	
Later, an instance can be restored::

	images = PynPoint.images.restore(file_used_by_save)


Data
----

To help you get started quickly and easily we provide access to data. As part of the distribution we provide data that has been stacked by averaging over 500 images at a time. See the install section for instructions on how to process this data. 

The path to the data can be retrieved by running::

	import PynPoint
	print(PynPoint.get_data_dir())

We also make available `the full data <http://www.phys.ethz.ch/~amaraa/Data_Shirley_L_Band_PynPoint_conv.hdf5>`_  (without stacking). 

This is the data that we used to develop PynPoint and is discussed in more detail in our papers. You can also find a short synopsis in the Science section under beta-pic.
