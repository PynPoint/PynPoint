.. _pipeline-architecture:

The PynPoint Architecture
=========================

In version 0.3.0 we have redesigned the whole PynPoint architecture in order to make it an end to end data processing tool for ADI data instead of a PSF-subtraction toolkit. Our goal was to create a pipeline which inheres a list of different pipeline-modules, one for each processing step. Moreover we designed the new pipeline to be extendable for new data processing techniques or data types in future. A list of all currently available Pipeline-modules is given in the :ref:`pipeline-modules` section.

The actual pipeline is located in a different sub-package then the functionality of the processing steps. This makes it possible to extend the functionality of the pipeline steps without changing the core of the pipeline.

The image below illustrates the structure of the central pipeline architecture using a UML class diagram:

.. image:: images/PynPoint-UML.png

As you can see the architecture is separated in three different components:

	* A data management component
	* Pipeline modules for reading, writing and processing
	* The actual pipeline

Central Data Storage
--------------------

One central idea of the new PynPoint Pipeline is to separate the data and the pipeline steps. This has different reasons:

	1. Some raw dataset are very large which makes it hard to work with them on a computer with small memory (RAM). Therefore we decided to use a central storage on the hard drive.
	2. Some data is used in different steps of the pipeline. A central database makes it easy to access that data without making a copy.
	3. The central storage on the hard drive will remain updated after each step. If the program crashes or is interrupted by the user you do not have to run the already finished steps again.

Usually if you just use the pipeline you do not have to worry about the central data storage classes. This is only important if you plan to write your own Pipeline modules (See :ref:`own_module`). But it is important to understand how **tags** work.

You might already have noticed in the :ref:`interactive` section that each pipeline module has some input and output tags. A tag is like a key for a specific dataset in the central database. A module with an image_in_tag = `im_arr` is looking for image data under the tag `im_arr` in the central database as input. The same way around a module with an image_out_tag = `im_arr_processed` will write out its result to the central database under the tag `im_arr_processed`. Note in_tags will never change the data in the database.

Under the surface of a pipeline module this access of data from the central database is implemented using Ports.

.. _pipeline-modules:

Pipeline modules
----------------

A pipeline module is like a task which can be added to the pipeline internal task queue. This task can read and write specific data from and to the pipeline database. You can think about a pipeline module as a block with input and output connections to that central database. For an illustration have a look at the PSF-subtraction module below:

.. image:: images/Pipeline_module.pdf

On the left you can see that the PSF-subtraction module needs two input tags which means it has two internal input ports to the central database. The first port imports the data which will be processed. The second port imports reference data which is used to calculate the PSF model using PCA. 

In the middle all module parameters are listed (e.g. the number of PCA components used for the PSF-fit).

On the right a list of all output tags (internal Output ports) which store the results of the PSF-subtraction to the internal database.

In order to create a valid pipeline you should check that the required input tags are linked to data which was created by a previous pipeline module. In other words there need to be a previous module with the same tag as output.

There are three different types of Pipeline modules:
	1 :class:`PynPoint.core.Processing.ReadingModule` - A module only with output tags / ports. The perfect interface to read raw data.

	1.2 :class:`PynPoint.core.Processing.ProcessingModule` - A module with input and output tags / ports. The typical processing step module.

	1.3 :class:`PynPoint.core.Processing.WritingModule` - A module only with input tags / ports which can be used to export data from the internal database.

If you just use pipeline modules the differences between these three module types are not important for you. However, if you are interested in writing own modules you should keep this in mind.

The Pipeline
------------

The :class:`PynPoint.core.Pypeline` module is the central component which manages the order and execution of the different pipeline processing steps. From a simple perspective it is just a ordered list of different pipeline modules. Each Pypeline instance has a input directory which is used as the default input location for reading modules, a working directory where the central pipeline database will be stored and a default output directory which can be used by all writing modules. 

At the moment there is one Pypeline method which can be used to append a pipeline module to the queue of modules: ::

    pipeline.add_module(pipeline_module)

And one method to remove modules: ::

    pipeline.remove_module(name)

If you what to check the names and order of the added pipeline modules use: ::

    pipeline.get_module_names()

Finally you can run all modules by calling: ::

    pipeline.run()

Or run a single module using: ::

    pipeline.run_module(name)

Both run methods will check if the pipeline has valid input and output tags.

A Pypeline instance can be used to directly access data from the central database. See section :ref:`dataaccess` for more information.
