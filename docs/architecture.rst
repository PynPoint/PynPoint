.. _architecture:

Architecture
============

PynPoint has evolved from PSF subtraction toolkit to an end-to-end pipeline for high-contrast imaging data obtained in pupil-stabilized mode. The architecture of PynPoint was redesigned in v0.3.0 with the goal to create a generic, modular, and open-source data reduction pipeline, which is extendable to new data processing techniques and data types in the future. An overview of the available IO and processing modules is provide in the :ref:`pynpoint-package` section.

The actual pipeline and processing functionalities are implemented in a different subpackages. Therefore it is possible to extend the processing functionalities of the pipeline without changing the core of the pipeline.

The UML class diagram below illustrates the pipeline architecture of PynPoint:

.. image:: _images/uml.png
   :width: 100%

The diagram shows that the architecture is subdivided in three components:

	* Data management
	* Pipeline modules for reading, writing, and processing of data
	* The actual pipeline

.. _database:

Central Database
----------------

The new architecture of PynPoint separates the data management from the data reduction steps for the following reasons:

	1. Raw datasets can be very large, in particular in the 3--5 μm wavelength regime, which challenges the processing on a computer with a small amount of memory (RAM). A central database is used to store the data on a computer's hard drive.
	2. Some data is used in different steps of the pipeline. A central database makes it easy to access that data without making a copy.
	3. The central data storage on the hard drive will remain updated after each step. Therefore, processing steps that already finished remain unaffected if an error occurs or the data reduction is interrupted by the user.

Understanding the central data storage classes is important if you plan to write your own Pipeline modules (see :ref:`writing`). When running the pipeline, it is enough to understand the concept of database tags.

As already encountered in the :ref:`end-to-end` section, each pipeline module has input and/or output tags. A tag is a label of a specific dataset in the central database. A module with ``image_in_tag=im_arr`` will look for a stack of input images in the central database under the tag name `im_arr`. Similarly, a module with ``image_out_tag=im_arr_processed`` will a stack of processed images to the central database under the tag `im_arr_processed`. Note that input tags will never change the data in the database.

Accessing the data storage occurs through instances of :class:`PynPoint.Core.DataIO.Port` which allow pipeline modules to read data from and write data to central database.

.. _modules:

Central configuration
---------------------

A central configuration file has to be stored in the ``working_place_in`` with the name ``PynPoint_config.ini``. The file will be created with default values in case it does not exist when the pipeline is initiated. The values of the configuration file are stored in a separate group of the central database, each time the pipeline is initiated.

The file contains two different sections of configuration parameters. The ``header`` section is used to link attributes in PynPoint with header values in the FITS files that will be imported into the database. For example, some of the pipeline modules require values for the dithering position. These attributes are stored as ``DITHER_X`` and ``DITHER_Y`` in the central database and are for example provided by the ``ESO SEQ CUMOFFSETX`` and ``ESO SEQ CUMOFFSETY`` values in the FITS header. Setting ``DITHER_X: ESO SEQ CUMOFFSETX`` in the ``header`` section of the configuration file makes sure that the relevant FITS header values are imported when :class:`PynPoint.IOmodules.FitsReading.FitsReadingModule` is executed. Therefore, FITS files have to be imported again if values in the ``header`` section are changes. Values can be set to ``None`` since ``header`` values are only required for some of the pipeline modules.

The second section of the configuration values contains the central settings that are used by the pipeline modules. These values are stored in the ``settings`` section of the configuration file. The pixel scale can be provided in arcsec per pixel (e.g. ``PIXSCALE: 0.027``), the number of images that will be simultaneously loaded into the memory (e.g. ``MEMORY: 1000``), and the number of cores that are used for pipeline modules that have multiprocessing capabilities (e.g. ``CPU: 8``) such as :class:`pynpoint.processing.PSFSubtractionPCA.PcaPsfSubtractionModule`, :class:`pynpoint.processing.FluxAndPosition.MCMCsamplingModule`, and :class:`pynpoint.processing.TimeDenoising.WaveletTimeDenoisingModule`.

Note that some of the pipeline modules provide also multithreading support, which by default runs on all available CPUs. The multithreading can be controlled from the command line by setting the ``OMP_NUM_THREADS`` environment variable::

   $ export OMP_NUM_THREADS=8

In this case a maximum of 8 threads is used. So, if a modules provide both multiprocessing and multithreading support, then the total number of used cores is equal to the product of the values chosen for ``CPU`` in the configuration file and ``OMP_NUM_THREADS`` from the command line.

An complete example of the configuration file looks like::

   [header]

   INSTRUMENT: INSTRUME
   NFRAMES: NAXIS3
   EXP_NO: ESO DET EXP NO
   NDIT: ESO DET NDIT
   PARANG_START: ESO ADA POSANG
   PARANG_END: ESO ADA POSANG END
   DITHER_X: ESO SEQ CUMOFFSETX
   DITHER_Y: ESO SEQ CUMOFFSETY
   DIT: ESO DET DIT
   PUPIL: ESO ADA PUPILPOS
   DATE: DATE-OBS
   LATITUDE: ESO TEL GEOLAT
   LONGITUDE: ESO TEL GEOLON
   RA: RA
   DEC: DEC

   [settings]

   PIXSCALE: 0.027
   MEMORY: 1000
   CPU: 8

Modules
-------

A pipeline module has a specific task that is appended to the internal queue of pipeline tasks. A module can read and write data tags from and to the central database through dedicated input and output connections. As illustration, this is the input and output structure of the :class:`pynpoint.processing.PSFSubtractionPCA.PSFSubtractionModule`:

.. image:: _images/module.jpg
   :width: 70%
   :align: center

The module requires two input tags (blue) which means that two internal input ports are used to access data from the central database. The first port imports the science images and the second port imports the reference images that are used to calculate the PSF model using principle component analysis (PCA). In this case, both input tags can have the same name and therefore point to the same data set. 

The module parameters are listed in the center of the illustration, which includes the number of principle components and the additional derotation that is applied.

The output tags (red) are required to setup the internal output ports which store the results of the PSF subtraction (e.g., mean and variance of the residuals) to the central database.

In order to create a valid pipeline one should check that the required input tags are linked to data which was previously created by a pipeline module. In other words, there need to be a previous module with the same tag as output.

There are three types of pipeline modules:

	1. :class:`pynpoint.core.processing.ReadingModule` - A module with only output tags/ports, used to read data to the central database.
	2. :class:`pynpoint.core.processing.WritingModule` - A module with only input tags/ports, used to export data from the central database.
	3. :class:`pynpoint.core.processing.ProcessingModule` - A module with both input and output tags/ports, used for processing of the data.

.. _pipeline:

Pipeline
--------

The :class:`pynpoint.core.pypeline` module is the central component which manages the order and execution of the different pipeline modules. Each ``Pypeline`` instance has an ``working_place_in`` path which is where the central database and configuration file are stored, an ``input_place_in`` path which is the default data location for reading modules, and an ``output_place_in`` path which is the default output path where the data will be saved by the writing modules: ::

    pipeline = Pypeline(working_place_in="/path/to/working_place",
                        input_place_in="/path/to/input_place",
                        output_place_in="/path/to/output_place")

A pipeline module is appended to the queue of modules as: ::

    pipeline.add_module("module")

And can be removed from the queue with the following ``Pypeline`` method: ::

    pipeline.remove_module("module")

The names and order of the pipeline modules are listed with: ::

    pipeline.get_module_names()

Running all modules attached to the pipeline is achieved with: ::

    pipeline.run()

Or a single module is executed as: ::

    pipeline.run_module("name")

Both run methods will check if the pipeline has valid input and output tags.

An instance of ``Pypeline`` can be used to directly access data from the central database. See the :ref:`hdf5-files` section for more information.
