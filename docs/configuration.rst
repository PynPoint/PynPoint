.. _configuration:

Configuration
=============

.. _config_intro:

Introduction
------------

A configuration file has to be stored in the ``working_place_in`` with the name ``PynPoint_config.ini``. The file will be created with default values in case it does not exist when the pipeline is initiated. The values of the configuration file are stored in a separate group of the central database, each time the pipeline is initiated.

.. _config_file:

Config File
-----------

The file contains two different sections of configuration parameters. The ``header`` section is used to link attributes in PynPoint with header values in the FITS files that will be imported into the database. For example, some of the pipeline modules require values for the dithering position. These attributes are stored as ``DITHER_X`` and ``DITHER_Y`` in the central database and are for example provided by the ``ESO SEQ CUMOFFSETX`` and ``ESO SEQ CUMOFFSETY`` values in the FITS header. Setting ``DITHER_X: ESO SEQ CUMOFFSETX`` in the ``header`` section of the configuration file makes sure that the relevant FITS header values are imported when :class:`~pynpoint.readwrite.fitsreading.FitsReadingModule` is executed. Therefore, FITS files have to be imported again if values in the ``header`` section are changed. Values can be set to ``None`` since ``header`` values are only required for some of the pipeline modules.

The second section of the configuration values contains the central settings that are used by the pipeline modules. These values are stored in the ``settings`` section of the configuration file. The pixel scale can be provided in arcsec per pixel (e.g. ``PIXSCALE: 0.027``), the number of images that will be simultaneously loaded into the memory (e.g. ``MEMORY: 1000``), and the number of cores that are used for pipeline modules that have multiprocessing capabilities (e.g. ``CPU: 8``) such as :class:`~pynpoint.processing.psfsubtraction.PcaPsfSubtractionModule` and :class:`~pynpoint.processing.fluxposition.MCMCsamplingModule`. A complete overview of the pipeline modules that support multiprocessing is available in the :ref:`overview` section.

Note that some of the pipeline modules provide also multithreading support, which by default runs on all available CPUs. The multithreading can be controlled from the command line by setting the ``OMP_NUM_THREADS`` environment variable::

   $ export OMP_NUM_THREADS=8

In this case a maximum of 8 threads is used. So, if a modules provide both multiprocessing and multithreading support, then the total number of used cores is equal to the product of the values chosen for ``CPU`` in the configuration file and ``OMP_NUM_THREADS`` from the command line.

.. _config_examples:

Examples
--------

In this section, several examples are provided of the configuration content for the following instruments:

- :ref:`config_naco`
- :ref:`config_sphere`
- :ref:`config_visir`

.. _config_naco:

VLT/NACO
^^^^^^^^

.. code-block:: ini

   [header]

   INSTRUMENT: INSTRUME
   NFRAMES: NAXIS3
   EXP_NO: ESO DET EXP NO
   DIT: ESO DET DIT
   NDIT: ESO DET NDIT
   PARANG_START: ESO ADA POSANG
   PARANG_END: ESO ADA POSANG END
   DITHER_X: ESO SEQ CUMOFFSETX
   DITHER_Y: ESO SEQ CUMOFFSETY
   PUPIL: ESO ADA PUPILPOS
   DATE: DATE-OBS
   LATITUDE: ESO TEL GEOLAT
   LONGITUDE: ESO TEL GEOLON
   RA: RA
   DEC: DEC

   [settings]

   PIXSCALE: 0.027
   MEMORY: 1000
   CPU: 1

.. _config_sphere:

VLT/SPHERE
^^^^^^^^^^

.. code-block:: ini

   [header]

   INSTRUMENT: INSTRUME
   NFRAMES: NAXIS3
   EXP_NO: ESO DET EXP ID
   DIT: EXPTIME
   NDIT: ESO DET NDIT
   PARANG_START: ESO TEL PARANG START
   PARANG_END: ESO TEL PARANG END
   DITHER_X: ESO INS1 DITH POSX
   DITHER_Y: ESO INS1 DITH POSY
   PUPIL: None
   DATE: DATE-OBS
   LATITUDE: ESO TEL GEOLAT
   LONGITUDE: ESO TEL GEOLON
   RA: ESO INS4 DROT2 RA
   DEC: ESO INS4 DROT2 DEC

   [settings]

   PIXSCALE: 0.01227
   MEMORY: 1000
   CPU: 1

.. _config_visir:

VLT/VISIR
^^^^^^^^^

.. code-block:: ini

   [header]

   INSTRUMENT: INSTRUME
   NFRAMES: NAXIS3
   EXP_NO: ESO TPL EXPNO
   DIT: ESO DET SEQ1 DIT
   NDIT: ESO DET CHOP NCYCLES
   PARANG_START: ESO ADA POSANG
   PARANG_END: ESO ADA POSANG END
   DITHER_X: None
   DITHER_Y: None
   PUPIL: ESO ADA PUPILPOS
   DATE: DATE-OBS
   LATITUDE: ESO TEL GEOLAT
   LONGITUDE: ESO TEL GEOLON
   RA: RA
   DEC: DEC

   [settings]

   PIXSCALE: 0.045
   MEMORY: 1000
   CPU: 1