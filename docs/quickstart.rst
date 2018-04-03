.. _quickstart:

Quickstart
==========

.. _installation:

Installation
------------

Installation of PynPoint is achieved from the command line with the |pip| (**not possible yet**): ::

    $ pip install --user PynPoint

Alternatively, the PynPoint repository can be cloned from Github, which will contain the most recent implementations: ::

    $ git clone git@github.com:PynPoint/PynPoint.git

Or the repository can be downloaded from the |Github| as a zip file. If needed, the required Python packages can be installed from the PynPoint folder with: ::

    $ pip install -r requirements.txt

.. |pip| raw:: html

   <a href="https://packaging.python.org/tutorials/installing-packages/" target="_blank">pip package manager</a>

.. |github| raw:: html

   <a href="https://github.com/PynPoint/PynPoint" target="_blank">Github website</a>

.. _running:

Running PynPoint
----------------

As a quick start example, we provide a preprocessed data cube of beta Pic in the M' band (4.8 μm). This archival data set was obtained with the high-resolution, adaptive optics assisted, near-infrared camera at the Very Large Telescope under the ESO program ID |id|. The exposure time of the individual images was 65 ms and the total field rotation about 50 deg.

Each image in the data cube has been obtained with a pre-stacking of every 200 images. The data is stored in an HDF5 database (see :ref:`hdf5-files`) which contains a stack of 263 images of 80x80 in size, the parallactic angles, and the pixel scale of the detector. The following script downloads the data (13 MB), runs the PSF subtraction with PynPoint, and plots an image of the mean residuals (make sure to adjust the path of ``working_path``, ``input_path``, and ``output_path``): ::

	import urllib
	import numpy as np
	import matplotlib.pyplot as plt

	import PynPoint

	from PynPoint import Pypeline
	from PynPoint.IOmodules.Hdf5Reading import Hdf5ReadingModule
	from PynPoint.ProcessingModules import PSFSubtractionModule

	working_path = "/path/to/working_place/"
	input_path = "/path/to/input_place/"
	output_path = "/path/to/output_place/"

	url = urllib.URLopener()
	url.retrieve("https://people.phys.ethz.ch/~stolkert/BetaPic_NACO_Mp.hdf5",
		     input_path+"BetaPic_NACO_Mp.hdf5")

	pipeline = Pypeline(working_place_in=working_path,
	                    input_place_in=input_path,
	                    output_place_in=output_path)

	read = Hdf5ReadingModule(name_in="read",
                                 input_filename="BetaPic_NACO_Mp.hdf5",
                                 input_dir=None,
                                 tag_dictionary={"stack":"stack"})

	pipeline.add_module(read)

	pca = PSFSubtractionModule(pca_number=20,
                                   svd="arpack",
                                   name_in="pca",
                                   images_in_tag="stack",
                                   reference_in_tag="stack",
                                   res_mean_tag="residuals",
                                   norm=False,
                                   cent_size=0.15,
                                   edge_size=1.1)

	pipeline.add_module(pca)

	pipeline.run()

	residuals = pipeline.get_data("residuals")
	pixscale = pipeline.get_attribute("stack", "PIXSCALE")

	size = pixscale*np.size(residuals, 0)/2.

	plt.imshow(residuals, origin='lower', extent=[size, -size, -size, size])
	plt.title("beta Pic b - NACO M' - mean residuals")
	plt.xlabel('R.A. offset [arcsec]', fontsize=12)
	plt.ylabel('Dec. offset [arcsec]', fontsize=12)
	plt.colorbar()
	plt.savefig(output_path+"residuals.png")

.. |id| raw:: html

   <a href="http://archive.eso.org/wdb/wdb/eso/sched_rep_arc/query?progid=090.C-0653(D)" target="_blank">090.C-0653(D)</a>

.. _detection:

Exoplanet Detection
-------------------

That's it! The mean residuals of the PSF subtraction are stored in the central database and an image of the residuals has been saved in the ``output_place_in`` folder. The image shows the direct detection of the exoplanet |beta_pic_b|:

.. |beta_pic_b| raw:: html

   <a href="http://en.wikipedia.org/wiki/Beta_Pictoris_b" target="_blank">beta Pic b</a>

.. image:: _images/residuals.png
   :width: 70%
   :align: center

The star of this planetary system is located in the the center of the image, which is masked here, and the orientation of the image is such that North is up and East is left. The bright yellow feature in the bottom right direction is the planet beta Pic b. The angular separation from the central star is 457 mas and the brightness contrast is 7.65 mag. This means that beta Pic b is a factor 1148 fainter than the central star.
