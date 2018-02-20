============
Installation
============

Through pip
-----------

At the command line you can install PynPoint using pip::

   $ pip install PynPoint-exoplanet --user

The required Python packages can be installed within the PynPoint folder as::

   $ pip install -r requirements.txt

For background information on pip, see our |install_help| pages.

You can also join the PynPoint mailing list by sending an email to pynpoint-join@lists.phys.ethz.ch (you can leave the subject line and body of the email blank).

.. |install_help| raw:: html

   <a href="https://wiki.phys.ethz.ch/PynPoint/installation_help" target="_blank">installation help</a>



Initial test
------------

We have provided some useable data as part of the package. To perform your first calculations do the following (change /some/location/Desktop into a location on your disk): ::

	import PynPoint
	from PynPoint import Pypeline
	from PynPoint.IOmodules.Hdf5Reading import Hdf5ReadingModule
	from PynPoint.ProcessingModules import PSFSubtractionModule
	from matplotlib import pyplot as plt


	pipeline = Pypeline("/some/location/Desktop",
	                    PynPoint.get_data_dir(),
	                    "/some/location/Desktop")

	reading_dict = {"im_arr": "im_arr"}

	reading = Hdf5ReadingModule(name_in="hdf5_reading",
	                            tag_dictionary=reading_dict)

	pipeline.add_module(reading)


	subtraction = PSFSubtractionModule(6,
	                                   name_in="PSF_subtraction",
	                                   images_in_tag="im_arr",
	                                   reference_in_tag="im_arr",
	                                   res_mean_tag="result",
	                                   cent_remove=True,
	                                   cent_size=0.07)

	pipeline.add_module(subtraction)

	pipeline.run()

	result = pipeline.get_data("result")

	plt.imshow(result,
	           origin='lower',
	           interpolation='nearest')
	plt.title("Residual Image: mean")
	plt.colorbar()

	plt.savefig("/some/location/result.png")
	
That is it! If this worked, you should have a picture like the one in the section below. What you see in the image is the planet |beta_pic|. 

.. |beta_pic| raw:: html

   <a href="http://en.wikipedia.org/wiki/Beta_Pictoris_b" target="_blank">beta-pic b</a>


You are now ready to go. As you use PynPoint for your exciting discoveries, **please cite the two PynPoint papers** that describe the method and the package: 

|Amara_Quanz| ; and |Amara_Quanz2|

.. |Amara_Quanz| raw:: html

   <a href="http://adsabs.harvard.edu/abs/2012MNRAS.427..948A" target="_blank">Amara, A. & Quanz, S. P., MNRAS vol. 427 (2012)</a>
   
.. |Amara_Quanz2| raw:: html

   <a href="http://adsabs.harvard.edu/abs/2015A%26C....10..107A" target="_blank">Amara, A., Quanz, S. P. and Akeret J., Astronomy and Computing vol. 10 (2015)</a>



Initial result
--------------

If you run the example above, you should see this:

.. image:: images/install_example.*
	
The image shows the final results at the end of the PynPoint analysis. The star of the planetary system sits at the center of the image, which is masked here. The prominent red blob to the top-right of center is the planet beta-pic b. The pixel scale for the image is 0.0135" (half of the original data), so the total image is 2"x 2". We see that beta-pic b is roughly 22 pixels from the star (image center), corresponding to roughly 0.3". 

When making this result, all the images have been aligned to the parallactic angle of the first image. In this particular case, this means that North is to the left. We have also made available the `the full data <http://www.phys.ethz.ch/~amaraa/Data_betapic_L_Band_PynPoint_conv.hdf5>`_.
