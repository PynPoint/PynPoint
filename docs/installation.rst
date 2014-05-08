============
Installation
============

Through pip
-----------

At the command line you can install using pip::

    $ pip install PynPoint --user
	
[For background information on pip see our `installation help <https://wiki.phys.ethz.ch/PynPoint/installation_help>`_ pages]


Initial test
------------

We have provided some useable data as part of the package. To perform your first calculations do ::

	import PynPoint
	from PynPoint import PynPlot
	
	data_dir = PynPoint.get_data_dir() #internal location of data directory
	test_file = '/Data_betapic_L_Band_PynPoint_conv_stck_500_.hdf5' 

	images = PynPoint.images.create_whdf5input(data_dir+test_file,cent_size=0.07)
	basis = PynPoint.basis.create_whdf5input(data_dir+test_file,cent_size=0.07)
	res = PynPoint.residuals.create_winstances(images, basis)
	
	PynPlot.plt_res(res,6,imtype='mean')
	
That's it! If this worked, you should have a picture like the one below. What you see in the image is the planet `beta-pic b <http://en.wikipedia.org/wiki/Beta_Pictoris>`_ ! 



You are now ready to go. As you use PynPoint for your exciting discoveries, **please cite the two PynPoint papers** that describe the method and the package: 

`Amara, A. & Quanz, S. P., MNRAS vol. 427 (2012) <http://adsabs.harvard.edu/abs/2012MNRAS.427..948A>`_ and 

Amara, A., Quanz, S. P. and Akeret J., Astronomy and Computing (submitted 2014)


Initial result
--------------

If you run the example above you should see this:

.. image:: install_example.*
	
The image shows the final results at the end of the PynPoint analysis. The star of the planetary system sits at the center of the image, which is masked here. The prominent red blob to the top-right of center is the planet beta-pic b. The pixel scale for the image is 0.0135'' (half of the original data) so the total image is 2''x2''. We see that beta-pic b is roughly 22 pixels from the star (image center) which corresponds to roughly 0.3''. 

When making this results all the images have been aligned to the parallactic angle of the first image. In this particular case this means that North is to the left. The pynplot.plt_res function is able to produce an image that is rotated by a user specified angle (for example to make North point up). For more discussion on the options available see usage, package notes and example tutorials. *** ADD LINKS *** We have also make available the `the full data <http://www.phys.ethz.ch/~amaraa/Data_betapic_L_Band_PynPoint_conv.hdf5>`_.

