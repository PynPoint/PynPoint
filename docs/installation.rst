============
Installation
============

Through pip
-----------

At the command line you can intall using pip::

    $ pip install PynPoint --user


Initial test
------------

We have provided some useable data as part of the package. To perform your first calculations do ::

	import PynPoint
	from PynPoint import pynplot
	
	data_dir = PynPoint.get_data_dir() #internal location of data directory
	test_file = '/Data_betapic_L_Band_PynPoint_conv_stck_500_.hdf5' 

	images = PynPoint.images.create_whdf5input(data_dir+test_file,cent_size=0.07)
	basis = PynPoint.basis.create_whdf5input(data_dir+test_file,cent_size=0.07)
	res = PynPoint.residuals.create_winstances(images, basis)
	
	pynplot.plt_res(res,6,imtype='mean')
	
That's it! If this worked, you should have a picture like the one below. What you see in the image is the planet `beta-pic b <http://en.wikipedia.org/wiki/Beta_Pictoris>`_ ! 

You are now ready to go. As you use PynPoint for your exciting dicoveries, **please rememeber to cite the two PynPoint papers** that describe the method and the package: 

`Amara, A. & Quanz, S. P., MNRAS vol. 427 (2012) <http://adsabs.harvard.edu/abs/2012MNRAS.427..948A>`_ and 

Amara, A., Quanz, S. P. and Akeret J., Astronomy and Computing (submitted 2014)

**image**

	

Other Options 
-------------
At the command line you can also install using easy_install::

    $ pip install PynPoint --user

Or, if you have virtualenvwrapper installed::

    $ mkvirtualenv PynPoint
    $ pip install PynPoint