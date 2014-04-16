============
Installation
============

Through pip
-----------

At the command line either via easy_install or pip::

    $ pip install PynPoint --user


Initial test
------------

We have provided some useable data as part of the package. To perform your first calculations do ::

	import PynPoint
	from PynPoint import pyplot
	
	data_dir = PynPoint.get_data_dir() #internal location of data directory
	test_file = '/Data_Shirley_L_Band_PynPoint_conv_stck_500_.hdf5' 

	images = PynPoint.images.create_whdf5input(data_dir+test_file)
	basis = PynPoint.basis.create_whdf5input(data_dir+test_file)
	res = PynPoint.residuals.create_winstances(images, basis)
	
	pynplot.plt_res(res,2,imtype='mean')
	
That's it! If this worked, you should have a picture like the one below and you are ready to go. If you use PynPoint for your exciting dicovery please rememeber to site the two PynPoint paper that describe the method and the package: 

Amara, A. & Quanz, S. P., MNRAS vol. 427 (2012) and 

Amara, A., Quanz, S. P. and Akeret J., Astronomy and Computing (submitted 2014).

**image**

	

Other Options 
-------------
At the command line you can also install using easy_install::

    $ pip install PynPoint --user

Or, if you have virtualenvwrapper installed::

    $ mkvirtualenv PynPoint
    $ pip install PynPoint