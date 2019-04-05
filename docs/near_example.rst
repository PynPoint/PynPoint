.. _near_data:

Data Reduction
==============

.. _near_example:

Example
-------

We provide here a short script how to run PynPoint with the data provided here. No bad pixel or other background subtraction modules than the chop and nod subtraction are used. An overview of all the modules available in PynPoint is available |all|.

In the project folder, create 5 different folders::

    $ mkdir working_place input_place input_place1 input_place2 output_place

For the pipeline we require a folder ``working_place``, ``input_place``, ``output_place``. However, in total we have 3 different import fits files (reference PSF, Nod A, Nod B) that we would like to have under a different tag in the database. Therefore we also create a folder ``input_place1`` and ``input_place2``

The simulation data used as input can be downloaded |data|. Put `set1.fits` in ``input_place1``, `set2.fits` in ``input_place2`` and `Ref_PSF_aCenA.fits` into ``input_place``.

The total script is then written in a .py file. You can download it |down_script|. Not all arguments of each function below are explained. Please visit the the full documentation of |pynpoint|, providing more in-depth information. Lastly, if you encounter any errors/mistakes, please visit this |contributions|. We very much welcome active contributions.

Lastly, for an end-to-end processing example of a pupil-stabilized data set of beta Pic in PynPoint, see |stolker|::

    import pynpoint as p

    # Define working place of this project
    working_place = "/path/to/working_place/"
    output_place = "/path/to/output_place/"
    input_place = "/path/to/input_place/"
    input_place1 = "/path/to/input_place1/"
    input_place2 = "/path/to/input_place2/"

    # Create a pipeline instance, which is the main building block of PynPoint
    pipeline = p.Pypeline(working_place_in=working_place,
                          input_place_in=input_place,
                          output_place_in=output_place)

    # Each module requires a name_in tag. This is an identifier for this specific module, allowing
    # you to run it alone, without running the other modules. At the end of the script it is shown
    # how to run each module individually.
    # The images are saved in a central database (in working_place). Each set of images
    # can be called with this tag from this database.
    # At last, each module is added to the pipeline

    # Read the fits file of Nod A.
    inputa = p.FitsReadingModule(name_in="inputa",
                                 input_dir=input_place1,
                                 image_tag="input1",
                                 overwrite=True,
                                 check=True)
    pipeline.add_module(inputa)

    # Read fits of Nod B
    inputb = p.FitsReadingModule(name_in="inputb",
                                 input_dir=input_place2,
                                 image_tag="input2",
                                 overwrite=True,
                                 check=True)
    pipeline.add_module(inputb)

    # Subtract the two input tags, Nod B from Nod A.
    subtract = p.SubtractImagesModule(name_in="subtract",
                                      image_in_tags=("input1", "input2"),
                                      image_out_tag="subtract",
                                      scaling=1.)
    pipeline.add_module(subtract)

    # Crop the image around the center with a size of 5 x 5 arcseconds.
    # The center tag is set to None. This will use the center of the input image.
    crop = p.CropImagesModule(name_in="crop",
                              image_in_tag="subtract",
                              image_out_tag="cropped",
                              size=5.,
                              center=None)
    pipeline.add_module(crop)

    # Write the tag 'cropped' from the central database to a fits file in the ouput_place directory
    write = p.FitsWritingModule(name_in="write",
                                file_name="cropped.fits",
                                output_dir=None,
                                data_tag="cropped",
                                data_range=None,
                                overwrite=True)
    pipeline.add_module(write)

    # Now the required steps in PynPoint are done.
    # Below we show how PCA can be evaluated. At last, we show how to do the contrastcurve.

    # Interpolate POSANG angles between start and end, necessary for the PCA & Contrastcurvemodule
    angle = p.AngleInterpolationModule(name_in="angle_inter",
                                       data_tag="cropped")
    pipeline.add_module(angle)

    # PCA with 10 principal components. I urge you to visit the pynpoint documentation for an
    # explanation for each keyword
    pca = p.PcaPsfSubtractionModule(name_in="pca",
                                    pca_numbers=10,
                                    images_in_tag="cropped",
                                    reference_in_tag="cropped",
                                    res_median_tag="residuals",
                                    res_mean_tag=None,
                                    res_weighted_tag=None,
                                    res_rot_mean_clip_tag=None,
                                    res_arr_out_tag=None,
                                    basis_out_tag=None,
                                    extra_rot=0.,
                                    subtract_mean=False)
    pipeline.add_module(pca)

    # Visually inspect the residuals
    writepca = p.FitsWritingModule(name_in="writepca",
                                   file_name="residuals.fits",
                                   output_dir=None,
                                   data_tag="residuals",
                                   data_range=None,
                                   overwrite=True)
    pipeline.add_module(writepca)


    # CONTRASTCURVE MODULE

    # Prepare psf in tag for contrastcurve module. This is the reference PSF.
    inputpsf = p.FitsReadingModule(name_in="input_psf",
                                   input_dir=input_place,
                                   image_tag="psf_large",
                                   overwrite=True,
                                   check=True)
    pipeline.add_module(inputpsf)

    croppsf = p.CropImagesModule(name_in="croppsf",
                                 image_in_tag="psf_large",
                                 image_out_tag="psf",
                                 center=None,
                                 size=5.)
    pipeline.add_module(croppsf)

    # Run the contrastcurve module, between 0.8 to 3 arcseconds, in steps of 0.1 at every 60 degrees.
    # Below is a fpf of 2.87e-6 used, equivalent to a 5-sigma gaussian confidence interval.
    # The PCA is done within this module.
    contrast = p.ContrastCurveModule(name_in="contrast1",
                                     image_in_tag="cropped",
                                     psf_in_tag="psf",
                                     contrast_out_tag="contrast_out",
                                     separation=(0.8, 3., 0.1),
                                     angle=(0., 360., 60.),
                                     magnitude=(9., 1.),
                                     threshold=("fpf", 2.87e-6),
                                     accuracy=0.1,
                                     psf_scaling=1.,
                                     aperture=0.2,
                                     ignore=True,
                                     pca_number=10,
                                     norm=False,
                                     cent_size=None,
                                     edge_size=None,
                                     extra_rot=0.)
    pipeline.add_module(contrast)

    # Write the result of the contrastcurve to a text file
    # columns: (separation, azimuthally averaged contrast, azimuthal variance of the contrast, false positive fraction)
    write_text = p.TextWritingModule(file_name="contrast",
                                     name_in="contrast_text",
                                     output_dir=None,
                                     data_tag="contrast_out",
                                     header=None)
    pipeline.add_module(write_text)

    # Run the pipeline
    pipeline.run()

    # or run each module individually
    pipeline.run_module("contrast1")

When the script starts running, PynPoint creates a ``PynPoint_config.ini`` file in ``working_place``. In this file, edit ``PIXSCALE`` to 0.045 for VISIR. Also, set ``CPU`` and ``MEMORY`` to a value desired.

.. _near_results:

Results
-------

The output of the PCA should look like this (Open it with DS9 to read the fits file, here we use zscale and color b):

.. image::
   https://lh3.googleusercontent.com/ZTpbnudQq1884kaEc0q5U6D9SZFWoyFXEOunAwNK2i7w5IDEm1uhqymKvnDkRhn3TG2HxnF5b5HL_qfATUAmBnkoK7qJrdpafq2P7xfIvpW5wiN4L2XlFVjqpGk7M2dpnkJ_p3INgtOxPIaPkWE0m6u3l4mmorRymER475h-7x8YFtWj-XRr0F04v6MOi0zQ0klIfzLL0M21EvtGmcjbqkQv5Rpt28x2elq2dMmpAJ-8Wrk_qu_hS3Va2bRDgv8hqOH5s6Ycd1yCKEXukwHiJ3UAa77ZU_NT3XdZX0u-sqOFvKusGVtGvghx-us_PySpNksf6cIjxF1AHxdgPrYCLw3bOCMMCjHWzK2lNHTZAfGZDSx0VTwqgVLBu0hS7-yB_H1jfEGpd_mNJ7pxPkxxWD25jJA0Lr2CeHViEKsmCftpOF41g1Ma2JXggAoL0FGlya-zN-DkJJBJj18G3FWyMxtU8dRsO02pQDdzJ6fKkQ1ekdzi09dj9sZEV2mDsdagACXK1AT34fhNQFvknv5gBRl-beUAhy4G6YMYljfIVuxESmLXRsKLD2b_Fs-Sk9wIR_hnOKmt6Yp4NEPkXUPgYWA-4gJzhZbKecLJCdKUeqdbcCV_y5ywXQAjC6NQLp6lbvlim9YMO0TqpnKO9QPI2yvE1scl6g=w1276-h1139-no?.png
  :width: 70%
  :align: center
  :alt: PCA Image

And plotting the output of the contrastcurve (errorbar is the variance of the different angles, the third column of the text file) using matplotlib will show:

.. image:: https://lh3.googleusercontent.com/wKRxB9hnf5CNG6L5pcPP-cfc7nc0G56nTauctrySP2GpSvg3IAYCrmfNPEItY91qQOGp2ehvWHsO4RRvuE8l7ooMtmiYKY3BCehzSNfiAWcWtNjVJP0QpGdx1OiKdWHZJBb4neGxfoCC7J5gFdH2cugG25GUsS13UitLQkCA6GTzUNZsmlE6LUzDJv4-C5AmuB83UvAYgK8Za4IKct1z2M7x7MsUcRRAy9p6I9NWyiazSz8AZodCkhnP2sVmXDeDEiY0lt6AemXjjiDWRQFqYxsa4eWPlycepefMGKZ5LF2GUnzDv2Ao7SweLnuzmMDNDBtIfugRfvNXu0QphHWgSB_s33n4I-ByqZN0v2zR_zH3NJLKMHqzl5nuyHSe4dunnzfBQjDA8VOSTXljPhlZu6U0zukMC1NjJtzeH_x6DoiaNifvwcQpeVSTqBCYEPUHkdb03nQEFJfZvPApOEo1GYJinS2MrZU0wVQxN8hrJDBQhH-0WLEI6IJAGRWTaO7kkYi2Ybds4r-G9JJmvacPQDftdzHwxAq7XY4lnbyw-g6hJZzX_wSFrVMPvXHt722EwEWkd8nuUc-_KET-t54NZdt9UehPunkNW5VJc7HcrEZXrqvOFGQtI1G7v4xhQzopayn6MwJOM65kqi_ie0T45n-5Dz6k2AuQHqi6LEJt-P3owZFiKv5xnUvSSprLYqEbsSVlcmPRYtE4Y6Jc_xDxpg8=w640-h480-no
  :width: 70%
  :align: center
  :alt: ContrastCurve

The contrast decrease after 1.8 arcseconds is a result of the sidelobes visible in the image (negative psf residue from the subtraction). This becomes more clear in the image below (cropped.fits file), where the green circle has a radius of 1.8 arcseconds.

.. image:: https://lh3.googleusercontent.com/6i9HxIqNP9cQRDmBBzxEHarLNODNr0UaGViTzooNqNYCbzXw03QZ006WrCQRl2VRp9jj49AH2aa_k1Ggkqj3VvQQM7qJDh4hKQDAm9G0DnOtB8sRgX_WB7RVUSoVqla_ZmR5gnsfZSnvT9DEPNYiw-Cf9gcV7wsDtOFP9UzDKTFkBSRPomTlboHkW_mLvQ50Pfi6Yc0uHK3XpEFTkwi4kWqUcYMPiowvd7q6m8-y1AbeUXOvR3f9lDQfo1o6sn4W6ZzSyVb9rr2izx-KXPjdxL3yh_WkDojXXEuugDISeMIxuf9J2ZvtHGvoCFn8AB7bevMsnKAIFI1FKiqxjoNXMJPvRBCuuEl1mHI21brO_lXwLuBIkqHhTXRWUUS9zMFhjIl_iqGdKDJ7As38ZGnvOsJ1z4dytteUG8hoepCPifyX5EZ7prg-mAQ18IgMabgGWqRIqZIj95VARnfhoneoxOBgeQAXBYUX5kqbbGeMBeweVyTweW37R1dO0KHG4z_y808O6hjwZ1ZfDktlrq35U19hi18gtMyvZOi_HzsHQ1KIbomTL3c6OKstXInTff4ONlqvJpV-MGuj5f9vrcrEFv71xEZQth1S9TgltasayljLIxHsR5z_QYbt5MTfBBvpANMhI5aBLHTdn2ouXF6vv9FKN9KwUu_wuNQBqfZMOL0vxB94m2ReTJMiTPHUD4jIpZPK02kQsOjAKNGylxU6QOs=w1273-h1134-no
  :width: 70%
  :align: center
  :alt: PSF

.. |all| raw:: html

   <a href="https://pynpoint.readthedocs.io/en/latest/overview.html" target="_blank">here</a>

.. |down_script| raw:: html

   <a href="https://drive.google.com/open?id=13cJ1a3gGwfI4YR7_wq2AIOP1PoYUSIbn" target="_blank">here</a>

.. |data| raw:: html

   <a href="https://drive.google.com/open?id=1TPSgXjazewwBsBVe-Zu5fstf9X2nlwQX" target="_blank">here</a>

.. |pynpoint| raw:: html

   <a href="https://pynpoint.readthedocs.io/en/latest/" target="_blank">PynPoint</a>

.. |contributions| raw:: html

   <a href="https://pynpoint.readthedocs.io/en/latest/contributing.html#contributing" target="_blank">page</a>

.. |stolker| raw:: html

   <a href="http://adsabs.harvard.edu/abs/2019A%26A...622A.156C" target="_blank">Stolker et al. (2019)</a>
