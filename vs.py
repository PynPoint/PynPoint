import os
import urllib
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

import sys
#sys.path.append('/Users/patapisp/Documents/PhD/Referenceless_PCA/PynPoint/')
#sys.path.append('/Users/patapisp/Documents/PhD/Referenceless_PCA/IPCA/')

#from ipca import IPCA

from pynpoint import Pypeline, Hdf5ReadingModule, FitsReadingModule,\
                     PSFpreparationModule, ParangReadingModule,\
                     PcaPsfSubtractionModule,\
                     FalsePositiveModule, CropImagesModule, FitsWritingModule

from pynpoint.processing.iterativepsfsubtraction import IterativePcaPsfSubtractionModule

working_place = "/home/Dropbox/Dropbox/1_Philipp/1_UZH/8_FS19/BachelorProject/PynPoint"
input_place = "/home/philipp/Documents/BA_In_out/raw/hd101412/"
output_place = "output/"


'''initial and ending ranks'''
rank_ipca_init = 15
rank_ipca_end = 30

# Python 3
#urllib.request.urlretrieve("https://people.phys.ethz.ch/~stolkert/pynpoint/betapic_naco_mp.hdf5",
#                           os.path.join(input_place, "betapic_naco_mp.hdf5"))

pipeline = Pypeline(working_place_in=working_place,
                    input_place_in=input_place,
                    output_place_in=output_place)

"""
module = Hdf5ReadingModule(name_in="read",
                           input_filename="betapic_naco_mp.hdf5",
                           input_dir=None,
                           tag_dictionary={"stack":"stack"})
"""

module = FitsReadingModule(name_in="read",
                           input_dir=input_place,
                           image_tag="science")

pipeline.add_module(module)

module = ParangReadingModule(file_name="parang.dat",
                             name_in="parang",
                             input_dir=input_place,
                             data_tag="science")

pipeline.add_module(module)


module = CropImagesModule(0.2,
                 center=None,
                 name_in="crop_image",
                 image_in_tag="science",
                 image_out_tag="science_cropped")
                 
pipeline.add_module(module)


#module = PSFpreparationModule(name_in="prep",
#                              image_in_tag="science",
#                              image_out_tag="prep",
#                              mask_out_tag=None,
#                              norm=False,
#                              resize=None,
#                              cent_size=None,
#                              edge_size=1.1)
#
#pipeline.add_module(module)


module = PcaPsfSubtractionModule(pca_numbers=(15, ),
                                 name_in="pca",
                                 images_in_tag="science_cropped",
                                 reference_in_tag="science_cropped",
                                 res_mean_tag="residuals_pca")


pipeline.add_module(module)


module = IterativePcaPsfSubtractionModule(pca_numbers=(25,),
                                 pca_number_init = 15,
                                 name_in="ipca",
                                 images_in_tag="science_cropped",
                                 reference_in_tag="science_cropped",
                                 res_mean_tag= "residuals_ipca",
                                 subtract_mean = False)

pipeline.add_module(module)


#module = FitsWritingModule("out.fits",
#                 name_in='fits_writing',
#                 output_dir=None,
#                 data_tag='residuals',
#                 data_range=None,
#                 overwrite=True)
#                 
#pipeline.add_module(module)


pipeline.run()
residuals_pca = pipeline.get_data("residuals_pca")
residuals_ipca = pipeline.get_data("residuals_ipca")



pixscale = pipeline.get_attribute("science", "PIXSCALE")
size = pixscale*residuals_pca.shape[-1]/2.
font = 8
font_title = 10
font_sup = 13


#plt.suptitle("PCA vs. IPCA", y = 0.8, fontsize=font_title)
plt.subplots_adjust(wspace=0.5)

plt.figure()
plt.imshow(residuals_pca[0, ], origin='lower', extent=[size, -size, -size, size])
plt.title("PCA (15 PC)")
plt.xlabel('R.A. offset [arcsec]')
plt.ylabel('Dec. offset [arcsec]')
#plt.tick_params(labelsize=font)
plt.colorbar()

plt.savefig(os.path.join(output_place, "pca.pdf"), bbox_inches='tight')


plt.figure()
plt.imshow(residuals_ipca[0, ], origin='lower', extent=[size, -size, -size, size])
plt.title("IPCA (15 - 25 PC)")
plt.xlabel('R.A. offset [arcsec]')
#plt.ylabel('Dec. offset [arcsec]')
#plt.tick_params(labelsize=font)
plt.colorbar()

plt.savefig(os.path.join(output_place, "ipca.pdf"), bbox_inches='tight')