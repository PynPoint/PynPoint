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
                     IterativePcaPsfSubtractionModule, PcaPsfSubtractionModule,\
                     FalsePositiveModule

working_place = "/home/Dropbox/Dropbox/1_Philipp/1_UZH/8_FS19/BachelorProject/PynPoint"
input_place = "/home/philipp/Documents/BA_In_out/raw/tauceti/"
output_place = "output/"


'''initial and ending ranks'''
rank_ipca_init = 5
rank_ipca_end = 15

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


#module = PcaPsfSubtractionModule(pca_numbers=(5, 10, 15),
#                                 name_in="pca",
#                                 images_in_tag="science",
#                                 reference_in_tag="science",
#                                 res_mean_tag="residuals")
#
#
#pipeline.add_module(module)


module = IterativePcaPsfSubtractionModule(pca_numbers=(rank_ipca_end, ),
                                 pca_number_init = rank_ipca_init,
                                 name_in="ipca",
                                 images_in_tag="science",
                                 reference_in_tag="science",
                                 res_mean_tag= "residuals",
                                 subtract_mean = False)

pipeline.add_module(module)


pipeline.run()
residuals = pipeline.get_data("residuals")

pixscale = pipeline.get_attribute("science", "PIXSCALE")
size = pixscale*residuals.shape[-1]/2.

plt.figure()
plt.imshow(residuals[0, ], origin='lower', extent=[size, -size, -size, size], vmax=None)
plt.title("Challenge Pynpoint - %i PCs"%(15))
plt.xlabel('R.A. offset [arcsec]', fontsize=12)
plt.ylabel('Dec. offset [arcsec]', fontsize=12)
plt.colorbar()
plt.savefig(os.path.join(output_place, "pynpoint.png"), bbox_inches='tight')
#hdu = fits.PrimaryHDU(data=residuals)
#hdu.writeto(os.path.join(output_place, "residuals_ipca_%i_%i.fits"%(rank_ipca_init, rank_ipca_end)))


#
#
#'''original IPCA'''
#
#'''declare input and output paths'''
#input_path = input_place
#output_path = output_place
#
#
#
#
#
#'''get data'''
#images = fits.open(input_path + "images.fits")
##images.info()
#data_cube = images[0].data
#
##import angles list
#parangs = []
#with open(input_path + "parang.txt") as file:
#    counter = 0
#    for line in file.readlines():
#        if counter != 0:
#            parangs.append(float(line.replace("\n", "")))
#        counter += 1
#
#
#'''process data'''
#frame_ipca = IPCA(data_cube, rank_ipca_end, rank_ipca_init, np.array(parangs))
#plt.figure()
#plt.imshow(frame_ipca, origin='lower', extent=[size, -size, -size, size], vmax=None)
#plt.title("Challenge iPCA - start:%i, end: %i PCs"%(rank_ipca_init, rank_ipca_end))
#plt.xlabel('R.A. offset [arcsec]', fontsize=12)
#plt.ylabel('Dec. offset [arcsec]', fontsize=12)
#plt.colorbar()
#plt.savefig(os.path.join(output_place, "original.png"), bbox_inches="tight")#"residuals_ipca_%i_%i.png"%(rank_ipca_init, rank_ipca_end)), bbox_inches='tight')
###hdu = fits.PrimaryHDU(data=residuals)
###hdu.writeto(os.path.join(output_place, "residuals_ipca_%i_%i.fits"%(rank_ipca_init, rank_ipca_end)))
#
#
#'''plot difference'''
#plt.figure()
#plt.imshow(residuals[0, ] - frame_ipca, origin='lower', extent=[size, -size, -size, size], vmax=None)
#plt.colorbar()
#plt.savefig(os.path.join(output_place, "difference.png"), bbox_inches="tight")