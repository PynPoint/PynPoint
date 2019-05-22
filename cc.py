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
                     FalsePositiveModule, ContrastCurveModule, DerotateAndStackModule

working_place = "/home/Dropbox/Dropbox/1_Philipp/1_UZH/8_FS19/BachelorProject/PynPoint"
input_place = "/home/philipp/Documents/BA_In_out/raw/tauceti/"
output_place = "output/"


'''initial and ending ranks'''
rank_ipca_init = 5
rank_ipca_end = 8

# Python 3
#urllib.request.urlretrieve("https://people.phys.ethz.ch/~stolkert/pynpoint/betapic_naco_mp.hdf5",
#                           os.path.join(input_place, "betapic_naco_mp.hdf5"))

pipeline = Pypeline(working_place_in=working_place,
                    input_place_in=input_place,
                    output_place_in=output_place)


module = Hdf5ReadingModule(name_in="hdf5_reading",
                           input_filename="tau_ceti_psfsub.h5py",
                           input_dir=input_place,
                           tag_dictionary={"40_eri_lprime_std_averaged_s4":"psf_in", "scale_fac":"scale_fac", "tau_ceti_lprime_im_arr_stacked":"stack"})
                           
                       
pipeline.add_module(module)
#
#pipeline.run_module("hdf5_reading")
#psf_in_stack = np.dstack([1]*pipeline.get_data("psf_in"))
#
#print(pipeline.get_tags())
#
#pipeline.set_attribute("psf_in", None, psf_in_stack)

#module = FitsReadingModule(name_in="read",
#                           input_dir=input_place,
#                           image_tag="science")
#
#pipeline.add_module(module)

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


module = ContrastCurveModule(name_in="contrast",
                 image_in_tag="stack",
                 psf_in_tag="psf_in",
                 contrast_out_tag="contrast_limits",
                 separation=(0.1, 1, 0.5),
                 angle=(0., 360., 90.),
                 threshold=("sigma", 5.),
                 psf_scaling=1,
                 aperture=0.05,
                 pca_number=20,
                 cent_size=None,
                 edge_size=None,
                 extra_rot=0.,
                 residuals="median",
                 snr_inject=100.)
                 
pipeline.add_module(module)

#
#module = PcaPsfSubtractionModule(pca_numbers=(10, ),
#                                 name_in="pca",
#                                 images_in_tag="science",
#                                 reference_in_tag="science",
#                                 res_mean_tag="residuals")
#
#
#pipeline.add_module(module)


#module = IterativePcaPsfSubtractionModule(pca_numbers=(rank_ipca_end, ),
#                                 pca_number_init = rank_ipca_init,
#                                 name_in="ipca",
#                                 images_in_tag="science",
#                                 reference_in_tag="science",
#                                 res_mean_tag= "residuals",
#                                 subtract_mean = False)
#
#pipeline.add_module(module)


pipeline.run()
#residuals = pipeline.get_data("residuals")

#dic = pipeline.get_data("hdf5_reading")
#print(dic)

contrast = pipeline.get_data("contrast_limits")
print(contrast)

#pixscale = pipeline.get_attribute("science", "PIXSCALE")
#size = pixscale*residuals.shape[-1]/2.

# plt.figure()
# plt.imshow(residuals[0, ], origin='lower', extent=[size, -size, -size, size], vmax=None)
# plt.title("Challenge Pynpoint - %i PCs"%(15))
# plt.xlabel('R.A. offset [arcsec]', fontsize=12)
# plt.ylabel('Dec. offset [arcsec]', fontsize=12)
# plt.colorbar()
# plt.savefig(os.path.join(output_place, "tauceti.png"), bbox_inches='tight')
#hdu = fits.PrimaryHDU(data=residuals)
#hdu.writeto(os.path.join(output_place, "residuals_ipca_%i_%i.fits"%(rank_ipca_init, rank_ipca_end)))
