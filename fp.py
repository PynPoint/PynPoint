import os
import urllib
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from copy import copy
from cycler import cycler

import sys
#sys.path.append('/Users/patapisp/Documents/PhD/Referenceless_PCA/PynPoint/')
#sys.path.append('/Users/patapisp/Documents/PhD/Referenceless_PCA/IPCA/')

#from ipca import IPCA

from pynpoint import Pypeline, Hdf5ReadingModule, FitsReadingModule,\
                     PSFpreparationModule, ParangReadingModule,\
                     PcaPsfSubtractionModule,\
                     FalsePositiveModule, ContrastCurveModule, DerotateAndStackModule,\
                     RemoveFramesModule, FakePlanetModule

from pynpoint.processing.iterativepsfsubtraction import IterativePcaPsfSubtractionModule
from pynpoint.processing.basic import RepeatImagesModule



working_place = "/home/Dropbox/Dropbox/1_Philipp/1_UZH/8_FS19/BachelorProject/PynPoint"
input_place = "/home/philipp/Documents/BA_In_out/raw/tauceti"
output_place = "output/"


'''initial and final number of principal components'''
pca_numbers = [1, 5, 10, 15, 20, 25, 30]
ipca_numbers_init = [1, 5, 10, 15, 20, 25]

pca_numbers = [1]
ipca_numbers_init = []


itime_tau_ceti = 0.175
itime_std = 0.3  

mag_tau_ceti = 1.685
mag_std = 7.555

scale_fac = itime_tau_ceti/itime_std
scale_fac *= 10.**(-(mag_tau_ceti-mag_std)/2.5)



# Python 3
#urllib.request.urlretrieve("https://people.phys.ethz.ch/~stolkert/pynpoint/betapic_naco_mp.hdf5",
#                           os.path.join(input_place, "betapic_naco_mp.hdf5"))

pipeline = Pypeline(working_place_in=working_place,
                    input_place_in=input_place,
                    output_place_in=output_place)

module = ParangReadingModule(file_name="parang.dat",
                             name_in="parang",
                             input_dir=input_place,
                             data_tag="stack",
                             overwrite=True)

pipeline.add_module(module)


module = Hdf5ReadingModule(name_in="hdf5_reading",
                           input_filename="tau_ceti_psfsub.h5py",
                           input_dir=input_place,
                           tag_dictionary={"40_eri_lprime_std_averaged_s4":"psf_in", "tau_ceti_lprime_im_arr_stacked":"stack"})
                           
                       
pipeline.add_module(module)


module = RepeatImagesModule(name_in="repeat",
                 image_in_tag="psf_in",
                 image_out_tag="psf_in_rep",
                 repeat=414)
                 
pipeline.add_module(module)                


module = RemoveFramesModule(np.arange(0, 410, 1),
                 name_in="remove_frames",
                 image_in_tag="stack",
                 selected_out_tag="stack_short",
                 removed_out_tag="im_arr_removed")
                 
pipeline.add_module(module)


module = RemoveFramesModule(np.arange(0, 410, 1),
                 name_in="remove_frames2",
                 image_in_tag="psf_in_rep",
                 selected_out_tag="psf_in_short",
                 removed_out_tag="psf_in_removed")
                 
pipeline.add_module(module)




module = FakePlanetModule(name_in="fp", 
                          image_in_tag="stack_short",
                          psf_in_tag="psf_in_short",
                          image_out_tag="fp",
                          position=(100, 100),
                          magnitude=-1,
                          psf_scaling=1.,
                          interpolation='spline')
                          
pipeline.add_module(module)


module = IterativePcaPsfSubtractionModule(pca_numbers=(25,),
                                 pca_number_init = 15,
                                 name_in="ipca",
                                 images_in_tag="fp",
                                 reference_in_tag="fp",
                                 res_mean_tag= "residuals",
                                 subtract_mean = False)

pipeline.add_module(module)



pipeline.run()





residuals = pipeline.get_data("residuals")

pixscale = pipeline.get_attribute("science", "PIXSCALE")
size = pixscale*residuals.shape[-1]/2.

plt.figure()
plt.imshow(residuals[0, ], origin='lower', extent=[size, -size, -size, size])
plt.title("PCA (15 PC)")
plt.xlabel('R.A. offset [arcsec]')
plt.ylabel('Dec. offset [arcsec]')
#plt.tick_params(labelsize=font)
plt.colorbar()

plt.savefig(os.path.join(output_place, "fp.pdf"), bbox_inches='tight')
