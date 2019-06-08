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
                     RemoveFramesModule

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



module = RemoveFramesModule(np.arange(0, 410, 1),
                 name_in="remove_frames",
                 image_in_tag="stack",
                 selected_out_tag="stack_short",
                 removed_out_tag="im_arr_removed")
                 
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

for pca_number in pca_numbers:
    module = ContrastCurveModule(name_in="contrast_pca_" + str(pca_number),
                     image_in_tag="stack",
                     psf_in_tag="psf_in",
                     contrast_out_tag="contrast_limits_pca_" + str(pca_number),
                     separation=(0.1, 1.0, 0.01),
                     angle=(0., 360., 60.),
                     threshold=("sigma", 5.),
                     psf_scaling=scale_fac,
                     aperture=0.05,
                     pca_number=pca_number,
                     pca_number_init=None,
                     cent_size=None,
                     edge_size=None,
                     extra_rot=0.,
                     residuals="median",
                     snr_inject=100.)
                 
    pipeline.add_module(module)


for ipca_number_init in ipca_numbers_init:
    for pca_number in pca_numbers:
        module = ContrastCurveModule(name_in="contrast_ipca_" + str(ipca_number_init) + "_" + str(pca_number),
                     image_in_tag="stack",
                     psf_in_tag="psf_in",
                     contrast_out_tag="contrast_limits_ipca_" + str(ipca_number_init) + "_" + str(pca_number),
                     separation=(0.1, 1.0, 0.01),
                     angle=(0., 360., 60.),
                     threshold=("sigma", 5.),
                     psf_scaling=scale_fac,
                     aperture=0.05,
                     pca_number=pca_number,
                     pca_number_init=ipca_number_init,
                     cent_size=None,
                     edge_size=None,
                     extra_rot=0.,
                     residuals="median",
                     snr_inject=100.)
                     
        pipeline.add_module(module)


pipeline.run()




plt.rc("axes", prop_cycle=(cycler("color", ["k", "r", "tab:orange", "y", "g", "c", "b", "m"])))



pca = {}

for pca_number in pca_numbers:
    pca[pca_number] = [[], [], [], []]
    for item in (pipeline.get_data("contrast_limits_pca_" + str(pca_number))):
        for counter, subitem in enumerate(item):
            pca[pca_number][counter].append(subitem)
    plt.plot(pca[pca_number][0], pca[pca_number][1], label="PCA " + str(pca_number))

#print(pca)


ipca = {}

for ipca_number_init in ipca_numbers_init:
    #omit pca_numbers that are smaller than pca_number_init
    if ipca_number_init >= min(pca_numbers):
        pca_numbers_new = []
        for pca_num in pca_numbers:
            if ipca_number_init < pca_num:
                pca_numbers_new.append(pca_num)
    else:
        pca_numbers_new = copy(pca_numbers)        
    for pca_number in pca_numbers_new:
        ipca[str(ipca_number_init) + "_" + str(pca_number)] = [[], [], [], []]
        for item in (pipeline.get_data("contrast_limits_ipca_" + str(ipca_number_init) + "_" + str(pca_number))):
            for counter, subitem in enumerate(item):
                ipca[str(ipca_number_init) + "_" + str(pca_number)][counter].append(subitem) 

        plt.plot(ipca[str(ipca_number_init) + "_" + str(pca_number)][0], ipca[str(ipca_number_init) + "_" + str(pca_number)][1], label="IPCA " + str(ipca_number_init) + "_" + str(pca_number))

#print(ipca)


plt.title("Contrast Limits")
plt.xlabel("Separation [arcsec]")
plt.ylabel("5$\sigma$ Contrast")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_place, "cc_final.pdf"), bbox_inches='tight')













#pixscale = pipeline.get_attribute("science", "PIXSCALE")
#size = pixscale*residuals.shape[-1]/2.

#plt.figure()
#plt.plot(ipca["1_2"][0],ipca["1_2"][1])
#plt.title("Challenge Pynpoint - %i PCs"%(15))
#plt.xlabel('R.A. offset [arcsec]', fontsize=12)
#plt.ylabel('Dec. offset [arcsec]', fontsize=12)
#plt.savefig(os.path.join(output_place, "cc_test.pdf"), bbox_inches='tight')
#hdu = fits.PrimaryHDU(data=residuals)
#hdu.writeto(os.path.join(output_place, "residuals_ipca_%i_%i.fits"%(rank_ipca_init, rank_ipca_end)))
