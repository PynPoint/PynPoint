import os
#import urllib
import numpy as np
import matplotlib.pyplot as plt
#from astropy.io import fits

#import sys
#sys.path.append('/Users/patapisp/Documents/PhD/Referenceless_PCA/PynPoint/')
#sys.path.append('/Users/patapisp/Documents/PhD/Referenceless_PCA/IPCA/')


from pynpoint import Pypeline, Hdf5ReadingModule, FitsReadingModule,\
                     PSFpreparationModule, ParangReadingModule,\
                     PcaPsfSubtractionModule,\
                     FalsePositiveModule, CropImagesModule, AperturePhotometryModule,\
                     FakePlanetModule

from pynpoint.processing.iterativepsfsubtraction import IterativePcaPsfSubtractionModule

working_place = "/home/Dropbox/Dropbox/1_Philipp/1_UZH/8_FS19/BachelorProject/PynPoint"
input_place = "/home/philipp/Documents/BA_In_out/raw/hd101412"
output_place = "output/"


# Python 3
#urllib.request.urlretrieve("https://people.phys.ethz.ch/~stolkert/pynpoint/betapic_naco_mp.hdf5",
#                           os.path.join(input_place, "betapic_naco_mp.hdf5"))

pipeline = Pypeline(working_place_in=working_place,
                    input_place_in=input_place,
                    output_place_in=output_place)
                    

                    
pca_numbers = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
ipca_numbers_init = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]



#pipeline = Pypeline(working_place_in=working_place,
#                    input_place_in=input_place,
#                    output_place_in=output_place)
#               
#
#module = FitsReadingModule(name_in="read",
#                           input_dir=input_place,
#                           image_tag="science")
#
#pipeline.add_module(module)
#
#
#
#
#module = CropImagesModule(2,
#                 center=None,
#                 name_in="crop_image",
#                 image_in_tag="science",
#                 image_out_tag="science_cropped")
#                 
#pipeline.add_module(module)
#
#
#module = ParangReadingModule(file_name="parang.dat",
#                             name_in="parang",
#                             input_dir=input_place,
#                             data_tag="science_cropped")
#
#pipeline.add_module(module)
#
#
#
#module = PcaPsfSubtractionModule(pca_numbers=pca_numbers,
#                                 name_in="pca",
#                                 images_in_tag="science_cropped",
#                                 reference_in_tag="science_cropped",
#                                 res_mean_tag="residuals_pca")
#
#
#pipeline.add_module(module)
#
#
#module = FalsePositiveModule(position=(49, 52),
#                                         aperture=0.1,
#                                         ignore=True,
#                                         name_in="snr_pca",
#                                         image_in_tag="residuals_pca", #residuals
#                                         snr_out_tag="snr_pca",
#                                         optimize=True,
#                                         tolerance=0.01)
#            
#pipeline.add_module(module)
#        
#        
#
#for ipca_number_init in ipca_numbers_init:
#    module = IterativePcaPsfSubtractionModule(pca_numbers=pca_numbers,
#                                     pca_number_init = ipca_number_init,
#                                     name_in="ipca_" + str(ipca_number_init),
#                                     images_in_tag="science_cropped",
#                                     reference_in_tag="science_cropped",
#                                     res_mean_tag= "residuals_ipca_" + str(ipca_number_init),
#                                     subtract_mean = False)
#    
#    pipeline.add_module(module)
#    
#    module = FalsePositiveModule(position=(49, 52),
#                                         aperture=0.1,
#                                         ignore=True,
#                                         name_in="snr_ipca_" + str(ipca_number_init),
#                                         image_in_tag="residuals_ipca_" + str(ipca_number_init),
#                                         snr_out_tag="snr_ipca_" + str(ipca_number_init),
#                                         optimize=True,
#                                         tolerance=0.01)
#            
#    pipeline.add_module(module)
#    
#    
#pipeline.run()

#create pca snr array
#snr_pca = np.zeros((len(pca_numbers)))
#for i in range(len(pca_numbers)):
#    snr_pca[i] = pipeline.get_data("snr_pca")[i][4]
#
##create ipca snr dictionairy
#snr_ipca = {}
#for ipca_number_init in ipca_numbers_init:
#    
#    snr_ipca[ipca_number_init] = pipeline.get_data("snr_ipca_" + str(ipca_number_init)).tolist()
#    for counter in range(len(snr_ipca[ipca_number_init])):
#        snr_ipca[ipca_number_init][counter] = snr_ipca[ipca_number_init][counter][4]
#    if len(snr_ipca[ipca_number_init]) != len(pca_numbers):
#        snr_ipca[ipca_number_init] = [None] * (len(pca_numbers) - len(snr_ipca[ipca_number_init])) + snr_ipca[ipca_number_init]
#
##create matrix_ipca
#matrix_ipca = np.zeros((len(ipca_numbers_init), len(pca_numbers)))
#
#for counter_i, i in enumerate(ipca_numbers_init):
#    for counter_j, j in enumerate(pca_numbers):
#        matrix_ipca[counter_i][counter_j] = snr_ipca[i][counter_j]
#
##create matrix_all
#snr_all = np.zeros((len(ipca_numbers_init)+1, len(pca_numbers)))
#
#for i in range(len(ipca_numbers_init)+1):
#    for j in range(len(pca_numbers)):
#        if i == 0:
#            snr_all[i][j] = snr_pca[j]
#        else:
#            snr_all[i][j] = matrix_ipca[i-1][j]

snr_all = np.loadtxt(output_place + "snr_all_2_bothfalse.txt")
                        
#np.savetxt(output_place + "snr_all.txt", snr_all)
    
sizex = max(pca_numbers)
sizey = max(ipca_numbers_init)

plt.figure()
plt.imshow(snr_all, origin='lower')
plt.title("Signal-To-Noise Ratio of IPCA and PCA")
plt.xlabel('Number of PC', fontsize=12)
plt.xticks(np.arange(0, len(pca_numbers), 1), pca_numbers)
plt.yticks(np.arange(0, len(ipca_numbers_init)+1, 1), ["PCA"] + ipca_numbers_init)
plt.colorbar()
plt.savefig(os.path.join(output_place, "snr_all_2_bothfalse.pdf"), bbox_inches='tight')