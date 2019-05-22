import os
#import urllib
import numpy as np
#import matplotlib.pyplot as plt
#from astropy.io import fits

#import sys
#sys.path.append('/Users/patapisp/Documents/PhD/Referenceless_PCA/PynPoint/')
#sys.path.append('/Users/patapisp/Documents/PhD/Referenceless_PCA/IPCA/')


from pynpoint import Pypeline, FitsReadingModule,\
                     FalsePositiveModule

working_place = "/home/Dropbox/Dropbox/1_Philipp/1_UZH/8_FS19/BachelorProject/PynPoint"
input_place = "/home/philipp/Documents/BA_In_out/processed/challenge/stack0/fits/sc1"
output_place = "/home/philipp/Documents/BA_In_out/processed/challenge/stack0/snr/sc1"


# Python 3
#urllib.request.urlretrieve("https://people.phys.ethz.ch/~stolkert/pynpoint/betapic_naco_mp.hdf5",
#                           os.path.join(input_place, "betapic_naco_mp.hdf5"))

pipeline = Pypeline(working_place_in=working_place,
                    input_place_in=input_place,
                    output_place_in=output_place)
                    

                    
pca_ipca_end = 80 #maximum pca/ipca value, must be larger than all values in init_list
ipca_init_list = [1, 3, 5, 7, 10, 15] #different inital values
interval = 5 #plot every 'interval' rank
prefix ="challenge"
                    
ipca_dic_inner = {}
ipca_dic_outer = {}

rank_list = []

for counter, i in enumerate(range(ipca_init_list[0], pca_ipca_end+1)):       
        if counter == 0 or ((i % interval) == 0):
            rank_list.append(i)
                
#print(rank_list)

for i in ipca_init_list:
    ipca_dic_inner["{0}".format(i)]= []
    ipca_dic_outer["{0}".format(i)]= []
    for j in rank_list:
        if j > i:
            os.chdir(input_place)
            os.system("mv * ..")
            os.chdir("..")    
            os.system("mv " + prefix + "_ipca_" + str(i) + "_" + str(j) + "_single.fits " + input_place)
            os.chdir(working_place)
    
        
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
            
            #module = ParangReadingModule(file_name="parang.txt",
            #                             name_in="parang",
            #                             input_dir=input_place,
            #                             data_tag="science")
            #
            #pipeline.add_module(module)
            
            
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
            
            
            
            #module = IterativePcaPsfSubtractionModule(pca_numbers=(rank_ipca_end, ),
            #                                 pca_number_init = rank_ipca_init,
            #                                 name_in="ipca",
            #                                 images_in_tag="prep",
            #                                 reference_in_tag="prep",
            #                                 res_mean_tag="residuals")
            #
            #"""
            #
            #module = PcaPsfSubtractionModule(pca_numbers=(5, ),
            #                                 name_in="pca",
            #                                 images_in_tag="prep",
            #                                 reference_in_tag="prep",
            #                                 res_mean_tag="residuals")
            #"""
            #
            #
            #pipeline.add_module(module)
            
            
            module = FalsePositiveModule(position=(28, 36),
                                         aperture=0.05,
                                         ignore=True,
                                         name_in="snr_inner",
                                         image_in_tag="science", #residuals
                                         snr_out_tag="snr_fpf_inner",
                                         optimize=True,
                                         tolerance=0.01)
            
            pipeline.add_module(module)
            
            module = FalsePositiveModule(position=(18, 11),
                                         aperture=0.05,
                                         ignore=True,
                                         name_in="snr_outer",
                                         image_in_tag="science", #residuals
                                         snr_out_tag="snr_fpf_outer",
                                         optimize=True,
                                         tolerance=0.01)
            
            pipeline.add_module(module)
            
            
            pipeline.run()
            residuals = pipeline.get_data("residuals")
            pixscale = pipeline.get_attribute("science", "PIXSCALE")
            snr_inner = pipeline.get_data("snr_fpf_inner")
            snr_outer = pipeline.get_data("snr_fpf_outer")
        #    print("\n\n inner:\n")
        #    print(snr_inner)
        #    print("\n\n outer:\n")
        #    print(snr_outer)
        #    print("\n\n")
        #    np.savetxt(output_place + "/inner.txt", snr_inner)
        #    np.savetxt(output_place + "/outer.txt", snr_outer)
            
            ipca_dic_inner["{0}".format(i)].append(snr_inner[0][4])
            ipca_dic_outer["{0}".format(i)].append(snr_outer[0][4])
            
    np.savetxt(output_place + "/ipca_inner_" + str(i) + ".txt", ipca_dic_inner["{0}".format(i)])
    np.savetxt(output_place + "/ipca_outer_" + str(i) + ".txt", ipca_dic_outer["{0}".format(i)])


pca_inner = []
pca_outer = []

for j in rank_list:
        os.chdir(input_place)
        os.system("mv * ..")
        os.chdir("..")    
        os.system("mv " + prefix + "_pca_" + str(j) + "_single.fits " + input_place)
        os.chdir(working_place)

    
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
        
        #module = ParangReadingModule(file_name="parang.txt",
        #                             name_in="parang",
        #                             input_dir=input_place,
        #                             data_tag="science")
        #
        #pipeline.add_module(module)
        
        
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
        
        
        
        #module = IterativePcaPsfSubtractionModule(pca_numbers=(rank_ipca_end, ),
        #                                 pca_number_init = rank_ipca_init,
        #                                 name_in="ipca",
        #                                 images_in_tag="prep",
        #                                 reference_in_tag="prep",
        #                                 res_mean_tag="residuals")
        #
        #"""
        #
        #module = PcaPsfSubtractionModule(pca_numbers=(5, ),
        #                                 name_in="pca",
        #                                 images_in_tag="prep",
        #                                 reference_in_tag="prep",
        #                                 res_mean_tag="residuals")
        #"""
        #
        #
        #pipeline.add_module(module)
        
        
        module = FalsePositiveModule(position=(28, 36),
                                     aperture=0.05,
                                     ignore=True,
                                     name_in="snr_inner",
                                     image_in_tag="science", #residuals
                                     snr_out_tag="snr_fpf_inner",
                                     optimize=True,
                                     tolerance=0.01)
        
        pipeline.add_module(module)
        
        module = FalsePositiveModule(position=(18, 11),
                                     aperture=0.05,
                                     ignore=True,
                                     name_in="snr_outer",
                                     image_in_tag="science", #residuals
                                     snr_out_tag="snr_fpf_outer",
                                     optimize=True,
                                     tolerance=0.01)
        
        pipeline.add_module(module)
        
        
        pipeline.run()
        residuals = pipeline.get_data("residuals")
        print(np.shape(residuals))
        pixscale = pipeline.get_attribute("science", "PIXSCALE")
        snr_inner = pipeline.get_data("snr_fpf_inner")
        snr_outer = pipeline.get_data("snr_fpf_outer")
    #    print("\n\n inner:\n")
    #    print(snr_inner)
    #    print("\n\n outer:\n")
    #    print(snr_outer)
    #    print("\n\n")
    #    np.savetxt(output_place + "/inner.txt", snr_inner)
    #    np.savetxt(output_place + "/outer.txt", snr_outer)
        
        pca_inner.append(snr_inner[0][4])
        pca_outer.append(snr_outer[0][4])
        
np.savetxt(output_place + "/pca_inner_.txt", pca_inner)
np.savetxt(output_place + "/pca_outer_.txt", pca_outer)

os.chdir(input_place + "/..")
os.system("mv *.fits " + input_place)
os.chdir(working_place)