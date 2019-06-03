
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
                     FalsePositiveModule, CropImagesModule, AperturePhotometryModule,\
                     FakePlanetModule

from pynpoint.processing.iterativepsfsubtraction import IterativePcaPsfSubtractionModule

working_place = "/home/Dropbox/Dropbox/1_Philipp/1_UZH/8_FS19/BachelorProject/PynPoint"
input_place = "/home/philipp/Documents/BA_In_out/raw/hd101412/"
output_place = "output/"


'''initial and ending ranks'''
rank_ipca_init = 1

# Python 3
#urllib.request.urlretrieve("https://people.phys.ethz.ch/~stolkert/pynpoint/betapic_naco_mp.hdf5",
#                           os.path.join(input_place, "betapic_naco_mp.hdf5"))

pipeline = Pypeline(working_place_in=working_place,
                    input_place_in=input_place,
                    output_place_in=output_place)
               

module = FitsReadingModule(name_in="read",
                           input_dir=input_place,
                           image_tag="science")

pipeline.add_module(module)



module = CropImagesModule(2,
                 center=None,
                 name_in="crop_image",
                 image_in_tag="science",
                 image_out_tag="science_cropped")
                 
pipeline.add_module(module)



#pipeline.run()
#
#print("shape\n\n\n")
#print(pipeline.get_shape("science"))
#print(pipeline.get_shape("science_cropped"))
#print("\n\n\n")




module = ParangReadingModule(file_name="parang.dat",
                             name_in="parang",
                             input_dir=input_place,
                             data_tag="science_cropped")

pipeline.add_module(module)
#
#
##module = PSFpreparationModule(name_in="prep",
##                              image_in_tag="science",
##                              image_out_tag="prep",
##                              mask_out_tag=None,
##                              norm=False,
##                              resize=None,
##                              cent_size=None,
##                              edge_size=1.1)
##
##pipeline.add_module(module)
#
#
module = PcaPsfSubtractionModule(pca_numbers=(2),
                                 name_in="pca",
                                 images_in_tag="science_cropped",
                                 reference_in_tag="science_cropped",
                                 res_mean_tag="residuals")


pipeline.add_module(module)


#module = IterativePcaPsfSubtractionModule(pca_numbers=(20, ),
#                                 pca_number_init = 15,
#                                 name_in="ipca",
#                                 images_in_tag="science_cropped",
#                                 reference_in_tag="science_cropped",
#                                 res_mean_tag= "residuals",
#                                 subtract_mean = False)
#
#pipeline.add_module(module)


module = AperturePhotometryModule(radius=0.1,
                 position=(49, 52),
                 name_in="aperture_photometry",
                 image_in_tag="residuals",
                 phot_out_tag="photometry")
                 
pipeline.add_module(module)



pipeline.run()

photometry = pipeline.get_data("photometry")
print(photometry)

#print(np.shape(pipeline.get_data("science_cropped")))
#
residuals = pipeline.get_data("residuals")
#fake = pipeline.get_data("im_fake")

pixscale = pipeline.get_attribute("science_cropped", "PIXSCALE")
size = pixscale*residuals.shape[-1]/2.

plt.figure()
plt.imshow(residuals[0, ], origin='lower', extent=[size, -size, -size, size], vmax=None)
plt.title("Challenge Pynpoint - %i PCs"%(15))
plt.xlabel('R.A. offset [arcsec]', fontsize=12)
plt.ylabel('Dec. offset [arcsec]', fontsize=12)
plt.colorbar()
plt.show()
#plt.savefig(os.path.join(output_place, "flux_real.png"), bbox_inches='tight')

#plt.figure()
#plt.imshow(fake[0, ], origin='lower', extent=[size, -size, -size, size], vmax=None)
#plt.title("Challenge Pynpoint - %i PCs"%(15))
#plt.xlabel('R.A. offset [arcsec]', fontsize=12)
#plt.ylabel('Dec. offset [arcsec]', fontsize=12)
#plt.colorbar()
#plt.savefig(os.path.join(output_place, "flux_fake.png"), bbox_inches='tight')
##hdu = fits.PrimaryHDU(data=residuals)
##hdu.writeto(os.path.join(output_place, "residuals_ipca_%i_%i.fits"%(rank_ipca_init, rank_ipca_end)))
