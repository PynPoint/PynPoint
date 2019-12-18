import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from pynpoint import Pypeline,AngleCalculationModule, DarkCalibrationModule,\
                     CropImagesModule, SubtractImagesModule, BadPixelSigmaFilterModule,\
                     StarAlignmentModule,ShiftImagesModule, FrameSelectionModule,\
                     FakePlanetModule, FitsReadingModule, MeanCubeModule,\
                     ContrastCurveModule, FitsWritingModule, TextWritingModule,\
                     PACOModule, StarExtractionModule, DerotateAndStackModule,\
                     PACOContrastModule, Hdf5ReadingModule\

fwhm = 4*0.027

data_dir = '/data/ipa/user/evertn/HD131399'
input_dir = data_dir
filename = "PynPoint_database.hdf5"
target = "HD131399"

pipeline = Pypeline(working_place_in = data_dir,
                    input_place_in = data_dir,
                    output_place_in = data_dir,)

#module = Hdf5ReadingModule(name_in = "read2",
#                           input_filename = filename,
#                           input_dir = input_dir,
#			   overwrite = True)
#pipeline.add_module(module)

module = FitsReadingModule(name_in = 'read1',
                           input_dir = data_dir + '/HD131399_DualCoron',
                           image_tag = 'science',
                           overwrite = True,
                           check = True)
pipeline.add_module(module)
pipeline.run_module('read1')

print(pipeline.get_shape('science'))
module = FitsReadingModule(name_in = 'read2',
                           input_dir = data_dir + '/HD131399_Flux',
                           image_tag = 'dark',
                           overwrite = True,
                           check = True)
pipeline.add_module(module)
pipeline.run_module('read2')
print(pipeline.get_shape('dark'))

module = FitsReadingModule(name_in = 'read3',
                           input_dir = data_dir + '/HD131399_Center',
                           image_tag = 'center',
                           overwrite = True,
                           check = True)
pipeline.add_module(module)
pipeline.run_module('read3')
module = AngleCalculationModule(name_in='angle',
                                data_tag='science',
                                instrument='SPHERE/IRDIS')

pipeline.add_module(module)
pipeline.run_module('angle')
module = DarkCalibrationModule(name_in='dark',
                               image_in_tag='science',
                               dark_in_tag='dark',
                               image_out_tag='science_dark')

pipeline.add_module(module)
pipeline.run_module('dark')
module = BadPixelSigmaFilterModule(name_in='bad',
                                  image_in_tag='science_dark',
                                  image_out_tag='science_bad',
                                  map_out_tag=None,
                                  box=9,
                                  sigma=5.,
                                  iterate=3)

pipeline.add_module(module)
pipeline.run_module('bad')

module = FrameSelectionModule(name_in='select',
                              image_in_tag='science_bad',
                              selected_out_tag='science_selected',
                              removed_out_tag='science_removed',
                              index_out_tag=None,
                              method='median',
                              threshold=2.,
                              fwhm=fwhm,
                              aperture=('circular', fwhm),
                              position=(None, None, 4.*fwhm))

pipeline.add_module(module)
pipeline.run_module('select')

module = CropImagesModule(size=2.,
                          center=(1508, 509),
                          name_in='crop1',
                          image_in_tag='science_bad',
                          image_out_tag='science_crop')

pipeline.add_module(module)
pipeline.run_module('crop1')
module = CropImagesModule(size=2.,
                          center=(1508, 509),
                          name_in='crop2',
                          image_in_tag='center',
                          image_out_tag='cent_crop')

pipeline.add_module(module)
pipeline.run_module('crop2')
print(pipeline.get_shape('science_crop'))
module = StarExtractionModule(name_in='extract',
                              image_in_tag='science_crop',
                              image_out_tag='science_extract',
                              index_out_tag=None,
                              image_size=2.,
                              fwhm_star=fwhm,
                              position=(None, None, 4.*fwhm))

pipeline.add_module(module)
pipeline.run_module('extract')
module = StarAlignmentModule(name_in = 'align1',
                             image_in_tag = 'science_extract',
                             ref_image_in_tag = 'cent_crop',
                             image_out_tag = 'science_align',
                             interpolation = 'spline',
                             accuracy = 10.,
                             resize = None,
                             num_references = 4,
                             subframe = 0.8)

pipeline.add_module(module)
pipeline.run_module('align1')
"""
module = FitCenterModule(name_in='center',
                         image_in_tag='science_align',
                         fit_out_tag='fit',
                         mask_out_tag=None,
                         method='mean',
                         radius=5.*fwhm,
                         sign='positive',
                         model='gaussian',
                         filter_size=None,
                         guess=(0., 0., 1., 1., 100., 0., 0.))

pipeline.add_module(module)
pipeline.run_module('center')

module = ShiftImagesModule(name_in='shift',
                           image_in_tag='science_extract',
                           image_out_tag='science_center',
                           shift_xy='fit',
                           interpolation='spline')

pipeline.add_module(module)
pipeline.run_module('shift')
"""
image = pipeline.get_data('science_align')
print(image.shape)
fig,ax = plt.subplots(figsize = (8,8))
ax.imshow(image[0, ], origin='lower')
fig.savefig('image1.png',dpi = 300)
module = DerotateAndStackModule(name_in = 'derotate',
                                image_in_tag = 'science_align',
                                image_out_tag = 'psf',
                                derotate = False,
                                stack = 'median')
pipeline.add_module(module)
pipeline.run_module('derotate')

module = PACOModule(name_in = "paco",
                    image_in_tag = "science_align",
                    snr_out_tag = "paco_snr_2",
                    flux_out_tag = "paco_flux_2",
                    psf_in_tag = 'psf',
                    psf_rad = fwhm,
                    scaling = 1.,
                    algorithm = "fastpaco",
                    flux_calc = False,
                    threshold = 5.0,
                    flux_prec = 0.2,
                    verbose = True)
pipeline.add_module(module)
pipeline.run_module('paco')

module = PACOContrastModule(name_in = "paco_contrast",
                            image_in_tag = "science_align",
                            psf_in_tag = "psf",
                            contrast_out_tag = "contrast_out_2",
                            angle = (0.,360.,15.),
                            separation = (0.2,1.2,0.01),
                            threshold = ('sigma',5.),
                            aperture = 0.1,
                            snr_inject = 100.,
                            extra_rot = 0.,
                            psf_rad = fwhm,
                            scaling = 1.0,
                            algorithm = 'fastpaco',
                            verbose = False)
#pipeline.add_module(module)
#pipeline.run_module('paco_contrast')

module = FitsWritingModule(name_in='write1',
                           file_name='HD131399_pacosnr.fits',
                           output_dir=data_dir,
                           data_tag='paco_snr_2',
                           data_range=None)

pipeline.add_module(module)
pipeline.run_module('write1')
module = FitsWritingModule(name_in='write2',
                           file_name='HD131399_pacoflux.fits',
                           output_dir=data_dir,
                           data_tag='paco_flux_2',
                           data_range=None)

pipeline.add_module(module)
pipeline.run_module('write2')
