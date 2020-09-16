import os
import sys
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from pynpoint import Pypeline, \
                     Hdf5ReadingModule, \
                     FitsReadingModule, \
                     FitsWritingModule, \
                     AngleCalculationModule, \
                     PACOModule, \
                     ContrastCurveModule, \
                     ParangReadingModule
class TestPACO:
    def __init__(self,
                 input_dir
                 science_path,
                 psf_path,
                 angles_path,
                 psf_rad,
                 output_dir = "",
                 contrast_curve = False):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.science = science_path
        self.psf = psf_path
        self.angles = angles_path
        self.contrast = contrast_curve
        self.psf_rad = psf_rad
                 
    def run_module(self):
        """
        WARNING
        This isn't an actual test suite for PACO. That still needs to be written.
        This used a known set of data to ensure that PACO could be run, and that
        the outputs were reasonable. It also provides a template for how to set
        up an analysis using PACO.
        """
        working_dir = os.getcwd()
        #input_dir = "/data/ipa/quanz/user_accounts/evertn/PACO/PACO/testData/vip_datasets/"
        #output_dir = "/data/ipa/quanz/user_accounts/evertn/PACO/PACO/output/"
        #input_dir = "/home/evert/Documents/PACO/testData/vip_datasets/"
        #output_dir = "/home/evert/Documents/PACO/output/"

        # Exit here until additional methods are written to generate test data.
        pipeline = Pypeline(working_place_in = working_dir,
                            input_place_in = self.input_dir,
                            output_place_in = self.output_dir)
        
        module = FitsReadingModule(name_in = "read1",
                                   image_tag = "science",
                                   input_dir = self.science)
        pipeline.add_module(module)
        
        module = FitsReadingModule(name_in = "read2",
                                   input_dir = self.psf,
                                   image_tag = 'psf')
        pipeline.add_module(module)
        
        module = ParangReadingModule(name_in = "parang_reading",
                                     data_tag = 'science',
                                     file_name = self.angles,
                                     input_dir = input_dir)
        pipeline.add_module(module)
        
        if self.contrast:
            # For computing a contrast curve for a data set
            module = ContrastCurveModule(name_in = "paco_contrast",
                                         image_in_tag = "science",
                                         psf_in_tag = "psf",
                                         contrast_out_tag = "contrast_out",
                                         angle = (0.,360.,15.),
                                         separation = (0.2,1.0,0.01),
                                         threshold = ('sigma',5.),
                                         aperture = 0.1,
                                         snr_inject = 10.,
                                         extra_rot = 0.,
                                         psf_rad = 0.108,
                                         scaling = 1.0,
                                         algorithm = 'fastpaco',
                                         psf_scaling = 1.0,
                                         verbose = True)
            pipeline.add_module(module)
        else:
            module = PACOModule(name_in = "paco",
                                image_in_tag = 'science',
                                psf_in_tag = 'psf',
                                snr_out_tag = 'paco_snr_fast',
                                flux_out_tag = 'paco_flux_fast',
                                psf_rad = self.psf_rad,
                                scaling = 1.0,
                                flux_calc = True,
                                verbose = True)
            pipeline.add_module(module)
            module = PACOModule(name_in = "paco",
                                image_in_tag = 'science',
                                psf_in_tag = 'psf',
                                snr_out_tag = 'paco_snr_full',
                                flux_out_tag = 'paco_flux_full',
                                psf_rad = self.psf_rad,
                                scaling = 1.0,
                                algorithm = 'fullpaco'
                                flux_calc = True,
                                verbose = True)
            pipeline.add_module(module)
        pipeline.run()
