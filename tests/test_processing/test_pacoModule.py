import os,sys
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
    def run_module(self):
        """
        WARNING + FIXME
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
        filename = "naco_betapic"
        fits_filename = filename + "_cube.fits"#"naco_betapic_cube.fits"
        psf_filename =  filename + "_psf.fits"#"naco_betapic_psf.fits"
        par_filename =  filename + "_pa.dat"#"naco_betapic_pa.dat"

        # FIXME FIXME FIXME
        # Exit here until additional methods are written to generate test data.
        sys.exit(1)
        
        pipeline = Pypeline(working_place_in = working_dir,
                            input_place_in = input_dir,
                            output_place_in = output_dir)
        
        module = FitsReadingModule(name_in = "read1",
                                   image_tag = "science",
                                   input_dir = input_dir + "bpic_data/")
        pipeline.add_module(module)
        
        module = FitsReadingModule(name_in = "read2",
                                   input_dir = input_dir + "gausspsf/",
                                   image_tag = 'psf')
        pipeline.add_module(module)
        
        module = ParangReadingModule(name_in = "parang_reading",
                                     data_tag = 'science',
                                     file_name = par_filename,
                                     input_dir = input_dir)
        pipeline.add_module(module)
        
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
        pipeline.run()
