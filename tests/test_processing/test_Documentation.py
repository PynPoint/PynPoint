import os
import math
import warnings

import numpy as np

from astropy.io import fits
from scipy.ndimage import shift

from PynPoint.Core.Pypeline import Pypeline
from PynPoint.Core.DataIO import DataStorage
from PynPoint.IOmodules.FitsReading import FitsReadingModule
from PynPoint.IOmodules.FitsWriting import FitsWritingModule
from PynPoint.ProcessingModules.BadPixelCleaning import BadPixelSigmaFilterModule
from PynPoint.ProcessingModules.DarkAndFlatCalibration import DarkCalibrationModule, FlatCalibrationModule
from PynPoint.ProcessingModules.ImageResizing import RemoveLinesModule
from PynPoint.ProcessingModules.PSFpreparation import AngleInterpolationModule
from PynPoint.ProcessingModules.BackgroundSubtraction import MeanBackgroundSubtractionModule
from PynPoint.ProcessingModules.StarAlignment import StarExtractionModule, StarAlignmentModule
from PynPoint.ProcessingModules.PSFSubtractionPCA import PSFSubtractionModule
from PynPoint.ProcessingModules.FrameSelection import RemoveLastFrameModule
from PynPoint.ProcessingModules.StackingAndSubsampling import StackAndSubsetModule
from PynPoint.Util.TestTools import create_config, create_fits, create_fake

warnings.simplefilter("always")

limit = 1e-10

def setup_module():
    test_dir = os.path.dirname(__file__) + "/"

    os.makedirs(test_dir + "adi")
    os.makedirs(test_dir + "dark")
    os.makedirs(test_dir + "flat")

    # SCIENCE

    create_fake(file_start=test_dir+'adi/adi',
                ndit=[22, 17, 21, 18],
                nframes=[23, 18, 22, 19],
                exp_no=[1, 2, 3, 4],
                npix=(100, 102),
                fwhm=3.,
                x0=[25, 75, 75, 25],
                y0=[75, 75, 25, 25],
                angles=[[0., 25.], [25., 50.], [50., 75.], [75., 100.]],
                sep=7.,
                contrast=1e-2)

    # DARK

    ndit = [ 3, 3, 5, 5 ]
    parang = [ [0., 0.], [0., 0.], [0., 0.], [0., 0.] ]

    np.random.seed(2)

    for j, n in enumerate(ndit):
        image = np.random.normal(loc=0, scale=2e-4, size=(n, 100, 100))

        filename = test_dir+'dark/dark'+str(j+1).zfill(2)+'.fits'
        create_fits(filename, image, ndit[j], 0, parang[j], 0., 0.)

    # FLAT

    ndit = [ 3, 3, 5, 5 ]
    parang = [ [0., 0.], [0., 0.], [0., 0.], [0., 0.] ]

    np.random.seed(3)

    for j, n in enumerate(ndit):
        image = np.random.normal(loc=1, scale=1e-2, size=(n, 100, 100))

        filename = test_dir+'flat/flat'+str(j+1).zfill(2)+'.fits'
        create_fits(filename, image, ndit[j], 0, parang[j], 0., 0.)

    filename = os.path.dirname(__file__) + "/PynPoint_config.ini"
    create_config(filename)

def teardown_module():
    test_dir = os.path.dirname(__file__) + "/"

    for i in range(4):
        file_in = test_dir + 'adi/adi'+str(i+1).zfill(2)+'.fits'
        dark_in = test_dir + 'dark/dark'+str(i+1).zfill(2)+'.fits'
        flat_in = test_dir + 'flat/flat'+str(i+1).zfill(2)+'.fits'

        os.remove(file_in)
        os.remove(dark_in)
        os.remove(flat_in)

    os.remove(test_dir + 'PynPoint_database.hdf5')
    os.remove(test_dir + 'test.fits')
    os.remove(test_dir + 'PynPoint_config.ini')

    os.rmdir(test_dir + 'adi')
    os.rmdir(test_dir + 'dark')
    os.rmdir(test_dir + 'flat')

class TestDocumentation(object):

    def setup(self):
        self.test_dir = os.path.dirname(__file__) + "/"
        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def test_docs(self):
        read_science = FitsReadingModule(name_in="read_science",
                                         input_dir=self.test_dir+"adi/",
                                         image_tag="im_arr")

        self.pipeline.add_module(read_science)

        read_dark = FitsReadingModule(name_in="read_dark",
                                      input_dir=self.test_dir+"dark/",
                                      image_tag="dark_arr")

        self.pipeline.add_module(read_dark)

        read_flat = FitsReadingModule(name_in="read_flat",
                                      input_dir=self.test_dir+"flat/",
                                      image_tag="flat_arr")

        self.pipeline.add_module(read_flat)

        remove_last = RemoveLastFrameModule(name_in="last_frame",
                                            image_in_tag="im_arr",
                                            image_out_tag="im_arr_last")

        self.pipeline.add_module(remove_last)

        cutting = RemoveLinesModule(lines=(0, 0, 0, 2),
                                    name_in="cut_lines",
                                    image_in_tag="im_arr_last",
                                    image_out_tag="im_arr_cut")

        self.pipeline.add_module(cutting)

        dark_sub = DarkCalibrationModule(name_in="dark_subtraction",
                                         image_in_tag="im_arr_cut",
                                         dark_in_tag="dark_arr",
                                         image_out_tag="dark_sub_arr")

        flat_sub = FlatCalibrationModule(name_in="flat_subtraction",
                                         image_in_tag="dark_sub_arr",
                                         flat_in_tag="flat_arr",
                                         image_out_tag="flat_sub_arr")

        self.pipeline.add_module(dark_sub)
        self.pipeline.add_module(flat_sub)

        bg_subtraction = MeanBackgroundSubtractionModule(shift=None,
                                                         cubes=1,
                                                         name_in="background_subtraction",
                                                         image_in_tag="flat_sub_arr",
                                                         image_out_tag="bg_cleaned_arr")


        self.pipeline.add_module(bg_subtraction)

        bp_cleaning = BadPixelSigmaFilterModule(name_in="sigma_filtering",
                                                image_in_tag="bg_cleaned_arr",
                                                image_out_tag="bp_cleaned_arr")

        self.pipeline.add_module(bp_cleaning)

        extraction = StarExtractionModule(name_in="star_cutting",
                                          image_in_tag="bp_cleaned_arr",
                                          image_out_tag="im_arr_extract",
                                          image_size=0.6,
                                          fwhm_star=0.1,
                                          position=None)

        # Required for ref_image_in_tag in StarAlignmentModule, otherwise a random frame is used
        ref_extract = StarExtractionModule(name_in="star_cut_ref",
                                           image_in_tag="bp_cleaned_arr",
                                           image_out_tag="im_arr_ref",
                                           image_size=0.6,
                                           fwhm_star=0.1,
                                           position=None)

        alignment = StarAlignmentModule(name_in="star_align",
                                        image_in_tag="im_arr_extract",
                                        ref_image_in_tag="im_arr_ref",
                                        image_out_tag="im_arr_aligned",
                                        accuracy=10,
                                        resize=2)

        self.pipeline.add_module(extraction)
        self.pipeline.add_module(ref_extract)
        self.pipeline.add_module(alignment)

        angle_calc = AngleInterpolationModule(name_in="angle_calculation",
                                            data_tag="im_arr_aligned")

        self.pipeline.add_module(angle_calc)

        subset = StackAndSubsetModule(name_in="stacking_subset",
                                      image_in_tag="im_arr_aligned",
                                      image_out_tag="im_arr_stacked",
                                      random=None,
                                      stacking=4)

        self.pipeline.add_module(subset)

        psf_sub = PSFSubtractionModule(pca_number=5,
                                       name_in="PSF_subtraction",
                                       images_in_tag="im_arr_stacked",
                                       reference_in_tag="im_arr_stacked",
                                       res_mean_tag="res_mean")

        self.pipeline.add_module(psf_sub)

        writing = FitsWritingModule(name_in="Fits_writing",
                                    file_name="test.fits",
                                    data_tag="res_mean")

        self.pipeline.add_module(writing)

        self.pipeline.run()

        data = self.pipeline.get_data("im_arr")
        assert np.allclose(data[0, 61, 39], -0.00022889163546536875, rtol=limit, atol=0.)

        data = self.pipeline.get_data("dark_arr")
        assert np.allclose(data[0, 61, 39], 2.368170995592123e-05, rtol=limit, atol=0.)

        data = self.pipeline.get_data("flat_arr")
        assert np.allclose(data[0, 61, 39], 0.98703416941301647, rtol=limit, atol=0.)

        data = self.pipeline.get_data("im_arr_last")
        assert np.allclose(data[0, 61, 39], -0.00022889163546536875, rtol=limit, atol=0.)

        data = self.pipeline.get_data("im_arr_cut")
        assert np.allclose(data[0, 61, 39], -0.00022889163546536875, rtol=limit, atol=0.)

        data = self.pipeline.get_data("dark_sub_arr")
        assert np.allclose(data[0, 61, 39], -0.00021601281733413911, rtol=limit, atol=0.)

        data = self.pipeline.get_data("flat_sub_arr")
        assert np.allclose(data[0, 61, 39], -0.00021647987125847178, rtol=limit, atol=0.)

        data = self.pipeline.get_data("bg_cleaned_arr")
        assert np.allclose(data[0, 61, 39], -0.00013095662386792948, rtol=limit, atol=0.)

        data = self.pipeline.get_data("bp_cleaned_arr")
        assert np.allclose(data[0, 61, 39], -0.00013095662386792948, rtol=limit, atol=0.)

        data = self.pipeline.get_data("im_arr_extract")
        assert np.allclose(data[0, 10, 10], 0.052958146579313935, rtol=limit, atol=0.)

        data = self.pipeline.get_data("im_arr_aligned")
        assert np.allclose(data[0, 10, 10], 1.1307471842831197e-05, rtol=limit, atol=0.)

        data = self.pipeline.get_data("im_arr_stacked")
        assert np.allclose(data[0, 10, 10], 2.5572805947810986e-05, rtol=limit, atol=0.)

        data = self.pipeline.get_data("res_mean")
        assert np.allclose(data[38, 22], 0.00018312083384477404, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), -1.598348168584834e-07, rtol=limit, atol=0.)
        assert data.shape == (44, 44)
