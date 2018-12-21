import os
import warnings

import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.fitsreading import FitsReadingModule
from pynpoint.readwrite.fitswriting import FitsWritingModule
from pynpoint.processing.badpixel import BadPixelSigmaFilterModule
from pynpoint.processing.darkflat import DarkCalibrationModule, FlatCalibrationModule
from pynpoint.processing.resizing import RemoveLinesModule
from pynpoint.processing.psfpreparation import AngleInterpolationModule
from pynpoint.processing.background import MeanBackgroundSubtractionModule
from pynpoint.processing.centering import StarExtractionModule, StarAlignmentModule
from pynpoint.processing.psfsubtraction import PcaPsfSubtractionModule
from pynpoint.processing.frameselection import RemoveLastFrameModule
from pynpoint.processing.stacksubset import StackAndSubsetModule
from pynpoint.util.tests import create_config, create_fits, create_fake, remove_test_data

warnings.simplefilter("always")

limit = 1e-10

class TestDocumentation(object):

    def setup_class(self):

        self.test_dir = os.path.dirname(__file__) + "/"

        # science

        create_fake(path=self.test_dir+'adi',
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

        # dark

        ndit = [3, 3, 5, 5]

        np.random.seed(2)

        for j, item in enumerate(ndit):
            image = np.random.normal(loc=0, scale=2e-4, size=(item, 100, 100))
            create_fits(self.test_dir+'dark', 'dark'+str(j+1).zfill(2)+'.fits', image, ndit[j])

        # flat

        ndit = [3, 3, 5, 5]

        np.random.seed(3)

        for j, item in enumerate(ndit):
            image = np.random.normal(loc=1, scale=1e-2, size=(item, 100, 100))
            create_fits(self.test_dir+'flat', 'flat'+str(j+1).zfill(2)+'.fits', image, ndit[j])

        create_config(self.test_dir+"PynPoint_config.ini")

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self):

        remove_test_data(self.test_dir, folders=["adi", "dark", "flat"], files=["test.fits"])

    def test_read_data(self):
        read_science = FitsReadingModule(name_in="read_science",
                                         input_dir=self.test_dir+"adi",
                                         image_tag="im_arr")

        self.pipeline.add_module(read_science)

        read_dark = FitsReadingModule(name_in="read_dark",
                                      input_dir=self.test_dir+"dark",
                                      image_tag="dark_arr")

        self.pipeline.add_module(read_dark)

        read_flat = FitsReadingModule(name_in="read_flat",
                                      input_dir=self.test_dir+"flat",
                                      image_tag="flat_arr")

        self.pipeline.add_module(read_flat)

        self.pipeline.run_module("read_science")
        self.pipeline.run_module("read_dark")
        self.pipeline.run_module("read_flat")

        data = self.pipeline.get_data("im_arr")
        assert np.allclose(data[0, 61, 39], -0.00022889163546536875, rtol=limit, atol=0.)
        assert data.shape == (82, 102, 100)

        data = self.pipeline.get_data("dark_arr")
        assert np.allclose(data[0, 61, 39], 2.368170995592123e-05, rtol=limit, atol=0.)
        assert data.shape == (16, 100, 100)

        data = self.pipeline.get_data("flat_arr")
        assert np.allclose(data[0, 61, 39], 0.98703416941301647, rtol=limit, atol=0.)
        assert data.shape == (16, 100, 100)

    def test_remove_last(self):

        remove_last = RemoveLastFrameModule(name_in="last_frame",
                                            image_in_tag="im_arr",
                                            image_out_tag="im_arr_last")

        self.pipeline.add_module(remove_last)
        self.pipeline.run_module("last_frame")

        data = self.pipeline.get_data("im_arr_last")
        assert np.allclose(data[0, 61, 39], -0.00022889163546536875, rtol=limit, atol=0.)
        assert data.shape == (78, 102, 100)

    def test_remove_lines(self):

        cutting = RemoveLinesModule(lines=(0, 0, 0, 2),
                                    name_in="cut_lines",
                                    image_in_tag="im_arr_last",
                                    image_out_tag="im_arr_cut")

        self.pipeline.add_module(cutting)
        self.pipeline.run_module("cut_lines")

        data = self.pipeline.get_data("im_arr_cut")
        assert np.allclose(data[0, 61, 39], -0.00022889163546536875, rtol=limit, atol=0.)
        assert data.shape == (78, 100, 100)

    def test_dark_calibration(self):

        dark_sub = DarkCalibrationModule(name_in="dark_subtraction",
                                         image_in_tag="im_arr_cut",
                                         dark_in_tag="dark_arr",
                                         image_out_tag="dark_sub_arr")

        self.pipeline.add_module(dark_sub)
        self.pipeline.run_module("dark_subtraction")

        data = self.pipeline.get_data("dark_sub_arr")
        assert np.allclose(data[0, 61, 39], -0.00021601281733413911, rtol=limit, atol=0.)
        assert data.shape == (78, 100, 100)

    def test_flat_calibration(self):

        flat_sub = FlatCalibrationModule(name_in="flat_subtraction",
                                         image_in_tag="dark_sub_arr",
                                         flat_in_tag="flat_arr",
                                         image_out_tag="flat_sub_arr")


        self.pipeline.add_module(flat_sub)
        self.pipeline.run_module("flat_subtraction")

        data = self.pipeline.get_data("flat_sub_arr")
        assert np.allclose(data[0, 61, 39], -0.00021647987125847178, rtol=limit, atol=0.)
        assert data.shape == (78, 100, 100)

    def test_mean_background(self):

        bg_subtraction = MeanBackgroundSubtractionModule(shift=None,
                                                         cubes=1,
                                                         name_in="background_subtraction",
                                                         image_in_tag="flat_sub_arr",
                                                         image_out_tag="bg_cleaned_arr")


        self.pipeline.add_module(bg_subtraction)
        self.pipeline.run_module("background_subtraction")

        data = self.pipeline.get_data("bg_cleaned_arr")
        assert np.allclose(data[0, 61, 39], -0.00013095662386792948, rtol=limit, atol=0.)
        assert data.shape == (78, 100, 100)

    def test_bad_pixel(self):

        bp_cleaning = BadPixelSigmaFilterModule(name_in="sigma_filtering",
                                                image_in_tag="bg_cleaned_arr",
                                                image_out_tag="bp_cleaned_arr")

        self.pipeline.add_module(bp_cleaning)
        self.pipeline.run_module("sigma_filtering")

        data = self.pipeline.get_data("bp_cleaned_arr")
        assert np.allclose(data[0, 61, 39], -0.00013095662386792948, rtol=limit, atol=0.)
        assert data.shape == (78, 100, 100)

    def test_star_extract(self):

        extraction = StarExtractionModule(name_in="star_cutting",
                                          image_in_tag="bp_cleaned_arr",
                                          image_out_tag="im_arr_extract",
                                          image_size=0.6,
                                          fwhm_star=0.1,
                                          position=None)

        self.pipeline.add_module(extraction)
        self.pipeline.run_module("star_cutting")

        data = self.pipeline.get_data("im_arr_extract")
        assert np.allclose(data[0, 10, 10], 0.052958146579313935, rtol=limit, atol=0.)
        assert data.shape == (78, 23, 23)

    def test_star_alignment(self):

        # Required for ref_image_in_tag in StarAlignmentModule, otherwise a random frame is used
        ref_extract = StarExtractionModule(name_in="star_cut_ref",
                                           image_in_tag="bp_cleaned_arr",
                                           image_out_tag="im_arr_ref",
                                           image_size=0.6,
                                           fwhm_star=0.1,
                                           position=None)

        self.pipeline.add_module(ref_extract)

        alignment = StarAlignmentModule(name_in="star_align",
                                        image_in_tag="im_arr_extract",
                                        ref_image_in_tag="im_arr_ref",
                                        image_out_tag="im_arr_aligned",
                                        accuracy=10,
                                        resize=2)

        self.pipeline.add_module(alignment)

        self.pipeline.run_module("star_cut_ref")
        self.pipeline.run_module("star_align")

        data = self.pipeline.get_data("im_arr_aligned")
        assert np.allclose(data[0, 10, 10], 1.1309588556526325e-05, rtol=limit, atol=0.)
        assert data.shape == (78, 46, 46)

    def test_angle_interpolation(self):

        angle_calc = AngleInterpolationModule(name_in="angle_interpolation",
                                              data_tag="im_arr_aligned")

        self.pipeline.add_module(angle_calc)
        self.pipeline.run_module("angle_interpolation")

        data = self.pipeline.get_data("header_im_arr_aligned/PARANG")
        assert np.allclose(data[5, ], 5.9523809523809526, rtol=limit, atol=0.)
        assert data.shape == (78, )

    def test_stack(self):

        subset = StackAndSubsetModule(name_in="stacking_subset",
                                      image_in_tag="im_arr_aligned",
                                      image_out_tag="im_arr_stacked",
                                      random=None,
                                      stacking=4)

        self.pipeline.add_module(subset)
        self.pipeline.run_module("stacking_subset")

        data = self.pipeline.get_data("im_arr_stacked")
        assert np.allclose(data[0, 10, 10], 2.55745339731812e-05, rtol=limit, atol=0.)
        assert data.shape == (20, 46, 46)

    def test_psf_subtraction(self):

        pca = PcaPsfSubtractionModule(pca_numbers=(5, ),
                                      name_in="psf_subtraction",
                                      images_in_tag="im_arr_stacked",
                                      reference_in_tag="im_arr_stacked",
                                      res_mean_tag="res_mean",
                                      res_median_tag=None,
                                      res_arr_out_tag=None,
                                      res_rot_mean_clip_tag=None,
                                      extra_rot=0.)

        self.pipeline.add_module(pca)
        self.pipeline.run_module("psf_subtraction")

        data = self.pipeline.get_data("res_mean")
        assert np.allclose(data[0, 38, 22], 2.073746596517549e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), -1.5133345435496935e-08, rtol=limit, atol=0.)
        assert data.shape == (1, 46, 46)

    def test_write_fits(self):

        writing = FitsWritingModule(name_in="fits_writing",
                                    file_name="test.fits",
                                    data_tag="res_mean")

        self.pipeline.add_module(writing)
        self.pipeline.run_module("fits_writing")

        assert os.path.exists(self.test_dir+"test.fits")
