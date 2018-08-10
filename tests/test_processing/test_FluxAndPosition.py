import os
import warnings

import numpy as np

from PynPoint.Core.Pypeline import Pypeline
from PynPoint.IOmodules.FitsReading import FitsReadingModule
from PynPoint.ProcessingModules.FluxAndPosition import FakePlanetModule, \
                                                       AperturePhotometryModule, \
                                                       FalsePositiveModule, \
                                                       SimplexMinimizationModule
from PynPoint.ProcessingModules.PSFpreparation import AngleInterpolationModule
from PynPoint.ProcessingModules.PSFSubtractionPCA import PcaPsfSubtractionModule
from PynPoint.Util.TestTools import create_config, create_star_data, remove_test_data

warnings.simplefilter("always")

limit = 1e-10

class TestFluxAndPosition(object):

    def setup_class(self):

        self.test_dir = os.path.dirname(__file__) + "/"

        create_star_data(path=self.test_dir+"flux", npix_x=101, npix_y=101)
        create_config(self.test_dir+"PynPoint_config.ini")

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self):

        remove_test_data(self.test_dir, folders=["flux"])

    def test_read_data(self):

        read = FitsReadingModule(name_in="read",
                                 image_tag="read",
                                 input_dir=self.test_dir+"flux")

        self.pipeline.add_module(read)
        self.pipeline.run_module("read")

        data = self.pipeline.get_data("read")
        assert np.allclose(data[0, 50, 50], 0.0986064357966972, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 9.827812356946396e-05, rtol=limit, atol=0.)
        assert data.shape == (40, 101, 101)

    def test_aperture_photometry(self):

        photometry = AperturePhotometryModule(radius=0.1,
                                              position=None,
                                              name_in="photometry",
                                              image_in_tag="read",
                                              phot_out_tag="photometry")

        self.pipeline.add_module(photometry)
        self.pipeline.run_module("photometry")

        data = self.pipeline.get_data("photometry")
        assert np.allclose(data[0][0], 0.9702137183213615, rtol=limit, atol=0.)
        assert np.allclose(data[39][0], 0.9691512171281103, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.9691752104364761, rtol=limit, atol=0.)
        assert data.shape == (40, 1)

    def test_angle_interpolation(self):

        angle = AngleInterpolationModule(name_in="angle",
                                         data_tag="read")

        self.pipeline.add_module(angle)
        self.pipeline.run_module("angle")

        data = self.pipeline.get_data("header_read/PARANG")
        assert data[5] == 2.7777777777777777
        assert np.allclose(np.mean(data), 10.0, rtol=limit, atol=0.)
        assert data.shape == (40, )

    def test_fake_planet(self):

        fake = FakePlanetModule(position=(0.5, 90.),
                                magnitude=6.,
                                psf_scaling=1.,
                                interpolation="spline",
                                name_in="fake",
                                image_in_tag="read",
                                psf_in_tag="read",
                                image_out_tag="fake",
                                verbose=True)

        self.pipeline.add_module(fake)
        self.pipeline.run_module("fake")

        data = self.pipeline.get_data("fake")
        assert np.allclose(data[0, 50, 50], 0.09860622347589054, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 9.867026482551375e-05, rtol=limit, atol=0.)
        assert data.shape == (40, 101, 101)

    def test_psf_subtraction(self):

        pca = PcaPsfSubtractionModule(pca_numbers=(2, ),
                                      name_in="pca",
                                      images_in_tag="fake",
                                      reference_in_tag="fake",
                                      res_mean_tag="res_mean",
                                      res_median_tag=None,
                                      res_arr_out_tag=None,
                                      res_rot_mean_clip_tag=None,
                                      extra_rot=0.)

        self.pipeline.add_module(pca)
        self.pipeline.run_module("pca")

        data = self.pipeline.get_data("res_mean")
        assert np.allclose(data[0, 49, 31], 4.8963214463463886e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 1.8409659677297164e-08, rtol=limit, atol=0.)
        assert data.shape == (1, 101, 101)

    def test_false_positive(self):

        false = FalsePositiveModule(position=(31., 49.),
                                    aperture=0.1,
                                    ignore=True,
                                    name_in="false",
                                    image_in_tag="res_mean",
                                    snr_out_tag="snr_fpf")

        self.pipeline.add_module(false)
        self.pipeline.run_module("false")

        data = self.pipeline.get_data("snr_fpf")
        assert np.allclose(data[0, 2], 0.5280553948214145, rtol=limit, atol=0.)
        assert np.allclose(data[0, 3], 94.39870535499551, rtol=limit, atol=0.)
        assert np.allclose(data[0, 4], 8.542166952478182, rtol=limit, atol=0.)
        assert np.allclose(data[0, 5], 9.54772666372783e-07, rtol=limit, atol=0.)

    def test_simplex_minimization(self):

        simplex = SimplexMinimizationModule(position=(31., 49.),
                                            magnitude=6.,
                                            psf_scaling=-1.,
                                            name_in="simplex",
                                            image_in_tag="fake",
                                            psf_in_tag="read",
                                            res_out_tag="simplex_res",
                                            flux_position_tag="flux_position",
                                            merit="hessian",
                                            aperture=0.1,
                                            sigma=0.,
                                            tolerance=0.1,
                                            pca_number=1,
                                            cent_size=0.1,
                                            edge_size=None,
                                            extra_rot=0.)

        self.pipeline.add_module(simplex)
        self.pipeline.run_module("simplex")

        data = self.pipeline.get_data("simplex_res")
        assert np.allclose(data[52, 49, 31], -5.57733704809696e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 9.455537503672783e-09, rtol=limit, atol=0.)
        assert data.shape == (53, 101, 101)

        data = self.pipeline.get_data("flux_position")
        assert np.allclose(data[52, 0], 32.35324020839204, rtol=limit, atol=0.)
        assert np.allclose(data[52, 1], 50.26878876587102, rtol=limit, atol=0.)
        assert np.allclose(data[52, 2], 0.4900022824802747, rtol=limit, atol=0.)
        assert np.allclose(data[52, 3], 90.72997666033649, rtol=limit, atol=0.)
        assert np.allclose(data[52, 4], 5.957337187612481, rtol=limit, atol=0.)
        assert data.shape == (53, 6)
