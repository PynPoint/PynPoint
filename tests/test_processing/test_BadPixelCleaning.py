import os
import warnings

import h5py
import numpy as np

from PynPoint.Core.Pypeline import Pypeline
from PynPoint.ProcessingModules.BadPixelCleaning import BadPixelSigmaFilterModule, \
                                                        BadPixelMapModule, \
                                                        BadPixelInterpolationModule
from PynPoint.Util.TestTools import create_config, remove_test_data

warnings.simplefilter("always")

limit = 1e-10

class TestBadPixelCleaning(object):

    def setup_class(self):

        self.test_dir = os.path.dirname(__file__) + "/"

        np.random.seed(1)
        images = np.random.normal(loc=0, scale=2e-4, size=(40, 100, 100))
        dark = np.random.normal(loc=0, scale=2e-4, size=(40, 100, 100))
        flat = np.random.normal(loc=0, scale=2e-4, size=(40, 100, 100))

        images[0, 10, 10] = 1.
        images[0, 12, 12] = 1.
        images[0, 14, 14] = 1.
        images[0, 20, 20] = 1.
        images[0, 22, 22] = 1.
        images[0, 24, 24] = 1.
        dark[:, 10, 10] = 1.
        dark[:, 12, 12] = 1.
        dark[:, 14, 14] = 1.
        flat[:, 20, 20] = -1.
        flat[:, 22, 22] = -1.
        flat[:, 24, 24] = -1.

        h5f = h5py.File(self.test_dir+"PynPoint_database.hdf5", "w")
        h5f.create_dataset("images", data=images)
        h5f.create_dataset("dark", data=dark)
        h5f.create_dataset("flat", data=flat)
        h5f.create_dataset("header_images/STAR_POSITION", data=np.full((40, 2), 50.))
        h5f.close()

        create_config(self.test_dir+"PynPoint_config.ini")

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self):

        remove_test_data(self.test_dir)

    def test_bad_pixel_sigma_filter(self):

        sigma = BadPixelSigmaFilterModule(name_in="sigma",
                                          image_in_tag="images",
                                          image_out_tag="sigma",
                                          box=9,
                                          sigma=5,
                                          iterate=1)

        self.pipeline.add_module(sigma)

        self.pipeline.run_module("sigma")

        data = self.pipeline.get_data("sigma")
        assert np.allclose(data[0, 0, 0], 0.00032486907273264834, rtol=limit, atol=0.)
        assert np.allclose(data[0, 10, 10], 0.025022559679385093, rtol=limit, atol=0.)
        assert np.allclose(data[0, 20, 20], 0.024962143884217046, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 6.721637736047109e-07, rtol=limit, atol=0.)
        assert data.shape == (40, 100, 100)

    def test_bad_pixel_map(self):

        bp_map = BadPixelMapModule(name_in="bp_map",
                                   dark_in_tag="dark",
                                   flat_in_tag="flat",
                                   bp_map_out_tag="bp_map",
                                   dark_threshold=0.99,
                                   flat_threshold=-0.99)

        self.pipeline.add_module(bp_map)

        self.pipeline.run_module("bp_map")

        data = self.pipeline.get_data("bp_map")
        assert data[0, 0] == 1.
        assert data[30, 30] == 1.
        assert data[10, 10] == 0.
        assert data[12, 12] == 0.
        assert data[14, 14] == 0.
        assert data[20, 20] == 0.
        assert data[22, 22] == 0.
        assert data[24, 24] == 0.
        assert np.mean(data) == 0.9993
        assert data.shape == (100, 100)

    def test_bad_pixel_interpolation(self):

        interpolation = BadPixelInterpolationModule(name_in="interpolation",
                                                    image_in_tag="images",
                                                    bad_pixel_map_tag="bp_map",
                                                    image_out_tag="interpolation",
                                                    iterations=100)

        self.pipeline.add_module(interpolation)

        self.pipeline.run_module("interpolation")

        data = self.pipeline.get_data("interpolation")
        assert np.allclose(data[0, 0, 0], 0.00032486907273264834, rtol=limit, atol=0.)
        assert np.allclose(data[0, 10, 10], 1.0139222106683477e-05, rtol=limit, atol=0.)
        assert np.allclose(data[0, 20, 20], -4.686852973820094e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 3.0499629451215465e-07, rtol=limit, atol=0.)
        assert data.shape == (40, 100, 100)
