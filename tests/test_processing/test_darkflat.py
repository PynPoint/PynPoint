import os
import warnings

import h5py
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.processing.darkflat import DarkCalibrationModule, FlatCalibrationModule
from pynpoint.util.tests import create_config, remove_test_data

warnings.simplefilter("always")

limit = 1e-10

class TestDarkAndFlatCalibration(object):

    def setup_class(self):

        self.test_dir = os.path.dirname(__file__) + "/"

        np.random.seed(1)

        images = np.random.normal(loc=0, scale=2e-4, size=(10, 100, 100))
        dark = np.random.normal(loc=0, scale=2e-4, size=(10, 100, 100))
        flat = np.random.normal(loc=0, scale=2e-4, size=(10, 100, 100))

        h5f = h5py.File(self.test_dir+"PynPoint_database.hdf5", "w")
        h5f.create_dataset("images", data=images)
        h5f.create_dataset("dark", data=dark)
        h5f.create_dataset("flat", data=flat)
        h5f.close()

        create_config(self.test_dir+"PynPoint_config.ini")

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self):

        remove_test_data(self.test_dir)

    def test_dark_calibration(self):

        dark = DarkCalibrationModule(name_in="dark",
                                     image_in_tag="images",
                                     dark_in_tag="dark",
                                     image_out_tag="dark_cal")

        self.pipeline.add_module(dark)
        self.pipeline.run_module("dark")

        data = self.pipeline.get_data("dark")
        assert np.allclose(data[0, 10, 10], 3.528694163309295e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 7.368663496379876e-07, rtol=limit, atol=0.)
        assert data.shape == (10, 100, 100)

    def test_flat_calibration(self):

        flat = FlatCalibrationModule(name_in="flat",
                                     image_in_tag="dark_cal",
                                     flat_in_tag="flat",
                                     image_out_tag="flat_cal")

        self.pipeline.add_module(flat)
        self.pipeline.run_module("flat")

        data = self.pipeline.get_data("flat")
        assert np.allclose(data[0, 10, 10], -0.0004053528990466237, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), -4.056978234798532e-07, rtol=limit, atol=0.)
        assert data.shape == (10, 100, 100)
