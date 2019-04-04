from __future__ import absolute_import

import os
import warnings

import h5py
import pytest
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.fitsreading import FitsReadingModule
from pynpoint.processing.badpixel import BadPixelSigmaFilterModule
from pynpoint.processing.darkflat import DarkCalibrationModule
from pynpoint.processing.resizing import RemoveLinesModule, ScaleImagesModule
from pynpoint.util.tests import create_config, create_star_data, remove_test_data

warnings.simplefilter("always")

limit = 1e-10

class TestPypeline(object):

    def setup_class(self):
        self.test_dir = os.path.dirname(__file__) + "/"

        np.random.seed(1)

        image_3d = np.random.normal(loc=0, scale=2e-4, size=(4, 10, 10))
        image_2d = np.random.normal(loc=0, scale=2e-4, size=(1, 10, 10))
        science = np.random.normal(loc=0, scale=2e-4, size=(4, 10, 10))
        dark = np.random.normal(loc=0, scale=2e-4, size=(4, 10, 10))

        h5f = h5py.File(self.test_dir+"PynPoint_database.hdf5", "w")
        h5f.create_dataset("image_3d", data=image_3d)
        h5f.create_dataset("image_2d", data=image_2d)
        h5f.create_dataset("science", data=science)
        h5f.create_dataset("dark", data=dark)
        h5f.close()

        create_star_data(path=self.test_dir+"images")
        create_config(self.test_dir+"PynPoint_config.ini")

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self):
        remove_test_data(self.test_dir, folders=["images"])

    def test_output_port_name(self):
        read = FitsReadingModule(name_in="read", input_dir=self.test_dir+"images",
                                 image_tag="images")
        read.add_output_port("test")

        with pytest.warns(UserWarning) as warning:
            read.add_output_port("test")

        assert len(warning) == 1
        assert warning[0].message.args[0] == "Tag 'test' of ReadingModule 'read' is already used."

        process = BadPixelSigmaFilterModule(name_in="badpixel", image_in_tag="images")
        process.add_output_port("test")

        with pytest.warns(UserWarning) as warning:
            process.add_output_port("test")

        assert len(warning) == 1
        assert warning[0].message.args[0] == "Tag 'test' of ProcessingModule 'badpixel' is " \
                                             "already used."

        self.pipeline.m_data_storage.close_connection()

        process._m_data_base = self.test_dir+"database.hdf5"
        process.add_output_port("new")

    def test_apply_function_to_images_3d(self):
        self.pipeline.set_attribute("config", "MEMORY", 1, static=True)

        remove = RemoveLinesModule(lines=(1, 0, 0, 0),
                                   name_in="remove1",
                                   image_in_tag="image_3d",
                                   image_out_tag="remove_3d")

        self.pipeline.add_module(remove)
        self.pipeline.run_module("remove1")

        data = self.pipeline.get_data("image_3d")
        assert np.allclose(np.mean(data), 1.0141852764605783e-05, rtol=limit, atol=0.)
        assert data.shape == (4, 10, 10)

        data = self.pipeline.get_data("remove_3d")
        assert np.allclose(np.mean(data), 1.1477029889801025e-05, rtol=limit, atol=0.)
        assert data.shape == (4, 10, 9)

    def test_apply_function_to_images_2d(self):
        remove = RemoveLinesModule(lines=(1, 0, 0, 0),
                                   name_in="remove2",
                                   image_in_tag="image_2d",
                                   image_out_tag="remove_2d")

        self.pipeline.add_module(remove)
        self.pipeline.run_module("remove2")

        data = self.pipeline.get_data("image_2d")
        assert np.allclose(np.mean(data), 1.2869483197883442e-05, rtol=limit, atol=0.)
        assert data.shape == (1, 10, 10)

        data = self.pipeline.get_data("remove_2d")
        assert np.allclose(np.mean(data), 1.3957075246029751e-05, rtol=limit, atol=0.)
        assert data.shape == (1, 10, 9)

    def test_apply_function_to_images_same_port(self):
        dark = DarkCalibrationModule(name_in="dark1",
                                     image_in_tag="science",
                                     dark_in_tag="dark",
                                     image_out_tag="science")

        self.pipeline.add_module(dark)
        self.pipeline.run_module("dark1")

        data = self.pipeline.get_data("science")
        assert np.allclose(np.mean(data), -3.190113568690675e-06, rtol=limit, atol=0.)
        assert data.shape == (4, 10, 10)

        self.pipeline.set_attribute("config", "MEMORY", 0, static=True)

        dark = DarkCalibrationModule(name_in="dark2",
                                     image_in_tag="science",
                                     dark_in_tag="dark",
                                     image_out_tag="science")

        self.pipeline.add_module(dark)
        self.pipeline.run_module("dark2")

        data = self.pipeline.get_data("science")
        assert np.allclose(np.mean(data), -1.026073475228737e-05, rtol=limit, atol=0.)
        assert data.shape == (4, 10, 10)

        remove = RemoveLinesModule(lines=(1, 0, 0, 0),
                                   name_in="remove3",
                                   image_in_tag="remove_3d",
                                   image_out_tag="remove_3d")

        self.pipeline.add_module(remove)

        with pytest.raises(ValueError) as error:
            self.pipeline.run_module("remove3")

        assert str(error.value) == "Input and output port have the same tag while the input " \
                                   "function is changing the image shape. This is only " \
                                   "possible with MEMORY=None."

    def test_apply_function_to_images_memory_none(self):
        remove = RemoveLinesModule(lines=(1, 0, 0, 0),
                                   name_in="remove4",
                                   image_in_tag="image_3d",
                                   image_out_tag="remove_3d_none")

        self.pipeline.add_module(remove)
        self.pipeline.run_module("remove4")

        data = self.pipeline.get_data("remove_3d_none")
        assert np.allclose(np.mean(data), 1.1477029889801025e-05, rtol=limit, atol=0.)
        assert data.shape == (4, 10, 9)

    def test_apply_function_to_images_3d_args(self):
        self.pipeline.set_attribute("config", "MEMORY", 1, static=True)
        self.pipeline.set_attribute("image_3d", "PIXSCALE", 0.1, static=True)

        scale = ScaleImagesModule(scaling=(1.2, 1.2, 10.),
                                  pixscale=True,
                                  name_in="scale1",
                                  image_in_tag="image_3d",
                                  image_out_tag="scale_3d")

        self.pipeline.add_module(scale)
        self.pipeline.run_module("scale1")

        data = self.pipeline.get_data("scale_3d")
        assert np.allclose(np.mean(data), 7.042953308754017e-05, rtol=limit, atol=0.)
        assert data.shape == (4, 12, 12)

        attribute = self.pipeline.get_attribute("scale_3d", "PIXSCALE", static=True)
        assert np.allclose(attribute, 0.08333333333333334, rtol=limit, atol=0.)

    def test_apply_function_to_images_2d_args(self):
        self.pipeline.set_attribute("image_2d", "PIXSCALE", 0.1, static=True)

        scale = ScaleImagesModule(scaling=(1.2, 1.2, 10.),
                                  pixscale=True,
                                  name_in="scale2",
                                  image_in_tag="image_2d",
                                  image_out_tag="scale_2d")

        self.pipeline.add_module(scale)
        self.pipeline.run_module("scale2")

        data = self.pipeline.get_data("scale_2d")
        assert np.allclose(np.mean(data), 8.937141109641279e-05, rtol=limit, atol=0.)
        assert data.shape == (1, 12, 12)

        attribute = self.pipeline.get_attribute("scale_2d", "PIXSCALE", static=True)
        assert np.allclose(attribute, 0.08333333333333334, rtol=limit, atol=0.)
