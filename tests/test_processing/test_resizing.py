import os
import warnings

import pytest
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.fitsreading import FitsReadingModule
from pynpoint.processing.resizing import CropImagesModule, ScaleImagesModule, \
                                         AddLinesModule, RemoveLinesModule
from pynpoint.util.tests import create_config, create_star_data, remove_test_data

warnings.simplefilter("always")

limit = 1e-10

class TestImageResizing(object):

    def setup_class(self):

        self.test_dir = os.path.dirname(__file__) + "/"

        create_star_data(path=self.test_dir+"resize")
        create_config(self.test_dir+"PynPoint_config.ini")

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self):

        remove_test_data(self.test_dir, folders=["resize"])

    def test_read_data(self):

        read = FitsReadingModule(name_in="read",
                                 image_tag="read",
                                 input_dir=self.test_dir+"resize",
                                 overwrite=True,
                                 check=True)

        self.pipeline.add_module(read)

        self.pipeline.run_module("read")

        data = self.pipeline.get_data("read")
        assert np.allclose(data[0, 50, 50], 0.09798413502193704, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010029494781738066, rtol=limit, atol=0.)
        assert data.shape == (40, 100, 100)

    def test_crop_images(self):

        crop = CropImagesModule(size=0.3,
                                center=None,
                                name_in="crop1",
                                image_in_tag="read",
                                image_out_tag="crop1")

        self.pipeline.add_module(crop)

        crop = CropImagesModule(size=0.3,
                                center=(10, 10),
                                name_in="crop2",
                                image_in_tag="read",
                                image_out_tag="crop2")

        self.pipeline.add_module(crop)

        self.pipeline.run_module("crop1")
        self.pipeline.run_module("crop2")

        data = self.pipeline.get_data("crop1")
        assert np.allclose(data[0, 7, 7], 0.09798413502193704, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.005917829617688413, rtol=limit, atol=0.)
        assert data.shape == (40, 13, 13)

        data = self.pipeline.get_data("crop2")
        assert np.allclose(data[0, 7, 7], 0.00021012292977345447, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), -2.9375442509680414e-06, rtol=limit, atol=0.)
        assert data.shape == (40, 13, 13)

    def test_scale_images(self):

        scale = ScaleImagesModule(scaling=(2., 2., None),
                                  name_in="scale1",
                                  image_in_tag="read",
                                  image_out_tag="scale1")

        self.pipeline.add_module(scale)

        scale = ScaleImagesModule(scaling=(None, None, 2.),
                                  name_in="scale2",
                                  image_in_tag="read",
                                  image_out_tag="scale2")

        self.pipeline.add_module(scale)

        self.pipeline.run_module("scale1")
        self.pipeline.run_module("scale2")

        data = self.pipeline.get_data("scale1")
        assert np.allclose(data[0, 100, 100], 0.02356955774929094, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 2.507373695434516e-05, rtol=limit, atol=0.)
        assert data.shape == (40, 200, 200)

        data = self.pipeline.get_data("scale2")
        assert np.allclose(data[0, 50, 50], 0.19596827004387415, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00020058989563476127, rtol=limit, atol=0.)
        assert data.shape == (40, 100, 100)

    def test_add_lines(self):

        add = AddLinesModule(lines=(2, 5, 0, 9),
                             name_in="add",
                             image_in_tag="read",
                             image_out_tag="add")

        self.pipeline.add_module(add)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module("add")

        assert len(warning) == 1
        assert warning[0].message.args[0] == "The dimensions of the output images (109, 107) " \
                                             "are not equal. PynPoint only supports square images."

        data = self.pipeline.get_data("add")
        assert np.allclose(data[0, 50, 50], 0.02851872141873229, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 8.599412485413757e-05, rtol=limit, atol=0.)
        assert data.shape == (40, 109, 107)

    def test_remove_lines(self):

        remove = RemoveLinesModule(lines=(2, 5, 0, 9),
                                   name_in="remove",
                                   image_in_tag="read",
                                   image_out_tag="remove")

        self.pipeline.add_module(remove)
        self.pipeline.run_module("remove")

        data = self.pipeline.get_data("remove")
        assert np.allclose(data[0, 50, 50], 0.028455980719223083, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00011848528804183087, rtol=limit, atol=0.)
        assert data.shape == (40, 91, 93)
