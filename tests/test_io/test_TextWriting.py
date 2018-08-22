import os
import warnings

import numpy as np

from PynPoint.Core.Pypeline import Pypeline
from PynPoint.IOmodules.TextWriting import TextWritingModule, \
                                           ParangWritingModule, \
                                           AttributeWritingModule
from PynPoint.Util.TestTools import create_config, create_random, remove_test_data

warnings.simplefilter("always")

limit = 1e-10

class TestTextReading(object):

    def setup_class(self):

        self.test_dir = os.path.dirname(__file__) + "/"

        create_random(self.test_dir, ndit=1)

        create_config(self.test_dir+"PynPoint_config.ini")

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self):

        remove_test_data(self.test_dir, files=["image.dat", "parang.dat", "attribute.dat"])

    def test_input_data(self):

        data = self.pipeline.get_data("images")
        assert np.allclose(data[0, 75, 25], 6.921353838812206e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 1.9545313398209947e-06, rtol=limit, atol=0.)
        assert data.shape == (1, 100, 100)

    def test_text_writing(self):

        text_write = TextWritingModule(file_name="image.dat",
                                       name_in="text_write",
                                       output_dir=None,
                                       data_tag="images",
                                       header=None)

        self.pipeline.add_module(text_write)
        self.pipeline.run_module("text_write")

        data = np.loadtxt(self.test_dir+"image.dat")

        assert np.allclose(data[75, 25], 6.921353838812206e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 1.9545313398209947e-06, rtol=limit, atol=0.)
        assert data.shape == (100, 100)

    def test_parang_writing(self):

        parang_write = ParangWritingModule(file_name="parang.dat",
                                           name_in="parang_write",
                                           output_dir=None,
                                           data_tag="images",
                                           header=None)

        self.pipeline.add_module(parang_write)
        self.pipeline.run_module("parang_write")

        data = np.loadtxt(self.test_dir+"parang.dat")

        assert np.allclose(data[0], 1.0, rtol=limit, atol=0.)
        assert np.allclose(data[9], 10.0, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 5.5, rtol=limit, atol=0.)
        assert data.shape == (10, )

    def test_attribute_writing(self):

        attr_write = AttributeWritingModule(file_name="attribute.dat",
                                            name_in="attr_write",
                                            output_dir=None,
                                            data_tag="images",
                                            attribute="PARANG",
                                            header=None)

        self.pipeline.add_module(attr_write)
        self.pipeline.run_module("attr_write")

        data = np.loadtxt(self.test_dir+"attribute.dat")

        assert np.allclose(data[0], 1.0, rtol=limit, atol=0.)
        assert np.allclose(data[9], 10.0, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 5.5, rtol=limit, atol=0.)
        assert data.shape == (10, )
