from __future__ import absolute_import
import os
import warnings

import h5py
import pytest
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.textwriting import TextWritingModule, ParangWritingModule, \
                                           AttributeWritingModule
from pynpoint.util.tests import create_config, create_random, remove_test_data

warnings.simplefilter("always")

limit = 1e-10

class TestTextReading(object):

    def setup_class(self):

        self.test_dir = os.path.dirname(__file__) + "/"

        create_random(self.test_dir, ndit=1)
        create_config(self.test_dir+"PynPoint_config.ini")

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self):

        remove_test_data(self.test_dir, files=["image.dat", "parang.dat",
                                               "attribute.dat", "data.dat"])

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

    def test_text_writing_string(self):

        with pytest.raises(ValueError) as error:
            TextWritingModule(file_name=0.,
                              name_in="text_write",
                              output_dir=None,
                              data_tag="images",
                              header=None)

        assert str(error.value) == "Output 'file_name' needs to be a string."

    def test_text_writing_ndim(self):

        data_4d = np.random.normal(loc=0, scale=2e-4, size=(5, 5, 5, 5))

        h5f = h5py.File(self.test_dir+"PynPoint_database.hdf5", "a")
        h5f.create_dataset("data_4d", data=data_4d)
        h5f.close()

        text_write = TextWritingModule(file_name="data.dat",
                                       name_in="write_4d",
                                       output_dir=None,
                                       data_tag="data_4d",
                                       header=None)

        self.pipeline.add_module(text_write)

        with pytest.raises(ValueError) as error:
            self.pipeline.run_module("write_4d")

        assert str(error.value) == "Only 1D or 2D arrays can be written to a text file."

    def test_text_writing_int(self):

        data_int = np.arange(1, 101, 1)

        h5f = h5py.File(self.test_dir+"PynPoint_database.hdf5", "a")
        h5f.create_dataset("data_int", data=data_int)
        h5f.close()

        text_write = TextWritingModule(file_name="data.dat",
                                       name_in="write_int",
                                       output_dir=None,
                                       data_tag="data_int",
                                       header=None)

        self.pipeline.add_module(text_write)
        self.pipeline.run_module("write_int")

        data = np.loadtxt(self.test_dir+"data.dat")

        assert np.allclose(data, data_int, rtol=limit, atol=0.)
        assert data.shape == (100, )

    def test_parang_writing(self):

        parang_write = ParangWritingModule(file_name="parang.dat",
                                           name_in="parang_write1",
                                           output_dir=None,
                                           data_tag="images",
                                           header=None)

        self.pipeline.add_module(parang_write)
        self.pipeline.run_module("parang_write1")

        data = np.loadtxt(self.test_dir+"parang.dat")

        assert np.allclose(data[0], 1.0, rtol=limit, atol=0.)
        assert np.allclose(data[9], 10.0, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 5.5, rtol=limit, atol=0.)
        assert data.shape == (10, )

    def test_parang_writing_string(self):

        with pytest.raises(ValueError) as error:
            ParangWritingModule(file_name=0.,
                                name_in="parang_write2",
                                output_dir=None,
                                data_tag="images",
                                header=None)

        assert str(error.value) == "Output 'file_name' needs to be a string."

    def test_attribute_writing(self):

        attr_write = AttributeWritingModule(file_name="attribute.dat",
                                            name_in="attr_write1",
                                            output_dir=None,
                                            data_tag="images",
                                            attribute="PARANG",
                                            header=None)

        self.pipeline.add_module(attr_write)
        self.pipeline.run_module("attr_write1")

        data = np.loadtxt(self.test_dir+"attribute.dat")

        assert np.allclose(data[0], 1.0, rtol=limit, atol=0.)
        assert np.allclose(data[9], 10.0, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 5.5, rtol=limit, atol=0.)
        assert data.shape == (10, )

    def test_attribute_writing_string(self):

        with pytest.raises(ValueError) as error:
            AttributeWritingModule(file_name=0.,
                                   name_in="attr_write2",
                                   output_dir=None,
                                   data_tag="images",
                                   attribute="PARANG",
                                   header=None)

        assert str(error.value) == "Output 'file_name' needs to be a string."

    def test_attribute_not_present(self):

        attr_write = AttributeWritingModule(file_name="attribute.dat",
                                            name_in="attr_write3",
                                            output_dir=None,
                                            data_tag="images",
                                            attribute="test",
                                            header=None)

        self.pipeline.add_module(attr_write)

        with pytest.raises(ValueError) as error:
            self.pipeline.run_module("attr_write3")

        assert str(error.value) == "The 'test' attribute is not present in 'images'."

    def test_parang_writing_not_present(self):

        h5f = h5py.File(self.test_dir+"PynPoint_database.hdf5", "a")
        del h5f["header_images/PARANG"]
        h5f.close()

        parang_write = ParangWritingModule(file_name="parang.dat",
                                           name_in="parang_write3",
                                           output_dir=None,
                                           data_tag="images",
                                           header=None)

        self.pipeline.add_module(parang_write)

        with pytest.raises(ValueError) as error:
            self.pipeline.run_module("parang_write3")

        assert str(error.value) == "The PARANG attribute is not present in 'images'."
