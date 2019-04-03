from __future__ import absolute_import

import os
import pytest
import warnings

import h5py
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.hdf5reading import Hdf5ReadingModule
from pynpoint.util.tests import create_config, create_random, remove_test_data

warnings.simplefilter("always")

limit = 1e-10

class TestHdf5ReadingModule(object):

    def setup_class(self):

        self.test_dir = os.path.dirname(__file__) + "/"

        create_random(self.test_dir+"data")
        create_config(self.test_dir+"PynPoint_config.ini")

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self):

        remove_test_data(self.test_dir, folders=["data"])

    def test_hdf5_reading(self):

        data = np.random.normal(loc=0, scale=2e-4, size=(4, 10, 10))

        h5f = h5py.File(self.test_dir+"data/PynPoint_database.hdf5", "a")
        h5f.create_dataset("extra", data=data)
        h5f.create_dataset("header_extra/PARANG", data=[1., 2., 3., 4.])
        h5f.close()

        read = Hdf5ReadingModule(name_in="read1",
                                 input_filename="PynPoint_database.hdf5",
                                 input_dir=self.test_dir+"data",
                                 tag_dictionary={"images":"images"})

        self.pipeline.add_module(read)
        self.pipeline.run_module("read1")

        data = self.pipeline.get_data("images")
        assert np.allclose(data[0, 75, 25], 6.921353838812206e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 1.0506056979365338e-06, rtol=limit, atol=0.)
        assert data.shape == (10, 100, 100)

    def test_dictionary_none(self):

        read = Hdf5ReadingModule(name_in="read2",
                                 input_filename="PynPoint_database.hdf5",
                                 input_dir=self.test_dir+"data",
                                 tag_dictionary=None)

        self.pipeline.add_module(read)
        self.pipeline.run_module("read2")

        data = self.pipeline.get_data("images")
        assert np.allclose(data[0, 75, 25], 6.921353838812206e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 1.0506056979365338e-06, rtol=limit, atol=0.)
        assert data.shape == (10, 100, 100)

    def test_wrong_tag(self):

        read = Hdf5ReadingModule(name_in="read3",
                                 input_filename="PynPoint_database.hdf5",
                                 input_dir=self.test_dir+"data",
                                 tag_dictionary={"test":"test"})

        self.pipeline.add_module(read)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module("read3")

        assert len(warning) == 1
        assert warning[0].message.args[0] == "The dataset with tag name 'test' is not found in " \
                                             "the HDF5 file."

        h5f = h5py.File(self.test_dir+"data/PynPoint_database.hdf5", "r")
        assert set(h5f.keys()) == set(["extra", "header_extra", "header_images", "images"])
        h5f.close()

    def test_no_input_filename(self):

        read = Hdf5ReadingModule(name_in="read4",
                                 input_filename=None,
                                 input_dir=self.test_dir+"data",
                                 tag_dictionary=None)

        self.pipeline.add_module(read)
        self.pipeline.run_module("read4")

        data = self.pipeline.get_data("images")
        assert np.allclose(data[0, 75, 25], 6.921353838812206e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 1.0506056979365338e-06, rtol=limit, atol=0.)
        assert data.shape == (10, 100, 100)
