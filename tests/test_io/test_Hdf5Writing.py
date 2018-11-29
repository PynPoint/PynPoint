from __future__ import absolute_import
import os
import warnings

import pytest
import numpy as np

from PynPoint.Core.Pypeline import Pypeline
from PynPoint.IOmodules.Hdf5Reading import Hdf5ReadingModule
from PynPoint.IOmodules.Hdf5Writing import Hdf5WritingModule
from PynPoint.Util.TestTools import create_config, create_random, remove_test_data

warnings.simplefilter("always")

limit = 1e-10

class TestHdf5WritingModule(object):

    def setup_class(self):

        self.test_dir = os.path.dirname(__file__) + "/"

        create_random(self.test_dir)
        create_config(self.test_dir+"PynPoint_config.ini")

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self):

        remove_test_data(self.test_dir, files=["test.hdf5"])

    def test_hdf5_writing(self):

        write = Hdf5WritingModule(file_name="test.hdf5",
                                  name_in="write1",
                                  output_dir=None,
                                  tag_dictionary={"images":"data1"},
                                  keep_attributes=True,
                                  overwrite=True)

        self.pipeline.add_module(write)
        self.pipeline.run_module("write1")

    def test_no_data_tag(self):

        write = Hdf5WritingModule(file_name="test.hdf5",
                                  name_in="write2",
                                  output_dir=None,
                                  tag_dictionary={"empty":"empty"},
                                  keep_attributes=True,
                                  overwrite=False)

        self.pipeline.add_module(write)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module("write2")

        assert len(warning) == 1
        assert warning[0].message.args[0] == "No data under the tag which is linked by the " \
                                             "InputPort."

    def test_overwrite_false(self):

        write = Hdf5WritingModule(file_name="test.hdf5",
                                  name_in="write3",
                                  output_dir=None,
                                  tag_dictionary={"images":"data2"},
                                  keep_attributes=True,
                                  overwrite=False)

        self.pipeline.add_module(write)
        self.pipeline.run_module("write3")

    def test_dictionary_none(self):

        write = Hdf5WritingModule(file_name="test.hdf5",
                                  name_in="write4",
                                  output_dir=None,
                                  tag_dictionary=None,
                                  keep_attributes=True,
                                  overwrite=False)

        self.pipeline.add_module(write)
        self.pipeline.run_module("write4")

    def test_hdf5_reading(self):

        read = Hdf5ReadingModule(name_in="read",
                                 input_filename="test.hdf5",
                                 input_dir=self.test_dir,
                                 tag_dictionary={"data1":"data1", "data2":"data2"})

        self.pipeline.add_module(read)
        self.pipeline.run_module("read")

        data1 = self.pipeline.get_data("data1")
        data2 = self.pipeline.get_data("data2")
        data3 = self.pipeline.get_data("images")
        assert np.allclose(data1, data2, rtol=limit, atol=0.)
        assert np.allclose(data2, data3, rtol=limit, atol=0.)

        attribute1 = self.pipeline.get_attribute("images", "PARANG", static=False)
        attribute2 = self.pipeline.get_attribute("data1", "PARANG", static=False)
        attribute3 = self.pipeline.get_attribute("data2", "PARANG", static=False)
        assert np.allclose(attribute1, attribute2, rtol=limit, atol=0.)
        assert np.allclose(attribute2, attribute3, rtol=limit, atol=0.)
