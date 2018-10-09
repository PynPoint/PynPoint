import os
import warnings

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
                                  name_in="write",
                                  output_dir=None,
                                  tag_dictionary={"images":"test"},
                                  keep_attributes=True,
                                  overwrite=False)

        self.pipeline.add_module(write)
        self.pipeline.run_module("write")

    def test_hdf5_reading(self):

        read = Hdf5ReadingModule(name_in="read",
                                 input_filename="test.hdf5",
                                 input_dir=self.test_dir,
                                 tag_dictionary={"test":"test"})

        self.pipeline.add_module(read)
        self.pipeline.run_module("read")

        data1 = self.pipeline.get_data("images")
        data2 = self.pipeline.get_data("test")
        assert np.allclose(data1, data2, rtol=limit, atol=0.)

        attribute1 = self.pipeline.get_attribute("images", "PARANG", static=False)
        attribute2 = self.pipeline.get_attribute("test", "PARANG", static=False)
        assert np.allclose(attribute1, attribute2, rtol=limit, atol=0.)
