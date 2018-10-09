import os
import warnings

import numpy as np

from PynPoint.Core.Pypeline import Pypeline
from PynPoint.IOmodules.Hdf5Reading import Hdf5ReadingModule
from PynPoint.Util.TestTools import create_config, create_random, remove_test_data

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

        read = Hdf5ReadingModule(name_in="read",
                                 input_filename="PynPoint_database.hdf5",
                                 input_dir=self.test_dir+"data",
                                 tag_dictionary={"images":"images"})

        self.pipeline.add_module(read)
        self.pipeline.run_module("read")

        data = self.pipeline.get_data("images")
        assert np.allclose(data[0, 75, 25], 6.921353838812206e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 1.0506056979365338e-06, rtol=limit, atol=0.)
        assert data.shape == (10, 100, 100)
