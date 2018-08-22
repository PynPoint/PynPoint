import os
import warnings

import numpy as np

from PynPoint.Core.Pypeline import Pypeline
from PynPoint.IOmodules.TextReading import ParangReadingModule
from PynPoint.Util.TestTools import create_config, create_random, remove_test_data

warnings.simplefilter("always")

limit = 1e-10

class TestTextReading(object):

    def setup_class(self):

        self.test_dir = os.path.dirname(__file__) + "/"

        create_random(self.test_dir, ndit=10, parang=None)

        create_config(self.test_dir+"PynPoint_config.ini")

        np.savetxt(self.test_dir+"parang.dat", np.arange(1., 11., 1.))

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self):

        remove_test_data(self.test_dir, files=["parang.dat"])

    def test_input_data(self):

        data = self.pipeline.get_data("images")
        assert np.allclose(data[0, 75, 25], 6.921353838812206e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 1.0506056979365338e-06, rtol=limit, atol=0.)
        assert data.shape == (10, 100, 100)

    def test_parang_reading(self):

        parang = ParangReadingModule(file_name="parang.dat",
                                     name_in="parang",
                                     input_dir=None,
                                     data_tag="images",
                                     overwrite=False)

        self.pipeline.add_module(parang)
        self.pipeline.run_module("parang")

        data = self.pipeline.get_data("header_images/PARANG")
        assert data[0] == 1.
        assert data[9] == 10.
        assert np.allclose(np.mean(data), 5.5, rtol=limit, atol=0.)
        assert data.shape == (10, )
