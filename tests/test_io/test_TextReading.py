import os
import warnings

import pytest
import numpy as np

from PynPoint.Core.Pypeline import Pypeline
from PynPoint.IOmodules.TextReading import ParangReadingModule, \
                                           AttributeReadingModule
from PynPoint.Util.TestTools import create_config, create_random, remove_test_data

warnings.simplefilter("always")

limit = 1e-10

class TestTextReading(object):

    def setup_class(self):

        self.test_dir = os.path.dirname(__file__) + "/"

        create_random(self.test_dir, ndit=10, parang=None)
        create_config(self.test_dir+"PynPoint_config.ini")

        np.savetxt(self.test_dir+"parang.dat", np.arange(1., 11., 1.))
        np.savetxt(self.test_dir+"attribute.dat", np.arange(1, 11, 1), fmt='%i')

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self):

        remove_test_data(self.test_dir, files=["parang.dat", "attribute.dat"])

    def test_input_data(self):

        data = self.pipeline.get_data("images")
        assert np.allclose(data[0, 75, 25], 6.921353838812206e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 1.0506056979365338e-06, rtol=limit, atol=0.)
        assert data.shape == (10, 100, 100)

    def test_parang_reading(self):

        parang = ParangReadingModule(file_name="parang.dat",
                                     name_in="parang1",
                                     input_dir=None,
                                     data_tag="images",
                                     overwrite=False)

        self.pipeline.add_module(parang)
        self.pipeline.run_module("parang1")

        data = self.pipeline.get_data("header_images/PARANG")
        assert data.dtype == 'float64'
        assert np.allclose(data, np.arange(1., 11., 1.), rtol=limit, atol=0.)
        assert data.shape == (10, )

        parang = ParangReadingModule(file_name="parang.dat",
                                     name_in="parang2",
                                     input_dir=None,
                                     data_tag="images",
                                     overwrite=False)

        self.pipeline.add_module(parang)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module("parang2")

        assert len(warning) == 1
        assert warning[0].message.args[0] == "The PARANG attribute is already present. Set the " \
                                             "overwrite argument to True in order to overwrite " \
                                             "the values with parang.dat."

    def test_attribute_reading(self):

        attribute = AttributeReadingModule(file_name="attribute.dat",
                                           attribute="EXP_NO",
                                           name_in="attribute1",
                                           input_dir=None,
                                           data_tag="images",
                                           overwrite=False)

        self.pipeline.add_module(attribute)
        self.pipeline.run_module("attribute1")

        data = self.pipeline.get_data("header_images/EXP_NO")
        assert data.dtype == 'int64'
        assert np.allclose(data, np.arange(1, 11, 1), rtol=limit, atol=0.)
        assert data.shape == (10, )

        attribute = AttributeReadingModule(file_name="attribute.dat",
                                           attribute="EXP_NO",
                                           name_in="attribute2",
                                           input_dir=None,
                                           data_tag="images",
                                           overwrite=False)

        self.pipeline.add_module(attribute)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module("attribute2")

        assert len(warning) == 1
        assert warning[0].message.args[0] == "The attribute 'EXP_NO' is already present. Set the " \
                                             "overwrite argument to True in order to overwrite " \
                                             "the values with attribute.dat."
