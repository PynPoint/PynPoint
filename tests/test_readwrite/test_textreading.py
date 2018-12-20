from __future__ import absolute_import
import os
import warnings

import pytest
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.textreading import ParangReadingModule, AttributeReadingModule
from pynpoint.util.tests import create_config, create_random, remove_test_data

warnings.simplefilter("always")

limit = 1e-10

class TestTextReading(object):

    def setup_class(self):

        self.test_dir = os.path.dirname(__file__) + "/"

        create_random(self.test_dir, ndit=10, parang=None)
        create_config(self.test_dir+"PynPoint_config.ini")

        np.savetxt(self.test_dir+"parang.dat", np.arange(1., 11., 1.))
        np.savetxt(self.test_dir+"new.dat", np.arange(10., 21., 1.))
        np.savetxt(self.test_dir+"attribute.dat", np.arange(1, 11, 1), fmt='%i')

        data2d = np.random.normal(loc=0, scale=2e-4, size=(10, 10))
        np.savetxt(self.test_dir+"data_2d.dat", data2d)

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self):

        remove_test_data(self.test_dir, files=["parang.dat", "new.dat",
                                               "attribute.dat", "data_2d.dat"])

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

    def test_parang_reading_same(self):

        parang = ParangReadingModule(file_name="parang.dat",
                                     name_in="parang2",
                                     input_dir=None,
                                     data_tag="images",
                                     overwrite=True)

        self.pipeline.add_module(parang)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module("parang2")

        assert len(warning) == 1
        assert warning[0].message.args[0] == "The PARANG attribute is already present and " \
                                             "contains the same values as are present in " \
                                             "parang.dat."

    def test_parang_reading_present(self):

        parang = ParangReadingModule(file_name="new.dat",
                                     name_in="parang3",
                                     input_dir=None,
                                     data_tag="images",
                                     overwrite=False)

        self.pipeline.add_module(parang)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module("parang3")

        assert len(warning) == 1
        assert warning[0].message.args[0] == "The PARANG attribute is already present. Set the " \
                                             "'overwrite' parameter to True in order to " \
                                             "overwrite the values with new.dat."

    def test_parang_reading_overwrite(self):

        parang = ParangReadingModule(file_name="new.dat",
                                     name_in="parang4",
                                     input_dir=None,
                                     data_tag="images",
                                     overwrite=True)

        self.pipeline.add_module(parang)
        self.pipeline.run_module("parang4")

    def test_parang_reading_string(self):

        with pytest.raises(ValueError) as error:
            ParangReadingModule(file_name=0.,
                                name_in="parang5",
                                input_dir=None,
                                data_tag="images",
                                overwrite=False)

        assert str(error.value) == "Input 'file_name' needs to be a string."

    def test_parang_reading_2d(self):

        parang = ParangReadingModule(file_name="data_2d.dat",
                                     name_in="parang6",
                                     input_dir=None,
                                     data_tag="images",
                                     overwrite=False)

        self.pipeline.add_module(parang)

        with pytest.raises(ValueError) as error:
            self.pipeline.run_module("parang6")

        assert str(error.value) == "The input file data_2d.dat should contain a 1D data set with " \
                                   "the parallactic angles."

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

    def test_attribute_reading_string(self):

        with pytest.raises(ValueError) as error:
            AttributeReadingModule(file_name=0.,
                                   attribute="EXP_NO",
                                   name_in="attribute2",
                                   input_dir=None,
                                   data_tag="images",
                                   overwrite=False)

        assert str(error.value) == "Input 'file_name' needs to be a string."

    def test_attribute_reading_present(self):

        attribute = AttributeReadingModule(file_name="parang.dat",
                                           attribute="PARANG",
                                           name_in="attribute3",
                                           input_dir=None,
                                           data_tag="images",
                                           overwrite=False)

        self.pipeline.add_module(attribute)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module("attribute3")

        assert warning[0].message.args[0] == "The attribute 'PARANG' is already present. Set " \
                                             "the 'overwrite' parameter to True in order to " \
                                             "overwrite the values with parang.dat."

    def test_attribute_reading_invalid(self):

        attribute = AttributeReadingModule(file_name="attribute.dat",
                                           attribute="test",
                                           name_in="attribute4",
                                           input_dir=None,
                                           data_tag="images",
                                           overwrite=False)

        self.pipeline.add_module(attribute)

        with pytest.raises(ValueError) as error:
            self.pipeline.run_module("attribute4")

        assert str(error.value) == "'test' is not a valid attribute."

    def test_attribute_reading_2d(self):

        attribute = AttributeReadingModule(file_name="data_2d.dat",
                                           attribute="DITHER_X",
                                           name_in="attribute5",
                                           input_dir=None,
                                           data_tag="images",
                                           overwrite=False)

        self.pipeline.add_module(attribute)

        with pytest.raises(ValueError) as error:
            self.pipeline.run_module("attribute5")

        assert str(error.value) == "The input file data_2d.dat should contain a 1D list with " \
                                   "attributes."

    def test_attribute_reading_same(self):

        attribute = AttributeReadingModule(file_name="attribute.dat",
                                           attribute="EXP_NO",
                                           name_in="attribute6",
                                           input_dir=None,
                                           data_tag="images",
                                           overwrite=True)

        self.pipeline.add_module(attribute)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module("attribute6")

        assert len(warning) == 1
        assert warning[0].message.args[0] == "The 'EXP_NO' attribute is already present and " \
                                             "contains the same values as are present in " \
                                             "attribute.dat."

    def test_attribute_reading_overwrite(self):

        attribute = AttributeReadingModule(file_name="parang.dat",
                                           attribute="PARANG",
                                           name_in="attribute7",
                                           input_dir=None,
                                           data_tag="images",
                                           overwrite=True)

        self.pipeline.add_module(attribute)
        self.pipeline.run_module("attribute7")

        attribute = self.pipeline.get_attribute("images", "PARANG", static=False)
        assert np.allclose(attribute, np.arange(1., 11., 1.), rtol=limit, atol=0.)
