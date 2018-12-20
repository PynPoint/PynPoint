from __future__ import absolute_import

import os
import warnings

import pytest
import h5py
import numpy as np

from pynpoint.core.dataio import DataStorage, InputPort, OutputPort
from pynpoint.util.tests import create_random, create_config, remove_test_data 

warnings.simplefilter("always")

limit = 1e-10

class TestInputPort(object):

    def setup_class(self):
        self.test_dir = os.path.dirname(__file__) + "/"
        create_random(self.test_dir)
        create_config(self.test_dir+"PynPoint_config.ini")

    def teardown_class(self):
        remove_test_data(self.test_dir)

    def setup(self):
        file_in = os.path.dirname(__file__) + "/PynPoint_database.hdf5"

        self.storage = DataStorage(file_in)

    def test_create_instance_access_data(self):
        with pytest.raises(ValueError) as error:
            InputPort("config", self.storage)

        assert str(error.value) == "The tag name 'config' is reserved for the central " \
                                   "configuration of PynPoint."

        with pytest.raises(ValueError) as error:
            InputPort("fits_header", self.storage)

        assert str(error.value) == "The tag name 'fits_header' is reserved for storage of the " \
                                   "FITS headers."

        port = InputPort("images", self.storage)

        assert np.allclose(port[0, 0, 0], 0.00032486907273264834, rtol=limit, atol=0.)
        assert np.allclose(np.mean(port.get_all()), 1.0506056979365338e-06, rtol=limit, atol=0.)

        arr_tmp = np.asarray((0.00032486907273264834, -2.4494781298462809e-05,
                              -0.00038631277795631806), dtype=np.float64)

        assert np.allclose(port[0:3, 0, 0], arr_tmp, rtol=limit, atol=0.)

        assert len(port[0:2, 0, 0]) == 2
        assert port.get_shape() == (10, 100, 100)

        assert port.get_attribute("PIXSCALE") == 0.01
        assert port.get_attribute("PARANG")[0] == 1

        with pytest.warns(UserWarning):
            assert port.get_attribute("none") is None

    def test_create_instance_access_non_existing_data(self):
        port = InputPort("test", self.storage)

        with pytest.warns(UserWarning):
            assert port[0, 0, 0] is None

        with pytest.warns(UserWarning):
            assert port.get_all() is None

        with pytest.warns(UserWarning):
            assert port.get_shape() is None

        with pytest.warns(UserWarning):
            assert port.get_attribute("num_files") is None

        with pytest.warns(UserWarning):
            assert port.get_all_non_static_attributes() is None

        with pytest.warns(UserWarning):
            assert port.get_all_static_attributes() is None

    def test_create_instance_no_data_storage(self):
        port = InputPort("test")

        with pytest.warns(UserWarning):
            assert port[0, 0, 0] is None

        with pytest.warns(UserWarning):
            assert port.get_all() is None

        with pytest.warns(UserWarning):
            assert port.get_shape() is None

        with pytest.warns(UserWarning):
            assert port.get_all_non_static_attributes() is None

        with pytest.warns(UserWarning):
            assert port.get_all_static_attributes() is None

    def test_get_all_attributes(self):
        port = InputPort('images', self.storage)

        assert port.get_all_static_attributes() == {'PIXSCALE': 0.01}
        assert port.get_all_non_static_attributes() == ['PARANG', ]

        port = OutputPort('images', self.storage)
        assert port.del_all_attributes() is None

        port = InputPort('images', self.storage)
        assert port.get_all_non_static_attributes() is None

    def test_get_ndim(self):
        with pytest.warns(UserWarning) as warning:
            ndim = InputPort('images', None).get_ndim()

        assert len(warning) == 1
        assert warning[0].message.args[0] == "InputPort can not load data unless a database is " \
                                             "connected."

        assert ndim is None

        port = InputPort('images', self.storage)
        assert port.get_ndim() == 3
