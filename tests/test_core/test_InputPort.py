import os
import warnings

import pytest
import h5py
import numpy as np

from PynPoint.Core.DataIO import InputPort, DataStorage

warnings.simplefilter("always")

limit = 1e-10

def setup_module():
    file_in = os.path.dirname(__file__) + "/PynPoint_database.hdf5"

    np.random.seed(1)
    images = np.random.normal(loc=0, scale=2e-4, size=(10, 100, 100))
    parang = np.arange(1, 11, 1)

    h5f = h5py.File(file_in, "w")
    dset = h5f.create_dataset("images", data=images)
    dset.attrs['PIXSCALE'] = 0.01
    h5f.create_dataset("header_images/PARANG", data=parang)
    h5f.close()

def teardown_module():
    file_in = os.path.dirname(__file__) + "/PynPoint_database.hdf5"

    os.remove(file_in)

class TestInputPort(object):

    def setup(self):
        file_in = os.path.dirname(__file__) + "/PynPoint_database.hdf5"

        self.storage = DataStorage(file_in)

    def test_create_instance_access_data(self):
        port = InputPort("images", self.storage)

        assert np.allclose(port[0, 0, 0], 0.00032486907273264834, rtol=limit)
        assert np.allclose(np.mean(port.get_all()), 1.0506056979365338e-06, rtol=limit)

        arr_tmp = np.asarray((0.00032486907273264834, -2.4494781298462809e-05, -0.00038631277795631806), dtype=np.float64)
        assert np.allclose(port[0:3, 0, 0], arr_tmp, rtol=limit)

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
