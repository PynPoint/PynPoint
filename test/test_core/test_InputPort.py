import os
import numpy as np
import pytest

from PynPoint.core.DataIO import InputPort, DataStorage


class TestInputPort(object):

    def setup(self):
        self.test_data_dir = (os.path.dirname(__file__)) + '/test_data/'

        dir_in = self.test_data_dir + "init/PynPoint_database.hdf5"
        self.storage = DataStorage(dir_in)

    def test_create_instance_access_data(self):

        port = InputPort("im_arr", self.storage)

        assert port[0, 0, 0] == 27.113279943585397
        assert np.mean(port.get_all()) == 467.20439057377075
        assert len(port[0:2, 0, 0]) == 2
        assert np.array_equal(port[0:3, 0, 0],
                              np.asarray([27.113279943585397,
                                          21.151920002341271,
                                          19.147920089185238],
                                         dtype=np.float64))
        assert port.get_shape() == (47, 146, 146)

        # attributes
        assert port.get_attribute("num_files") == 47
        assert port.get_attribute("NEW_PARA")[0] == -11.961975200000001

        with pytest.warns(UserWarning):
            assert port.get_attribute("none") is None

    def test_create_instance_access_non_existing_data(self):

        port = InputPort("bla", self.storage)

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

        port = InputPort("bla")

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

        port = InputPort('im_arr', self.storage)

        assert port.get_all_static_attributes() == {'num_files': 47}

        assert port.get_all_non_static_attributes() == ['NEW_PARA', 'Used_Files']
