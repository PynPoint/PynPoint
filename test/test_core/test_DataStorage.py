import os
import numpy as np
import pytest

from PynPoint.core.DataIO import DataStorage


class TestDataStorage(object):

    def setup(self):
        self.test_data_dir = (os.path.dirname(__file__)) + '/test_data/'

    def test_create_storage_with_existing_database(self):
        file_in = self.test_data_dir + "init/PynPoint_database.hdf5"
        storage = DataStorage(file_in)

        storage.open_connection()

        data = storage.m_data_bank["im_arr"]

        assert data[0, 0, 0] == 27.113279943585397
        assert np.mean(data) == 467.20439057377075

    def test_create_storage_without_existing_database(self):
        file_in = self.test_data_dir + "new/PynPoint_database.hdf5"

        storage = DataStorage(file_in)

        storage.open_connection()

        storage.m_data_bank["data"] = [0, 1, 2, 5, 7]

        assert storage.m_data_bank["data"][2] == 2
        assert storage.m_data_bank.keys() == ["data", ]

        storage.close_connection()

        os.remove(file_in)

    def test_create_storage_with_wrong_location(self):
        file_in = "/bla/test.hdf5"

        with pytest.raises(AssertionError):
            storage = DataStorage(file_in)

    def test_open_close_connection(self):
        file_in = self.test_data_dir + "init/PynPoint_database.hdf5"
        storage = DataStorage(file_in)

        storage.open_connection()

        assert storage.m_open is True

        storage.open_connection()

        assert storage.m_open is True

        storage.close_connection()

        assert storage.m_open is False

        storage.close_connection()

        assert storage.m_open is False
