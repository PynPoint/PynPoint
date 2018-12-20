from __future__ import absolute_import

import os
import warnings

import pytest
import h5py
import numpy as np

from pynpoint.core.dataio import DataStorage

warnings.simplefilter("always")

limit = 1e-10

class TestDataStorage(object):

    def setup(self):
        self.test_data = os.path.dirname(__file__) + "/PynPoint_database.hdf5"

    def test_create_storage_with_existing_database(self):
        np.random.seed(1)
        images = np.random.normal(loc=0, scale=2e-4, size=(10, 100, 100))

        h5f = h5py.File(self.test_data, "w")
        h5f.create_dataset("images", data=images)
        h5f.close()

        storage = DataStorage(self.test_data)
        storage.open_connection()
        data = storage.m_data_bank["images"]

        assert np.allclose(data[0, 0, 0], 0.00032486907273264834, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 1.0506056979365338e-06, rtol=limit, atol=0.)

        os.remove(self.test_data)

    def test_create_storage_without_existing_database(self):
        storage = DataStorage(self.test_data)
        storage.open_connection()
        storage.m_data_bank["data"] = [0, 1, 2, 5, 7]

        assert storage.m_data_bank["data"][2] == 2
        assert list(storage.m_data_bank.keys()) == ["data", ]

        storage.close_connection()

        os.remove(self.test_data)

    def test_create_storage_with_wrong_location(self):
        file_in = "/test/test.hdf5"

        with pytest.raises(AssertionError):
            DataStorage(file_in)

    def test_open_close_connection(self):
        storage = DataStorage(self.test_data)

        storage.open_connection()
        assert storage.m_open is True

        storage.open_connection()
        assert storage.m_open is True

        storage.close_connection()
        assert storage.m_open is False

        storage.close_connection()
        assert storage.m_open is False

        os.remove(self.test_data)
