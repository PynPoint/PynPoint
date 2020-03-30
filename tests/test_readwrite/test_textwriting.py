import os
import warnings

import h5py
import pytest
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.textwriting import TextWritingModule
from pynpoint.util.tests import create_config, create_random, remove_test_data

warnings.simplefilter('always')

limit = 1e-10


class TestTextWriting:

    def setup_class(self) -> None:

        self.test_dir = os.path.dirname(__file__) + '/'

        create_random(self.test_dir, ndit=1)
        create_config(self.test_dir+'PynPoint_config.ini')

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self) -> None:

        remove_test_data(self.test_dir, files=['image.dat', 'data.dat'])

    def test_input_data(self) -> None:

        data = self.pipeline.get_data('images')
        assert np.allclose(data[0, 75, 25], 6.921353838812206e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 1.9545313398209947e-06, rtol=limit, atol=0.)
        assert data.shape == (1, 100, 100)

    def test_text_writing(self) -> None:

        text_write = TextWritingModule(file_name='image.dat',
                                       name_in='text_write',
                                       output_dir=None,
                                       data_tag='images',
                                       header=None)

        self.pipeline.add_module(text_write)
        self.pipeline.run_module('text_write')

        data = np.loadtxt(self.test_dir+'image.dat')

        assert np.allclose(data[75, 25], 6.921353838812206e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 1.9545313398209947e-06, rtol=limit, atol=0.)
        assert data.shape == (100, 100)

    def test_text_writing_ndim(self) -> None:

        data_4d = np.random.normal(loc=0, scale=2e-4, size=(5, 5, 5, 5))

        with h5py.File(self.test_dir+'PynPoint_database.hdf5', 'a') as hdf_file:
            hdf_file.create_dataset('data_4d', data=data_4d)

        text_write = TextWritingModule(file_name='data.dat',
                                       name_in='write_4d',
                                       output_dir=None,
                                       data_tag='data_4d',
                                       header=None)

        self.pipeline.add_module(text_write)

        with pytest.raises(ValueError) as error:
            self.pipeline.run_module('write_4d')

        assert str(error.value) == 'Only 1D or 2D arrays can be written to a text file.'

    def test_text_writing_int(self) -> None:

        data_int = np.arange(1, 101, 1)

        with h5py.File(self.test_dir+'PynPoint_database.hdf5', 'a') as hdf_file:
            hdf_file.create_dataset('data_int', data=data_int)

        text_write = TextWritingModule(file_name='data.dat',
                                       name_in='write_int',
                                       output_dir=None,
                                       data_tag='data_int',
                                       header=None)

        self.pipeline.add_module(text_write)
        self.pipeline.run_module('write_int')

        data = np.loadtxt(self.test_dir+'data.dat')

        assert np.allclose(data, data_int, rtol=limit, atol=0.)
        assert data.shape == (100, )
