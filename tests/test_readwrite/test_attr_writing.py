import os
import warnings

import h5py
import pytest
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.attr_writing import ParangWritingModule, AttributeWritingModule
from pynpoint.util.tests import create_config, create_random, remove_test_data

warnings.simplefilter('always')

limit = 1e-10


class TestAttributeWriting:

    def setup_class(self) -> None:

        self.test_dir = os.path.dirname(__file__) + '/'

        create_random(self.test_dir, ndit=1)
        create_config(self.test_dir+'PynPoint_config.ini')

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self) -> None:

        remove_test_data(self.test_dir, files=['parang.dat', 'attribute.dat'])

    def test_input_data(self) -> None:

        data = self.pipeline.get_data('images')
        assert np.allclose(data[0, 75, 25], 6.921353838812206e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 1.9545313398209947e-06, rtol=limit, atol=0.)
        assert data.shape == (1, 100, 100)

    def test_parang_writing(self) -> None:

        parang_write = ParangWritingModule(file_name='parang.dat',
                                           name_in='parang_write1',
                                           output_dir=None,
                                           data_tag='images',
                                           header=None)

        self.pipeline.add_module(parang_write)
        self.pipeline.run_module('parang_write1')

        data = np.loadtxt(self.test_dir+'parang.dat')

        assert np.allclose(data[0], 1.0, rtol=limit, atol=0.)
        assert np.allclose(data[9], 10.0, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 5.5, rtol=limit, atol=0.)
        assert data.shape == (10, )

    def test_attribute_writing(self) -> None:

        attr_write = AttributeWritingModule(file_name='attribute.dat',
                                            name_in='attr_write1',
                                            output_dir=None,
                                            data_tag='images',
                                            attribute='PARANG',
                                            header=None)

        self.pipeline.add_module(attr_write)
        self.pipeline.run_module('attr_write1')

        data = np.loadtxt(self.test_dir+'attribute.dat')

        assert np.allclose(data[0], 1.0, rtol=limit, atol=0.)
        assert np.allclose(data[9], 10.0, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 5.5, rtol=limit, atol=0.)
        assert data.shape == (10, )

    def test_attribute_not_present(self) -> None:

        attr_write = AttributeWritingModule(file_name='attribute.dat',
                                            name_in='attr_write3',
                                            output_dir=None,
                                            data_tag='images',
                                            attribute='test',
                                            header=None)

        self.pipeline.add_module(attr_write)

        with pytest.raises(ValueError) as error:
            self.pipeline.run_module('attr_write3')

        assert str(error.value) == 'The \'test\' attribute is not present in \'images\'.'

    def test_parang_writing_not_present(self) -> None:

        with h5py.File(self.test_dir+'PynPoint_database.hdf5', 'a') as hdf_file:
            del hdf_file['header_images/PARANG']

        parang_write = ParangWritingModule(file_name='parang.dat',
                                           name_in='parang_write3',
                                           output_dir=None,
                                           data_tag='images',
                                           header=None)

        self.pipeline.add_module(parang_write)

        with pytest.raises(ValueError) as error:
            self.pipeline.run_module('parang_write3')

        assert str(error.value) == 'The PARANG attribute is not present in \'images\'.'
