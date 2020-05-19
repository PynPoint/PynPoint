import os

import h5py
import pytest
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.attr_writing import ParangWritingModule, AttributeWritingModule
from pynpoint.util.tests import create_config, create_random, remove_test_data


class TestAttributeWriting:

    def setup_class(self) -> None:

        self.limit = 1e-10
        self.test_dir = os.path.dirname(__file__) + '/'

        create_random(self.test_dir)
        create_config(self.test_dir+'PynPoint_config.ini')

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self) -> None:

        remove_test_data(self.test_dir, files=['parang.dat', 'attribute.dat'])

    def test_input_data(self) -> None:

        data = self.pipeline.get_data('images')
        assert np.sum(data) == pytest.approx(0.007153603490533874, rel=self.limit, abs=0.)
        assert data.shape == (5, 11, 11)

    def test_parang_writing(self) -> None:

        module = ParangWritingModule(file_name='parang.dat',
                                     name_in='parang_write1',
                                     output_dir=None,
                                     data_tag='images',
                                     header=None)

        self.pipeline.add_module(module)
        self.pipeline.run_module('parang_write1')

        data = np.loadtxt(self.test_dir+'parang.dat')
        assert np.sum(data) == pytest.approx(10., rel=self.limit, abs=0.)
        assert data.shape == (5, )

    def test_attribute_writing(self) -> None:

        module = AttributeWritingModule(file_name='attribute.dat',
                                        name_in='attr_write1',
                                        output_dir=None,
                                        data_tag='images',
                                        attribute='PARANG',
                                        header=None)

        self.pipeline.add_module(module)
        self.pipeline.run_module('attr_write1')

        data = np.loadtxt(self.test_dir+'attribute.dat')

        assert np.sum(data) == pytest.approx(10., rel=self.limit, abs=0.)
        assert data.shape == (5, )

    def test_attribute_not_present(self) -> None:

        module = AttributeWritingModule(file_name='attribute.dat',
                                        name_in='attr_write3',
                                        output_dir=None,
                                        data_tag='images',
                                        attribute='test',
                                        header=None)

        self.pipeline.add_module(module)

        with pytest.raises(ValueError) as error:
            self.pipeline.run_module('attr_write3')

        assert str(error.value) == 'The \'test\' attribute is not present in \'images\'.'

    def test_parang_writing_not_present(self) -> None:

        with h5py.File(self.test_dir+'PynPoint_database.hdf5', 'a') as hdf_file:
            del hdf_file['header_images/PARANG']

        module = ParangWritingModule(file_name='parang.dat',
                                     name_in='parang_write3',
                                     output_dir=None,
                                     data_tag='images',
                                     header=None)

        self.pipeline.add_module(module)

        with pytest.raises(ValueError) as error:
            self.pipeline.run_module('parang_write3')

        assert str(error.value) == 'The PARANG attribute is not present in \'images\'.'
