import os

import h5py
import pytest
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.fitsreading import FitsReadingModule
from pynpoint.processing.extract import StarExtractionModule, ExtractBinaryModule
from pynpoint.util.tests import create_config, create_star_data, create_fake_data, remove_test_data


class TestExtract:

    def setup_class(self) -> None:

        self.limit = 1e-10
        self.test_dir = os.path.dirname(__file__) + '/'

        create_star_data(self.test_dir+'star')
        create_fake_data(self.test_dir+'binary')
        create_config(self.test_dir+'PynPoint_config.ini')

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self) -> None:

        remove_test_data(path=self.test_dir, folders=['star', 'binary'])

    def test_read_data(self) -> None:

        module = FitsReadingModule(name_in='read1',
                                   image_tag='star',
                                   input_dir=self.test_dir+'star',
                                   overwrite=True,
                                   check=True)

        self.pipeline.add_module(module)
        self.pipeline.run_module('read1')

        data = self.pipeline.get_data('star')
        assert np.sum(data) == pytest.approx(105.54278879805277, rel=self.limit, abs=0.)
        assert data.shape == (10, 11, 11)

        module = FitsReadingModule(name_in='read2',
                                   image_tag='binary',
                                   input_dir=self.test_dir+'binary',
                                   overwrite=True,
                                   check=True)

        self.pipeline.add_module(module)
        self.pipeline.run_module('read2')

        data = self.pipeline.get_data('binary')
        assert np.sum(data) == pytest.approx(11.012854046962481, rel=self.limit, abs=0.)
        assert data.shape == (10, 21, 21)

        self.pipeline.set_attribute('binary', 'PARANG', -1.*np.linspace(0., 180., 10), static=False)

    def test_extract_position_none(self) -> None:

        module = StarExtractionModule(name_in='extract1',
                                      image_in_tag='star',
                                      image_out_tag='extract1',
                                      index_out_tag='index',
                                      image_size=0.2,
                                      fwhm_star=0.1,
                                      position=None)

        self.pipeline.add_module(module)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module('extract1')

        assert len(warning) == 2

        assert warning[0].message.args[0] == 'The new dataset that is stored under the tag name ' \
                                             '\'index\' is empty.'

        assert warning[1].message.args[0] == 'The new dataset that is stored under the tag name ' \
                                             '\'header_extract1/STAR_POSITION\' is empty.'

        data = self.pipeline.get_data('extract1')
        assert np.sum(data) == pytest.approx(104.93318507061295, rel=self.limit, abs=0.)
        assert data.shape == (10, 9, 9)

        # attr = self.pipeline.get_attribute('extract1', 'STAR_POSITION', static=False)
        # assert np.sum(attr) == pytest.approx(100, rel=self.limit, abs=0.)
        # assert attr.shape == (10, 2)

    def test_extract_center_none(self) -> None:

        module = StarExtractionModule(name_in='extract2',
                                      image_in_tag='star',
                                      image_out_tag='extract2',
                                      index_out_tag='index',
                                      image_size=0.2,
                                      fwhm_star=0.1,
                                      position=(None, None, 0.2))

        self.pipeline.add_module(module)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module('extract2')

        assert len(warning) == 2

        assert warning[0].message.args[0] == 'The new dataset that is stored under the tag name ' \
                                             '\'index\' is empty.'

        assert warning[1].message.args[0] == 'The new dataset that is stored under the tag name ' \
                                             '\'header_extract2/STAR_POSITION\' is empty.'

        data = self.pipeline.get_data('extract2')
        assert np.sum(data) == pytest.approx(104.93318507061295, rel=self.limit, abs=0.)
        assert data.shape == (10, 9, 9)

        # attr = self.pipeline.get_attribute('extract2', 'STAR_POSITION', static=False)
        # assert np.sum(attr) == pytest.approx(100, rel=self.limit, abs=0.)
        # assert attr.shape == (10, 2)

    def test_extract_position(self) -> None:

        module = StarExtractionModule(name_in='extract7',
                                      image_in_tag='star',
                                      image_out_tag='extract7',
                                      index_out_tag=None,
                                      image_size=0.2,
                                      fwhm_star=0.1,
                                      position=(5, 5, 0.2))

        self.pipeline.add_module(module)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module('extract7')

        assert len(warning) == 1

        assert warning[0].message.args[0] == 'The new dataset that is stored under the tag name ' \
                                             '\'header_extract7/STAR_POSITION\' is empty.'

        data = self.pipeline.get_data('extract7')
        assert np.sum(data) == pytest.approx(104.93318507061295, rel=self.limit, abs=0.)
        assert data.shape == (10, 9, 9)

        # attr = self.pipeline.get_attribute('extract7', 'STAR_POSITION', static=False)
        # assert np.sum(attr) == pytest.approx(100, rel=self.limit, abs=0.)
        # assert attr.shape == (10, 2)

    def test_extract_too_large(self) -> None:

        module = StarExtractionModule(name_in='extract3',
                                      image_in_tag='star',
                                      image_out_tag='extract3',
                                      index_out_tag=None,
                                      image_size=0.2,
                                      fwhm_star=0.1,
                                      position=(2, 2, 0.05))

        self.pipeline.add_module(module)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module('extract3')

        assert len(warning) == 11

        # assert warning[0].message.args[0] == f'Chosen image size is too large to crop the image ' \
        #                                      f'around the brightest pixel (image index = 0, ' \
        #                                      f'pixel [x, y] = [2, 2]). Using the center of ' \
        #                                      f'the image instead.'

        # assert warning[0].message.args[0] == 'Chosen image size is too large to crop the image ' \
        #                                      'brightest pixel. Using the center of the image ' \
        #                                      'instead.'

        data = self.pipeline.get_data('extract3')
        assert np.sum(data) == pytest.approx(104.93318507061295, rel=self.limit, abs=0.)
        assert data.shape == (10, 9, 9)

        # attr = self.pipeline.get_attribute('extract3', 'STAR_POSITION', static=False)
        # assert np.sum(attr) == pytest.approx(100, rel=self.limit, abs=0.)
        # assert attr.shape == (10, 2)

    def test_star_extract_cpu(self) -> None:

        with h5py.File(self.test_dir+'PynPoint_database.hdf5', 'a') as hdf_file:
            hdf_file['config'].attrs['CPU'] = 4

        module = StarExtractionModule(name_in='extract4',
                                      image_in_tag='star',
                                      image_out_tag='extract4',
                                      index_out_tag='index',
                                      image_size=0.2,
                                      fwhm_star=0.1,
                                      position=(2, 2, 0.05))

        self.pipeline.add_module(module)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module('extract4')

        assert len(warning) == 1

        assert warning[0].message.args[0] == 'Chosen image size is too large to crop the image ' \
                                             'around the brightest pixel. Using the center of ' \
                                             'the image instead.'

    def test_extract_binary(self) -> None:

        with h5py.File(self.test_dir+'PynPoint_database.hdf5', 'a') as hdf_file:
            hdf_file['config'].attrs['CPU'] = 1

        module = ExtractBinaryModule(pos_center=(10., 10.),
                                     pos_binary=(10., 16.),
                                     name_in='extract5',
                                     image_in_tag='binary',
                                     image_out_tag='extract5',
                                     image_size=0.15,
                                     search_size=0.07,
                                     filter_size=None)

        self.pipeline.add_module(module)
        self.pipeline.run_module('extract5')

        data = self.pipeline.get_data('extract5')
        assert np.sum(data) == pytest.approx(1.3419098759577548, rel=self.limit, abs=0.)
        assert data.shape == (10, 7, 7)

    def test_extract_binary_filter(self) -> None:

        module = ExtractBinaryModule(pos_center=(10., 10.),
                                     pos_binary=(10., 16.),
                                     name_in='extract6',
                                     image_in_tag='binary',
                                     image_out_tag='extract6',
                                     image_size=0.15,
                                     search_size=0.07,
                                     filter_size=0.05)

        self.pipeline.add_module(module)
        self.pipeline.run_module('extract6')

        data = self.pipeline.get_data('extract6')
        assert np.sum(data) == pytest.approx(1.3789593661036972, rel=self.limit, abs=0.)
        assert data.shape == (10, 7, 7)
