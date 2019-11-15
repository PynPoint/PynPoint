import os
import warnings

import h5py
import pytest
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.fitsreading import FitsReadingModule
from pynpoint.processing.extract import StarExtractionModule, ExtractBinaryModule
from pynpoint.processing.psfpreparation import AngleInterpolationModule
from pynpoint.util.tests import create_config, create_star_data, create_fake, remove_test_data

warnings.simplefilter('always')

limit = 1e-10


class TestExtract:

    def setup_class(self):

        self.test_dir = os.path.dirname(__file__) + '/'

        create_star_data(path=self.test_dir+'star',
                         npix_x=51,
                         npix_y=51,
                         x0=[10., 10., 10., 10.],
                         y0=[10., 10., 10., 10.])

        create_fake(path=self.test_dir+'binary',
                    ndit=[20, 20, 20, 20],
                    nframes=[20, 20, 20, 20],
                    exp_no=[1, 2, 3, 4],
                    npix=(101, 101),
                    fwhm=3.,
                    x0=[50, 50, 50, 50],
                    y0=[50, 50, 50, 50],
                    angles=[[0., 25.], [25., 50.], [50., 75.], [75., 100.]],
                    sep=20.,
                    contrast=1.)

        create_config(self.test_dir+'PynPoint_config.ini')

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self):

        remove_test_data(path=self.test_dir, folders=['star', 'binary'])

    def test_read_data(self):

        module = FitsReadingModule(name_in='read1',
                                   image_tag='star',
                                   input_dir=self.test_dir+'star',
                                   overwrite=True,
                                   check=True)

        self.pipeline.add_module(module)
        self.pipeline.run_module('read1')

        data = self.pipeline.get_data('star')
        assert np.allclose(data[0, 10, 10], 0.09834884212021108, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00038538535294683216, rtol=limit, atol=0.)
        assert data.shape == (40, 51, 51)

        module = FitsReadingModule(name_in='read2',
                                   image_tag='binary',
                                   input_dir=self.test_dir+'binary',
                                   overwrite=True,
                                   check=True)

        self.pipeline.add_module(module)
        self.pipeline.run_module('read2')

        data = self.pipeline.get_data('binary')
        assert np.allclose(data[0, 50, 50], 0.0986064357966972, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00019636787665654158, rtol=limit, atol=0.)
        assert data.shape == (80, 101, 101)

    def test_angle_interpolation(self):

        module = AngleInterpolationModule(name_in='angle',
                                          data_tag='binary')

        self.pipeline.add_module(module)
        self.pipeline.run_module('angle')

        data = self.pipeline.get_attribute('binary', 'PARANG', static=False)
        assert data[5] == 6.578947368421053
        assert np.allclose(np.mean(data), 50.0, rtol=limit, atol=0.)
        assert data.shape == (80, )

        parang = self.pipeline.get_attribute('binary', 'PARANG', static=False)
        self.pipeline.set_attribute('binary', 'PARANG', -1.*parang, static=False)

        data = self.pipeline.get_attribute('binary', 'PARANG', static=False)
        assert data[5] == -6.578947368421053
        assert np.allclose(np.mean(data), -50.0, rtol=limit, atol=0.)
        assert data.shape == (80, )

    def test_extract_position_none(self):

        module = StarExtractionModule(name_in='extract1',
                                      image_in_tag='star',
                                      image_out_tag='extract1',
                                      index_out_tag='index',
                                      image_size=0.4,
                                      fwhm_star=0.1,
                                      position=None)

        self.pipeline.add_module(module)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module('extract1')

        assert len(warning) == 1
        assert warning[0].message.args[0] == 'The new dataset that is stored under the tag name ' \
                                             '\'index\' is empty.'

        data = self.pipeline.get_data('extract1')

        assert np.allclose(data[0, 7, 7], 0.09834884212021108, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.004444871536643222, rtol=limit, atol=0.)
        assert data.shape == (40, 15, 15)

        attr = self.pipeline.get_attribute('extract1', 'STAR_POSITION', static=False)
        assert attr[10, 0] == attr[10, 1] == 10

    def test_extract_center_none(self):

        module = StarExtractionModule(name_in='extract2',
                                      image_in_tag='star',
                                      image_out_tag='extract2',
                                      index_out_tag='index',
                                      image_size=0.4,
                                      fwhm_star=0.1,
                                      position=(None, None, 1.))

        self.pipeline.add_module(module)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module('extract2')

        assert len(warning) == 1
        assert warning[0].message.args[0] == 'The new dataset that is stored under the tag name ' \
                                             '\'index\' is empty.'

        data = self.pipeline.get_data('extract2')

        assert np.allclose(data[0, 7, 7], 0.09834884212021108, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.004444871536643222, rtol=limit, atol=0.)
        assert data.shape == (40, 15, 15)

        attr = self.pipeline.get_attribute('extract2', 'STAR_POSITION', static=False)
        assert attr[10, 0] == attr[10, 1] == 10

    def test_extract_position(self):

        module = StarExtractionModule(name_in='extract7',
                                      image_in_tag='star',
                                      image_out_tag='extract7',
                                      index_out_tag=None,
                                      image_size=0.4,
                                      fwhm_star=0.1,
                                      position=(10, 10, 0.1))

        self.pipeline.add_module(module)
        self.pipeline.run_module('extract7')

        data = self.pipeline.get_data('extract7')

        assert np.allclose(data[0, 7, 7], 0.09834884212021108, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.004444871536643222, rtol=limit, atol=0.)
        assert data.shape == (40, 15, 15)

        attr = self.pipeline.get_attribute('extract7', 'STAR_POSITION', static=False)
        assert attr[10, 0] == attr[10, 1] == 10

    def test_extract_too_large(self):

        module = StarExtractionModule(name_in='extract3',
                                      image_in_tag='star',
                                      image_out_tag='extract3',
                                      index_out_tag=None,
                                      image_size=0.8,
                                      fwhm_star=0.1,
                                      position=None)

        self.pipeline.add_module(module)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module('extract3')

        assert len(warning) == 40

        for i, item in enumerate(warning):
            assert item.message.args[0] == f'Chosen image size is too large to crop the image ' \
                                           f'around the brightest pixel (image index = {i}, ' \
                                           f'pixel [x, y] = [10, 10]). Using the center of ' \
                                           f'the image instead.'

        data = self.pipeline.get_data('extract3')

        assert np.allclose(data[0, 0, 0], 0.09834884212021108, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.0004499242959139202, rtol=limit, atol=0.)
        assert data.shape == (40, 31, 31)

        attr = self.pipeline.get_attribute('extract3', 'STAR_POSITION', static=False)
        assert attr[10, 0] == attr[10, 1] == 25

    def test_star_extract_cpu(self):

        with h5py.File(self.test_dir+'PynPoint_database.hdf5', 'a') as hdf_file:
            hdf_file['config'].attrs['CPU'] = 4

        module = StarExtractionModule(name_in='extract4',
                                      image_in_tag='star',
                                      image_out_tag='extract4',
                                      index_out_tag='index',
                                      image_size=0.8,
                                      fwhm_star=0.1,
                                      position=None)

        self.pipeline.add_module(module)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module('extract4')

        assert len(warning) == 1
        assert warning[0].message.args[0] == 'Chosen image size is too large to crop the image ' \
                                             'around the brightest pixel. Using the center of ' \
                                             'the image instead.'

    def test_extract_binary(self):

        with h5py.File(self.test_dir+'PynPoint_database.hdf5', 'a') as hdf_file:
            hdf_file['config'].attrs['CPU'] = 1

        module = ExtractBinaryModule(pos_center=(50., 50.),
                                     pos_binary=(50., 70.),
                                     name_in='extract5',
                                     image_in_tag='binary',
                                     image_out_tag='extract5',
                                     image_size=0.5,
                                     search_size=0.2,
                                     filter_size=None)

        self.pipeline.add_module(module)
        self.pipeline.run_module('extract5')

        data = self.pipeline.get_data('extract5')

        assert np.allclose(data[0, 9, 9], 0.09774483733119443, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.0027700881940171283, rtol=limit, atol=0.)
        assert data.shape == (80, 19, 19)

    def test_extract_binary_filter(self):

        module = ExtractBinaryModule(pos_center=(50., 50.),
                                     pos_binary=(50., 70.),
                                     name_in='extract6',
                                     image_in_tag='binary',
                                     image_out_tag='extract6',
                                     image_size=0.5,
                                     search_size=0.2,
                                     filter_size=0.1)

        self.pipeline.add_module(module)
        self.pipeline.run_module('extract6')

        data = self.pipeline.get_data('extract6')

        assert np.allclose(data[0, 9, 9], 0.09774483733119443, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.002770040591615301, rtol=limit, atol=0.)
        assert data.shape == (80, 19, 19)
