import os

import h5py
import pytest
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.fitsreading import FitsReadingModule
from pynpoint.processing.background import LineSubtractionModule
from pynpoint.processing.badpixel import BadPixelSigmaFilterModule
from pynpoint.processing.extract import StarExtractionModule
from pynpoint.processing.timedenoising import TimeNormalizationModule
from pynpoint.util.tests import create_config, create_star_data, remove_test_data


class TestProcessing:

    def setup_class(self) -> None:

        self.limit = 1e-10
        self.test_dir = os.path.dirname(__file__) + '/'

        np.random.seed(1)

        images = np.random.normal(loc=0, scale=2e-4, size=(5, 11, 11))
        large_data = np.random.normal(loc=0, scale=2e-4, size=(10000, 5, 5))

        with h5py.File(self.test_dir+'PynPoint_database.hdf5', 'w') as hdf_file:
            hdf_file.create_dataset('images', data=images)
            hdf_file.create_dataset('large_data', data=large_data)

        create_star_data(path=self.test_dir+'images')
        create_config(self.test_dir+'PynPoint_config.ini')

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

        self.pipeline.set_attribute('images', 'PIXSCALE', 0.1, static=True)
        self.pipeline.set_attribute('large_data', 'PIXSCALE', 0.1, static=True)

    def teardown_class(self) -> None:

        remove_test_data(self.test_dir, folders=['images'])

    def test_output_port_name(self) -> None:

        module = FitsReadingModule(name_in='read',
                                   image_tag='images',
                                   input_dir=self.test_dir+'images')

        module.add_output_port('test')

        with pytest.warns(UserWarning) as warning:
            module.add_output_port('test')

        assert len(warning) == 1
        assert warning[0].message.args[0] == 'Tag \'test\' of ReadingModule \'read\' is already ' \
                                             'used.'

        module = BadPixelSigmaFilterModule(name_in='badpixel',
                                           image_in_tag='images',
                                           image_out_tag='im_out')

        module.add_output_port('test')

        with pytest.warns(UserWarning) as warning:
            module.add_output_port('test')

        assert len(warning) == 1
        assert warning[0].message.args[0] == 'Tag \'test\' of ProcessingModule \'badpixel\' is ' \
                                             'already used.'

    def test_output_port_set_connection(self) -> None:

        self.pipeline.m_data_storage.open_connection()

        module = BadPixelSigmaFilterModule(name_in='badpixel2',
                                           image_in_tag='images',
                                           image_out_tag='im_out')

        self.pipeline.add_module(module)

        port = module.add_output_port('test1')

        self.pipeline.m_data_storage.close_connection()

    def test_apply_function(self) -> None:

        self.pipeline.set_attribute('config', 'MEMORY', 20, static=True)
        self.pipeline.set_attribute('config', 'CPU', 4, static=True)

        module = LineSubtractionModule(name_in='subtract',
                                       image_in_tag='images',
                                       image_out_tag='im_subtract',
                                       combine='mean',
                                       mask=None)

        self.pipeline.add_module(module)
        self.pipeline.run_module('subtract')

        data = self.pipeline.get_data('images')
        assert np.mean(data) == pytest.approx(1.1824138000882435e-05, rel=self.limit, abs=0.)
        assert data.shape == (5, 11, 11)

        data = self.pipeline.get_data('im_subtract')
        assert np.mean(data) == pytest.approx(-1.2544487946113274e-21, rel=self.limit, abs=0.)
        assert data.shape == (5, 11, 11)

    def test_apply_function_args_none(self) -> None:

        module = TimeNormalizationModule(name_in='norm',
                                         image_in_tag='images',
                                         image_out_tag='im_norm')

        self.pipeline.add_module(module)
        self.pipeline.run_module('norm')

        data = self.pipeline.get_data('im_norm')
        assert np.mean(data) == pytest.approx(2.4012571778516812e-06, rel=self.limit, abs=0.)
        assert data.shape == (5, 11, 11)

    def test_apply_function_args_none_memory_none(self) -> None:

        self.pipeline.set_attribute('config', 'MEMORY', 0, static=True)

        module = TimeNormalizationModule(name_in='norm_none',
                                         image_in_tag='images',
                                         image_out_tag='im_norm')

        self.pipeline.add_module(module)
        self.pipeline.run_module('norm_none')

        data = self.pipeline.get_data('im_norm')
        assert np.mean(data) == pytest.approx(2.4012571778516812e-06, rel=self.limit, abs=0.)
        assert data.shape == (5, 11, 11)

    def test_apply_function_same_port(self) -> None:

        module = LineSubtractionModule(name_in='subtract_same',
                                       image_in_tag='im_subtract',
                                       image_out_tag='im_subtract',
                                       combine='mean',
                                       mask=None)

        self.pipeline.add_module(module)
        self.pipeline.run_module('subtract_same')

        data = self.pipeline.get_data('im_subtract')
        assert np.mean(data) == pytest.approx(-1.4336557652700885e-21, rel=self.limit, abs=0.)
        assert data.shape == (5, 11, 11)

    def test_apply_function_args_none_memory_none_same_port(self) -> None:

        self.pipeline.set_attribute('config', 'MEMORY', 0, static=True)

        data = self.pipeline.get_data('images')
        assert np.mean(data) == pytest.approx(1.1824138000882435e-05, rel=self.limit, abs=0.)
        assert data.shape == (5, 11, 11)

        module = TimeNormalizationModule(name_in='norm_none_same',
                                         image_in_tag='images',
                                         image_out_tag='images')

        self.pipeline.add_module(module)
        self.pipeline.run_module('norm_none_same')

        data = self.pipeline.get_data('images')
        assert np.mean(data) == pytest.approx(2.4012571778516812e-06, rel=self.limit, abs=0.)
        assert data.shape == (5, 11, 11)

    # def test_apply_function_to_images_memory_none(self) -> None:
    #
    #     module = StarExtractionModule(name_in='extract',
    #                                   image_in_tag='im_subtract',
    #                                   image_out_tag='extract',
    #                                   index_out_tag=None,
    #                                   image_size=0.5,
    #                                   fwhm_star=0.1,
    #                                   position=(None, None, 0.1))
    #
    #     self.pipeline.add_module(module)
    #     self.pipeline.run_module('extract')
    #
    #     data = self.pipeline.get_data('extract')
    #     assert np.mean(data) == pytest.approx(1.8259937251367536e-05, rel=self.limit, abs=0.)
    #     assert data.shape == (5, 5, 5)

    # def test_multiproc_large_data(self) -> None:
    #
    #     self.pipeline.set_attribute('config', 'MEMORY', 1000, static=True)
    #     self.pipeline.set_attribute('config', 'CPU', 1, static=True)
    #
    #     module = LineSubtractionModule(name_in='subtract_single',
    #                                    image_in_tag='large_data',
    #                                    image_out_tag='im_sub_single',
    #                                    combine='mean',
    #                                    mask=None)
    #
    #     self.pipeline.add_module(module)
    #     self.pipeline.run_module('subtract_single')
    #
    #     self.pipeline.set_attribute('config', 'CPU', 4, static=True)
    #
    #     module = LineSubtractionModule(name_in='subtract_multi',
    #                                    image_in_tag='large_data',
    #                                    image_out_tag='im_sub_multi',
    #                                    combine='mean',
    #                                    mask=None)
    #
    #     self.pipeline.add_module(module)
    #     self.pipeline.run_module('subtract_multi')
    #
    #     data_single = self.pipeline.get_data('im_sub_single')
    #     data_multi = self.pipeline.get_data('im_sub_multi')
    #     assert data_single == pytest.approx(data_multi, rel=self.limit, abs=0.)
    #     assert data_single.shape == data_multi.shape
