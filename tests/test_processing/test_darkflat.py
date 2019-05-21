import os
import warnings

import h5py
import pytest
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.processing.darkflat import DarkCalibrationModule, FlatCalibrationModule
from pynpoint.util.tests import create_config, remove_test_data

warnings.simplefilter('always')

limit = 1e-10

class TestDarkFlat:

    def setup_class(self):

        self.test_dir = os.path.dirname(__file__) + '/'

        np.random.seed(1)

        images = np.random.normal(loc=0, scale=2e-4, size=(10, 100, 100))
        dark = np.random.normal(loc=0, scale=2e-4, size=(10, 100, 100))
        flat = np.random.normal(loc=0, scale=2e-4, size=(10, 100, 100))
        crop = np.random.normal(loc=0, scale=2e-4, size=(10, 60, 60))

        with h5py.File(self.test_dir+'PynPoint_database.hdf5', 'w') as hdf_file:
            hdf_file.create_dataset('images', data=images)
            hdf_file.create_dataset('dark', data=dark)
            hdf_file.create_dataset('flat', data=flat)
            hdf_file.create_dataset('crop', data=crop)

        create_config(self.test_dir+'PynPoint_config.ini')

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self):

        remove_test_data(self.test_dir)

    def test_input_data(self):

        data = self.pipeline.get_data('dark')
        assert np.allclose(data[0, 10, 10], 3.528694163309295e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 7.368663496379876e-07, rtol=limit, atol=0.)
        assert data.shape == (10, 100, 100)

        data = self.pipeline.get_data('flat')
        assert np.allclose(data[0, 10, 10], -0.0004053528990466237, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), -4.056978234798532e-07, rtol=limit, atol=0.)
        assert data.shape == (10, 100, 100)

    def test_dark_calibration(self):

        module = DarkCalibrationModule(name_in='dark',
                                       image_in_tag='images',
                                       dark_in_tag='dark',
                                       image_out_tag='dark_cal')

        self.pipeline.add_module(module)
        self.pipeline.run_module('dark')

        data = self.pipeline.get_data('dark_cal')
        assert np.allclose(data[0, 10, 10], 0.0001881700200690493, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 3.137393482985464e-07, rtol=limit, atol=0.)
        assert data.shape == (10, 100, 100)

    def test_flat_calibration(self):

        module = FlatCalibrationModule(name_in='flat',
                                       image_in_tag='dark_cal',
                                       flat_in_tag='flat',
                                       image_out_tag='flat_cal')

        self.pipeline.add_module(module)
        self.pipeline.run_module('flat')

        data = self.pipeline.get_data('flat_cal')
        assert np.allclose(data[0, 10, 10], 0.00018817459122508784, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 3.136868889624257e-07, rtol=limit, atol=0.)
        assert data.shape == (10, 100, 100)

    def test_flat_crop(self):

        module = FlatCalibrationModule(name_in='flat_crop',
                                       image_in_tag='crop',
                                       flat_in_tag='flat',
                                       image_out_tag='flat_crop')

        self.pipeline.add_module(module)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module('flat_crop')

        assert len(warning) == 1
        assert warning[0].message.args[0] == 'The calibration images were cropped around their ' \
                                             'center to match the shape of the science images.'

        data = self.pipeline.get_data('flat_crop')
        assert np.allclose(data[0, 10, 10], -8.061511635865643e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 1.7280213934907495e-06, rtol=limit, atol=0.)
        assert data.shape == (10, 60, 60)

    def test_flat_too_small(self):

        module = FlatCalibrationModule(name_in='flat_small',
                                       image_in_tag='flat',
                                       flat_in_tag='crop',
                                       image_out_tag='flat_small')

        self.pipeline.add_module(module)

        with pytest.raises(ValueError) as error:
            self.pipeline.run_module('flat_small')

        assert str(error.value) == 'Shape of the calibration images is smaller than the ' \
                                   'science images.'
