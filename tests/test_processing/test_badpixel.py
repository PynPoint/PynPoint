import os
import math
import warnings

import h5py
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.processing.badpixel import BadPixelSigmaFilterModule, BadPixelMapModule, \
                                         BadPixelInterpolationModule, BadPixelTimeFilterModule, \
                                         ReplaceBadPixelsModule
from pynpoint.util.tests import create_config, remove_test_data

warnings.simplefilter('always')

limit = 1e-10

class TestBadPixel:

    def setup_class(self):

        self.test_dir = os.path.dirname(__file__) + '/'

        np.random.seed(1)
        images = np.random.normal(loc=0, scale=2e-4, size=(40, 100, 100))
        dark = np.random.normal(loc=0, scale=2e-4, size=(40, 100, 100))
        flat = np.random.normal(loc=0, scale=2e-4, size=(40, 100, 100))

        images[0, 10, 10] = 1.
        images[0, 12, 12] = 1.
        images[0, 14, 14] = 1.
        images[0, 20, 20] = 1.
        images[0, 22, 22] = 1.
        images[0, 24, 24] = 1.
        dark[:, 10, 10] = 1.
        dark[:, 12, 12] = 1.
        dark[:, 14, 14] = 1.
        flat[:, 20, 20] = -1.
        flat[:, 22, 22] = -1.
        flat[:, 24, 24] = -1.

        h5f = h5py.File(self.test_dir+'PynPoint_database.hdf5', 'w')
        h5f.create_dataset('images', data=images)
        h5f.create_dataset('dark', data=dark)
        h5f.create_dataset('flat', data=flat)
        h5f.create_dataset('header_images/STAR_POSITION', data=np.full((40, 2), 50.))
        h5f.close()

        create_config(self.test_dir+'PynPoint_config.ini')

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self):

        remove_test_data(self.test_dir)

    def test_bad_pixel_sigma_filter(self):

        module = BadPixelSigmaFilterModule(name_in='sigma1',
                                           image_in_tag='images',
                                           image_out_tag='sigma1',
                                           map_out_tag='None',
                                           box=9,
                                           sigma=5,
                                           iterate=1)

        self.pipeline.add_module(module)
        self.pipeline.run_module('sigma1')

        data = self.pipeline.get_data('sigma1')
        assert np.allclose(data[0, 0, 0], 0.00032486907273264834, rtol=limit, atol=0.)
        assert np.allclose(data[0, 10, 10], 0.025022559679385093, rtol=limit, atol=0.)
        assert np.allclose(data[0, 20, 20], 0.024962143884217046, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 6.721637736047109e-07, rtol=limit, atol=0.)
        assert data.shape == (40, 100, 100)

    def test_bad_pixel_map_out(self):

        module = BadPixelSigmaFilterModule(name_in='sigma2',
                                           image_in_tag='images',
                                           image_out_tag='sigma2',
                                           map_out_tag='bpmap',
                                           box=9,
                                           sigma=5,
                                           iterate=0)

        self.pipeline.add_module(module)
        self.pipeline.run_module('sigma2')

        data = self.pipeline.get_data('sigma2')
        assert np.allclose(data[0, 0, 0], 0.00032486907273264834, rtol=limit, atol=0.)
        assert np.allclose(data[0, 10, 10], 0.025022559679385093, rtol=limit, atol=0.)
        assert np.allclose(data[0, 20, 20], 0.024962143884217046, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 6.721637736047109e-07, rtol=limit, atol=0.)
        assert data.shape == (40, 100, 100)

        data = self.pipeline.get_data('bpmap')
        assert np.allclose(data[0, 0, 0], 1.0, rtol=limit, atol=0.)
        assert np.allclose(np.sum(data), 399994.0, rtol=limit, atol=0.)
        assert data.shape == (40, 100, 100)

    def test_bad_pixel_map(self):

        module = BadPixelMapModule(name_in='bp_map1',
                                   dark_in_tag='dark',
                                   flat_in_tag='flat',
                                   bp_map_out_tag='bp_map1',
                                   dark_threshold=0.99,
                                   flat_threshold=-0.99)

        self.pipeline.add_module(module)
        self.pipeline.run_module('bp_map1')

        data = self.pipeline.get_data('bp_map1')
        assert data[0, 0, 0] == 1.
        assert data[0, 30, 30] == 1.
        assert data[0, 10, 10] == 0.
        assert data[0, 12, 12] == 0.
        assert data[0, 14, 14] == 0.
        assert data[0, 20, 20] == 0.
        assert data[0, 22, 22] == 0.
        assert data[0, 24, 24] == 0.
        assert np.mean(data) == 0.9993
        assert data.shape == (1, 100, 100)

    def test_map_no_dark(self):

        module = BadPixelMapModule(name_in='bp_map2',
                                   dark_in_tag=None,
                                   flat_in_tag='flat',
                                   bp_map_out_tag='bp_map2',
                                   dark_threshold=0.99,
                                   flat_threshold=-0.99)

        self.pipeline.add_module(module)
        self.pipeline.run_module('bp_map2')

        data = self.pipeline.get_data('bp_map2')
        assert np.sum(data) == 9996.0
        assert data.shape == (1, 100, 100)

    def test_map_no_flat(self):

        module = BadPixelMapModule(name_in='bp_map3',
                                   dark_in_tag='dark',
                                   flat_in_tag=None,
                                   bp_map_out_tag='bp_map3',
                                   dark_threshold=0.99,
                                   flat_threshold=-0.99)

        self.pipeline.add_module(module)
        self.pipeline.run_module('bp_map3')

        data = self.pipeline.get_data('bp_map3')
        assert np.sum(data) == 9997.0
        assert data.shape == (1, 100, 100)

    def test_bad_pixel_interpolation(self):

        module = BadPixelInterpolationModule(name_in='interpolation',
                                             image_in_tag='images',
                                             bad_pixel_map_tag='bp_map1',
                                             image_out_tag='interpolation',
                                             iterations=100)

        self.pipeline.add_module(module)
        self.pipeline.run_module('interpolation')

        data = self.pipeline.get_data('interpolation')
        assert np.allclose(data[0, 0, 0], 0.00032486907273264834, rtol=1e-8, atol=0.)
        assert np.allclose(data[0, 10, 10], 1.0139222106683477e-05, rtol=1e-8, atol=0.)
        assert np.allclose(data[0, 20, 20], -4.686852973820094e-05, rtol=1e-8, atol=0.)
        assert np.allclose(np.mean(data), 3.0499629451215465e-07, rtol=1e-8, atol=0.)
        assert data.shape == (40, 100, 100)

    def test_bad_pixel_time(self):

        module = BadPixelTimeFilterModule(name_in='time',
                                          image_in_tag='images',
                                          image_out_tag='time',
                                          sigma=(5., 5.))

        self.pipeline.add_module(module)
        self.pipeline.run_module('time')

        data = self.pipeline.get_data('time')
        assert np.allclose(data[0, 0, 0], 0.00032486907273264834, rtol=limit, atol=0.)
        assert np.allclose(data[0, 10, 10], 8.32053029919322e-06, rtol=limit, atol=0.)
        assert np.allclose(data[0, 20, 20], -2.3565404332481378e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 2.9727173024489924e-07, rtol=limit, atol=0.)
        assert data.shape == (40, 100, 100)

    def test_replace_bad_pixels(self):

        module = ReplaceBadPixelsModule(name_in='replace1',
                                        image_in_tag='images',
                                        map_in_tag='bp_map1',
                                        image_out_tag='replace',
                                        size=2,
                                        replace='mean')

        self.pipeline.add_module(module)
        self.pipeline.run_module('replace1')

        data = self.pipeline.get_data('replace')
        assert np.allclose(data[0, 0, 0], 0.00032486907273264834, rtol=limit, atol=0.)
        assert np.allclose(data[0, 10, 10], 5.493280938051695e-05, rtol=limit, atol=0.)
        assert np.allclose(data[0, 20, 20], -5.51613113375153e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 3.019578931588861e-07, rtol=limit, atol=0.)
        assert data.shape == (40, 100, 100)

        module = ReplaceBadPixelsModule(name_in='replace2',
                                        image_in_tag='images',
                                        map_in_tag='bp_map1',
                                        image_out_tag='replace',
                                        size=2,
                                        replace='median')

        self.pipeline.add_module(module)
        self.pipeline.run_module('replace2')

        data = self.pipeline.get_data('replace')
        assert np.allclose(data[0, 0, 0], 0.00032486907273264834, rtol=limit, atol=0.)
        assert np.allclose(data[0, 10, 10], 9.195634004203678e-05, rtol=limit, atol=0.)
        assert np.allclose(data[0, 20, 20], -6.101079878960902e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 2.983915326435115e-07, rtol=limit, atol=0.)
        assert data.shape == (40, 100, 100)

        module = ReplaceBadPixelsModule(name_in='replace3',
                                        image_in_tag='images',
                                        map_in_tag='bp_map1',
                                        image_out_tag='replace',
                                        size=2,
                                        replace='nan')

        self.pipeline.add_module(module)
        self.pipeline.run_module('replace3')

        data = self.pipeline.get_data('replace')
        assert np.allclose(data[0, 0, 0], 0.00032486907273264834, rtol=limit, atol=0.)
        assert math.isnan(data[0, 10, 10])
        assert math.isnan(data[0, 20, 20])
        assert np.allclose(np.nanmean(data), 3.0516702371516344e-07, rtol=limit, atol=0.)
        assert data.shape == (40, 100, 100)
