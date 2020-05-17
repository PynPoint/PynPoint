import os
import math

import h5py
import pytest
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.processing.badpixel import BadPixelSigmaFilterModule, BadPixelMapModule, \
                                         BadPixelInterpolationModule, BadPixelTimeFilterModule, \
                                         ReplaceBadPixelsModule
from pynpoint.util.tests import create_config, remove_test_data


class TestBadPixel:

    def setup_class(self) -> None:

        self.limit = 1e-10
        self.test_dir = os.path.dirname(__file__) + '/'

        np.random.seed(1)

        images = np.random.normal(loc=0, scale=2e-4, size=(5, 11, 11))
        dark = np.random.normal(loc=0, scale=2e-4, size=(5, 11, 11))
        flat = np.random.normal(loc=0, scale=2e-4, size=(5, 11, 11))

        images[0, 5, 5] = 1.
        dark[:, 5, 5] = 1.
        flat[:, 8, 8] = -1.
        flat[:, 9, 9] = -1.
        flat[:, 10, 10] = -1.

        with h5py.File(self.test_dir+'PynPoint_database.hdf5', 'w') as hdf_file:
            hdf_file.create_dataset('images', data=images)
            hdf_file.create_dataset('dark', data=dark)
            hdf_file.create_dataset('flat', data=flat)

        create_config(self.test_dir+'PynPoint_config.ini')

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self) -> None:

        remove_test_data(self.test_dir)

    def test_bad_pixel_sigma_filter(self) -> None:

        module = BadPixelSigmaFilterModule(name_in='sigma1',
                                           image_in_tag='images',
                                           image_out_tag='sigma1',
                                           map_out_tag='None',
                                           box=9,
                                           sigma=5.,
                                           iterate=1)

        self.pipeline.add_module(module)
        self.pipeline.run_module('sigma1')

        data = self.pipeline.get_data('sigma1')
        assert np.sum(data) == pytest.approx(0.007314386854009355, rel=self.limit, abs=0.)
        assert data.shape == (5, 11, 11)

    def test_bad_pixel_map_out(self) -> None:

        module = BadPixelSigmaFilterModule(name_in='sigma2',
                                           image_in_tag='images',
                                           image_out_tag='sigma2',
                                           map_out_tag='bpmap',
                                           box=9,
                                           sigma=5.,
                                           iterate=1)

        self.pipeline.add_module(module)
        self.pipeline.run_module('sigma2')

        data = self.pipeline.get_data('sigma2')
        assert data[0, 0, 0] == pytest.approx(0.00032486907273264834, rel=self.limit, abs=0.)
        assert data[0, 5, 5] == pytest.approx(9.903775276151606e-06, rel=self.limit, abs=0.)
        assert np.sum(data) == pytest.approx(0.007314386854009355, rel=self.limit, abs=0.)
        assert data.shape == (5, 11, 11)

        data = self.pipeline.get_data('bpmap')
        assert data[0, 0, 0] == pytest.approx(1., rel=self.limit, abs=0.)
        assert data[0, 5, 5] == pytest.approx(0., rel=self.limit, abs=0.)
        assert np.sum(data) == pytest.approx(604., rel=self.limit, abs=0.)
        assert data.shape == (5, 11, 11)

    def test_bad_pixel_map(self) -> None:

        module = BadPixelMapModule(name_in='bp_map1',
                                   dark_in_tag='dark',
                                   flat_in_tag='flat',
                                   bp_map_out_tag='bp_map1',
                                   dark_threshold=0.99,
                                   flat_threshold=-0.99)

        self.pipeline.add_module(module)
        self.pipeline.run_module('bp_map1')

        data = self.pipeline.get_data('bp_map1')
        assert data[0, 0, 0] == pytest.approx(1., rel=self.limit, abs=0.)
        assert data[0, 5, 5] == pytest.approx(0., rel=self.limit, abs=0.)
        assert np.sum(data) == pytest.approx(117., rel=self.limit, abs=0.)
        assert data.shape == (1, 11, 11)

    def test_map_no_dark(self) -> None:

        module = BadPixelMapModule(name_in='bp_map2',
                                   dark_in_tag=None,
                                   flat_in_tag='flat',
                                   bp_map_out_tag='bp_map2',
                                   dark_threshold=0.99,
                                   flat_threshold=-0.99)

        self.pipeline.add_module(module)
        self.pipeline.run_module('bp_map2')

        data = self.pipeline.get_data('bp_map2')
        assert np.sum(data) == pytest.approx(118., rel=self.limit, abs=0.)
        assert data.shape == (1, 11, 11)

    def test_map_no_flat(self) -> None:

        module = BadPixelMapModule(name_in='bp_map3',
                                   dark_in_tag='dark',
                                   flat_in_tag=None,
                                   bp_map_out_tag='bp_map3',
                                   dark_threshold=0.99,
                                   flat_threshold=-0.99)

        self.pipeline.add_module(module)
        self.pipeline.run_module('bp_map3')

        data = self.pipeline.get_data('bp_map3')
        assert np.sum(data) == pytest.approx(120., rel=self.limit, abs=0.)
        assert data.shape == (1, 11, 11)

    def test_bad_pixel_interpolation(self) -> None:

        module = BadPixelInterpolationModule(name_in='interpolation',
                                             image_in_tag='images',
                                             bad_pixel_map_tag='bp_map1',
                                             image_out_tag='interpolation',
                                             iterations=10)

        self.pipeline.add_module(module)
        self.pipeline.run_module('interpolation')

        data = self.pipeline.get_data('interpolation')
        assert data[0, 0, 0] == pytest.approx(0.00032486907273264834, rel=1e-8, abs=0.)
        assert data[0, 5, 5] == pytest.approx(-1.4292408645473845e-05, rel=1e-8, abs=0.)
        assert np.sum(data) == pytest.approx(0.008683344127872174, rel=1e-8, abs=0.)
        assert data.shape == (5, 11, 11)

    def test_bad_pixel_time(self) -> None:

        module = BadPixelTimeFilterModule(name_in='time',
                                          image_in_tag='images',
                                          image_out_tag='time',
                                          sigma=(2., 2.))

        self.pipeline.add_module(module)
        self.pipeline.run_module('time')

        data = self.pipeline.get_data('time')
        assert data[0, 0, 0] == pytest.approx(0.00032486907273264834, rel=self.limit, abs=0.)
        assert data[0, 5, 5] == pytest.approx(-0.00017468532119886812, rel=self.limit, abs=0.)
        assert np.sum(data) == pytest.approx(0.004175672043832705, rel=self.limit, abs=0.)
        assert data.shape == (5, 11, 11)

    def test_replace_bad_pixels(self) -> None:

        module = ReplaceBadPixelsModule(name_in='replace1',
                                        image_in_tag='images',
                                        map_in_tag='bp_map1',
                                        image_out_tag='replace',
                                        size=2,
                                        replace='mean')

        self.pipeline.add_module(module)
        self.pipeline.run_module('replace1')

        data = self.pipeline.get_data('replace')
        assert data[0, 0, 0] == pytest.approx(0.00032486907273264834, rel=self.limit, abs=0.)
        assert data[0, 5, 5] == pytest.approx(4.260114004413933e-05, rel=self.limit, abs=0.)
        assert np.sum(data) == pytest.approx(0.00883357395370896, rel=self.limit, abs=0.)
        assert data.shape == (5, 11, 11)

        module = ReplaceBadPixelsModule(name_in='replace2',
                                        image_in_tag='images',
                                        map_in_tag='bp_map1',
                                        image_out_tag='replace',
                                        size=2,
                                        replace='median')

        self.pipeline.add_module(module)
        self.pipeline.run_module('replace2')

        data = self.pipeline.get_data('replace')
        assert data[0, 0, 0] == pytest.approx(0.00032486907273264834, rel=self.limit, abs=0.)
        assert data[0, 5, 5] == pytest.approx(4.327154179438619e-05, rel=self.limit, abs=0.)
        assert np.sum(data) == pytest.approx(0.008489525337709688, rel=self.limit, abs=0.)
        assert data.shape == (5, 11, 11)

        module = ReplaceBadPixelsModule(name_in='replace3',
                                        image_in_tag='images',
                                        map_in_tag='bp_map1',
                                        image_out_tag='replace',
                                        size=2,
                                        replace='nan')

        self.pipeline.add_module(module)
        self.pipeline.run_module('replace3')

        data = self.pipeline.get_data('replace')
        assert data[0, 0, 0] == pytest.approx(0.00032486907273264834, rel=self.limit, abs=0.)
        assert math.isnan(data[0, 5, 5])
        assert np.nansum(data) == pytest.approx(0.009049653234723834, rel=self.limit, abs=0.)
        assert data.shape == (5, 11, 11)
