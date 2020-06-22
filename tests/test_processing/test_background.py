import os

import pytest
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.fitsreading import FitsReadingModule
from pynpoint.processing.background import MeanBackgroundSubtractionModule, \
                                           SimpleBackgroundSubtractionModule, \
                                           LineSubtractionModule, \
                                           NoddingBackgroundModule
from pynpoint.processing.pcabackground import DitheringBackgroundModule
from pynpoint.processing.stacksubset import StackCubesModule
from pynpoint.util.tests import create_config, create_dither_data, create_star_data, \
                                remove_test_data


class TestBackground:

    def setup_class(self) -> None:

        self.limit = 1e-10
        self.test_dir = os.path.dirname(__file__) + '/'

        create_dither_data(self.test_dir+'dither')
        create_star_data(self.test_dir+'science')
        create_star_data(self.test_dir+'sky')
        create_config(self.test_dir+'PynPoint_config.ini')

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self) -> None:

        remove_test_data(self.test_dir, folders=['dither', 'science', 'sky'])

    def test_read_data(self) -> None:

        module = FitsReadingModule(name_in='read1',
                                   image_tag='dither',
                                   input_dir=self.test_dir+'dither')

        self.pipeline.add_module(module)
        self.pipeline.run_module('read1')

        data = self.pipeline.get_data('dither')
        assert np.sum(data) == pytest.approx(211.20534661914408, rel=self.limit, abs=0.)
        assert data.shape == (20, 21, 21)

        module = FitsReadingModule(name_in='read2',
                                   image_tag='science',
                                   input_dir=self.test_dir+'science')

        self.pipeline.add_module(module)
        self.pipeline.run_module('read2')

        data = self.pipeline.get_data('science')
        assert np.sum(data) == pytest.approx(105.54278879805277, rel=self.limit, abs=0.)
        assert data.shape == (10, 11, 11)

        module = FitsReadingModule(name_in='read3',
                                   image_tag='sky',
                                   input_dir=self.test_dir+'sky')

        self.pipeline.add_module(module)
        self.pipeline.run_module('read3')

        data = self.pipeline.get_data('sky')
        assert np.sum(data) == pytest.approx(105.54278879805277, rel=self.limit, abs=0.)
        assert data.shape == (10, 11, 11)

    def test_simple_background(self) -> None:

        module = SimpleBackgroundSubtractionModule(shift=5,
                                                   name_in='simple',
                                                   image_in_tag='dither',
                                                   image_out_tag='simple')

        self.pipeline.add_module(module)
        self.pipeline.run_module('simple')

        data = self.pipeline.get_data('simple')
        assert np.sum(data) == pytest.approx(3.552713678800501e-15, rel=self.limit, abs=0.)
        assert data.shape == (20, 21, 21)

    def test_mean_background_shift(self) -> None:

        module = MeanBackgroundSubtractionModule(shift=5,
                                                 cubes=1,
                                                 name_in='mean2',
                                                 image_in_tag='dither',
                                                 image_out_tag='mean2')

        self.pipeline.add_module(module)
        self.pipeline.run_module('mean2')

        data = self.pipeline.get_data('mean2')
        assert np.sum(data) == pytest.approx(2.473864361018551, rel=self.limit, abs=0.)
        assert data.shape == (20, 21, 21)

    def test_mean_background_nframes(self) -> None:

        module = MeanBackgroundSubtractionModule(shift=None,
                                                 cubes=1,
                                                 name_in='mean1',
                                                 image_in_tag='dither',
                                                 image_out_tag='mean1')

        self.pipeline.add_module(module)
        self.pipeline.run_module('mean1')

        data = self.pipeline.get_data('mean1')
        assert np.sum(data) == pytest.approx(2.473864361018551, rel=self.limit, abs=0.)
        assert data.shape == (20, 21, 21)

    def test_dithering_attributes(self) -> None:

        module = DitheringBackgroundModule(name_in='pca_dither1',
                                           image_in_tag='dither',
                                           image_out_tag='pca_dither1',
                                           center=None,
                                           cubes=None,
                                           size=0.2,
                                           gaussian=0.05,
                                           subframe=0.1,
                                           pca_number=1,
                                           mask_star=0.05,
                                           crop=True,
                                           prepare=True,
                                           pca_background=True,
                                           combine='pca')

        self.pipeline.add_module(module)
        self.pipeline.run_module('pca_dither1')

        data = self.pipeline.get_data('dither_dither_crop1')
        assert np.sum(data) == pytest.approx(54.62410860562912, rel=self.limit, abs=0.)
        assert data.shape == (20, 9, 9)

        data = self.pipeline.get_data('dither_dither_star1')
        assert np.sum(data) == pytest.approx(54.873885838788595, rel=self.limit, abs=0.)
        assert data.shape == (5, 9, 9)

        data = self.pipeline.get_data('dither_dither_mean1')
        assert np.sum(data) == pytest.approx(54.204960755115245, rel=self.limit, abs=0.)
        assert data.shape == (5, 9, 9)

        data = self.pipeline.get_data('dither_dither_background1')
        assert np.sum(data) == pytest.approx(-0.24977723315947564, rel=self.limit, abs=0.)
        assert data.shape == (15, 9, 9)

        data = self.pipeline.get_data('dither_dither_pca_fit1')
        assert np.sum(data) == pytest.approx(-0.01019999314121019, rel=1e-6, abs=0.)
        assert data.shape == (5, 9, 9)

        data = self.pipeline.get_data('dither_dither_pca_res1')
        assert np.sum(data) == pytest.approx(54.884085831929795, rel=self.limit, abs=0.)
        assert data.shape == (5, 9, 9)

        data = self.pipeline.get_data('dither_dither_pca_mask1')
        assert np.sum(data) == pytest.approx(360.0, rel=self.limit, abs=0.)
        assert data.shape == (5, 9, 9)

        data = self.pipeline.get_data('pca_dither1')
        assert np.sum(data) == pytest.approx(208.774670964812, rel=self.limit, abs=0.)
        assert data.shape == (20, 9, 9)

        attr = self.pipeline.get_attribute('dither_dither_pca_res1', 'STAR_POSITION', static=False)
        assert np.sum(attr) == pytest.approx(51., rel=self.limit, abs=0.)
        assert attr.shape == (5, 2)

    def test_dithering_center(self) -> None:

        module = DitheringBackgroundModule(name_in='pca_dither2',
                                           image_in_tag='dither',
                                           image_out_tag='pca_dither2',
                                           center=((5, 5), (5, 15), (15, 15), (15, 5)),
                                           cubes=1,
                                           size=0.2,
                                           gaussian=0.05,
                                           subframe=None,
                                           pca_number=1,
                                           mask_star=0.05,
                                           crop=True,
                                           prepare=True,
                                           pca_background=True,
                                           combine='pca')

        self.pipeline.add_module(module)
        self.pipeline.run_module('pca_dither2')

        data = self.pipeline.get_data('pca_dither2')
        assert np.sum(data) == pytest.approx(209.8271898501695, rel=self.limit, abs=0.)
        assert data.shape == (20, 9, 9)

    def test_nodding_background(self) -> None:

        module = StackCubesModule(name_in='mean',
                                  image_in_tag='sky',
                                  image_out_tag='mean',
                                  combine='mean')

        self.pipeline.add_module(module)
        self.pipeline.run_module('mean')

        data = self.pipeline.get_data('mean')
        assert np.sum(data) == pytest.approx(21.108557759610548, rel=self.limit, abs=0.)
        assert data.shape == (2, 11, 11)

        attr = self.pipeline.get_attribute('mean', 'INDEX', static=False)
        assert np.sum(attr) == pytest.approx(1, rel=self.limit, abs=0.)
        assert attr.shape == (2, )

        attr = self.pipeline.get_attribute('mean', 'NFRAMES', static=False)
        assert np.sum(attr) == pytest.approx(2, rel=self.limit, abs=0.)
        assert attr.shape == (2, )

        module = NoddingBackgroundModule(name_in='nodding',
                                         sky_in_tag='mean',
                                         science_in_tag='science',
                                         image_out_tag='nodding',
                                         mode='both')

        self.pipeline.add_module(module)
        self.pipeline.run_module('nodding')

        data = self.pipeline.get_data('nodding')
        assert np.sum(data) == pytest.approx(1.793466459074705, rel=self.limit, abs=0.)
        assert data.shape == (10, 11, 11)

    def test_line_background_mean(self) -> None:

        module = LineSubtractionModule(name_in='line1',
                                       image_in_tag='science',
                                       image_out_tag='science_line1',
                                       combine='mean',
                                       mask=0.1)

        self.pipeline.add_module(module)
        self.pipeline.run_module('line1')

        data = self.pipeline.get_data('science_line1')
        assert np.sum(data) == pytest.approx(104.55443019231085, rel=self.limit, abs=0.)
        assert data.shape == (10, 11, 11)

    def test_line_background_median(self) -> None:

        module = LineSubtractionModule(name_in='line2',
                                       image_in_tag='science',
                                       image_out_tag='science_line2',
                                       combine='median',
                                       mask=0.1)

        self.pipeline.add_module(module)
        self.pipeline.run_module('line2')

        data = self.pipeline.get_data('science_line2')
        assert np.sum(data) == pytest.approx(106.09825573198366, rel=self.limit, abs=0.)
        assert data.shape == (10, 11, 11)
