import os

import pytest
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.fitsreading import FitsReadingModule
from pynpoint.processing.basic import SubtractImagesModule, AddImagesModule, RotateImagesModule, \
                                      RepeatImagesModule
from pynpoint.util.tests import create_config, remove_test_data, create_star_data


class TestBasic:

    def setup_class(self) -> None:

        self.limit = 1e-10
        self.test_dir = os.path.dirname(__file__) + '/'

        create_star_data(self.test_dir+'data1')
        create_star_data(self.test_dir+'data2')
        create_star_data(self.test_dir+'data3')

        create_config(self.test_dir+'PynPoint_config.ini')

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self) -> None:

        remove_test_data(self.test_dir, folders=['data1', 'data2', 'data3'])

    def test_read_data(self) -> None:

        module = FitsReadingModule(name_in='read1',
                                   image_tag='data1',
                                   input_dir=self.test_dir+'data1',
                                   overwrite=True,
                                   check=True)

        self.pipeline.add_module(module)

        module = FitsReadingModule(name_in='read2',
                                   image_tag='data2',
                                   input_dir=self.test_dir+'data2',
                                   overwrite=True,
                                   check=True)

        self.pipeline.add_module(module)

        module = FitsReadingModule(name_in='read3',
                                   image_tag='data3',
                                   input_dir=self.test_dir+'data3',
                                   overwrite=True,
                                   check=True)

        self.pipeline.add_module(module)

        self.pipeline.run_module('read1')
        self.pipeline.run_module('read2')
        self.pipeline.run_module('read3')

        data = self.pipeline.get_data('data1')
        assert np.sum(data) == pytest.approx(10.006938694903914, rel=self.limit, abs=0.)
        assert data.shape == (10, 11, 11)

        data = self.pipeline.get_data('data2')
        assert np.sum(data) == pytest.approx(10.006938694903914, rel=self.limit, abs=0.)
        assert data.shape == (10, 11, 11)

        data = self.pipeline.get_data('data3')
        assert np.sum(data) == pytest.approx(10.006938694903914, rel=self.limit, abs=0.)
        assert data.shape == (10, 11, 11)

    def test_subtract_images(self) -> None:

        module = SubtractImagesModule(name_in='subtract',
                                      image_in_tags=('data1', 'data2'),
                                      image_out_tag='subtract',
                                      scaling=1.)

        self.pipeline.add_module(module)
        self.pipeline.run_module('subtract')

        data = self.pipeline.get_data('subtract')
        assert np.sum(data) == pytest.approx(0., rel=self.limit, abs=0.)
        assert data.shape == (10, 11, 11)

    def test_add_images(self) -> None:

        module = AddImagesModule(name_in='add',
                                 image_in_tags=('data1', 'data2'),
                                 image_out_tag='add',
                                 scaling=1.)

        self.pipeline.add_module(module)
        self.pipeline.run_module('add')

        data = self.pipeline.get_data('add')
        assert np.sum(data) == pytest.approx(20.013877389807828, rel=self.limit, abs=0.)
        assert data.shape == (10, 11, 11)

    def test_rotate_images(self) -> None:

        module = RotateImagesModule(name_in='rotate',
                                    image_in_tag='data1',
                                    image_out_tag='rotate',
                                    angle=10.)

        self.pipeline.add_module(module)
        self.pipeline.run_module('rotate')

        data = self.pipeline.get_data('rotate')
        assert np.sum(data) == pytest.approx(10.004703952129002, rel=self.limit, abs=0.)
        assert data.shape == (10, 11, 11)

    def test_repeat_images(self) -> None:

        module = RepeatImagesModule(name_in='repeat',
                                    image_in_tag='data1',
                                    image_out_tag='repeat',
                                    repeat=2)

        self.pipeline.add_module(module)
        self.pipeline.run_module('repeat')

        data1 = self.pipeline.get_data('data1')
        assert data1.shape == (10, 11, 11)

        data2 = self.pipeline.get_data('repeat')
        assert data2.shape == (20, 11, 11)

        assert data1 == pytest.approx(data2[0:10, ], rel=self.limit, abs=0.)
        assert data1 == pytest.approx(data2[10:20, ], rel=self.limit, abs=0.)
