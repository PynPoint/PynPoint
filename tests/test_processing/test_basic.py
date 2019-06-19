import os
import warnings

import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.fitsreading import FitsReadingModule
from pynpoint.processing.basic import SubtractImagesModule, AddImagesModule, RotateImagesModule, \
                                      RepeatImagesModule
from pynpoint.util.tests import create_config, remove_test_data, create_star_data

warnings.simplefilter('always')

limit = 1e-10


class TestBasic:

    def setup_class(self):

        self.test_dir = os.path.dirname(__file__) + '/'

        create_star_data(path=self.test_dir+'data1',
                         npix_x=100,
                         npix_y=100,
                         x0=[50, 50, 50, 50],
                         y0=[50, 50, 50, 50],
                         parang_start=[0., 25., 50., 75.],
                         parang_end=[25., 50., 75., 100.],
                         exp_no=[1, 2, 3, 4])

        create_star_data(path=self.test_dir+'data2',
                         npix_x=100,
                         npix_y=100,
                         x0=[50, 50, 50, 50],
                         y0=[50, 50, 50, 50],
                         parang_start=[0., 25., 50., 75.],
                         parang_end=[25., 50., 75., 100.],
                         exp_no=[1, 2, 3, 4])

        create_star_data(path=self.test_dir+'data3',
                         npix_x=100,
                         npix_y=100,
                         x0=[50, 50, 50, 50],
                         y0=[50, 50, 50, 50],
                         parang_start=[0., 25., 50., 75.],
                         parang_end=[25., 50., 75., 100.],
                         exp_no=[1, 2, 3, 4])

        create_config(self.test_dir+'PynPoint_config.ini')

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self):

        remove_test_data(self.test_dir, folders=['data1', 'data2', 'data3'])

    def test_read_data(self):

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
        assert np.allclose(data[0, 50, 50], 0.09798413502193704, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010029494781738066, rtol=limit, atol=0.)
        assert data.shape == (40, 100, 100)

        data = self.pipeline.get_data('data2')
        assert np.allclose(data[0, 50, 50], 0.09798413502193704, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010029494781738066, rtol=limit, atol=0.)
        assert data.shape == (40, 100, 100)

        data = self.pipeline.get_data('data3')
        assert np.allclose(data[0, 50, 50], 0.09798413502193704, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010029494781738066, rtol=limit, atol=0.)
        assert data.shape == (40, 100, 100)

    def test_subtract_images(self):

        module = SubtractImagesModule(name_in='subtract',
                                      image_in_tags=('data1', 'data2'),
                                      image_out_tag='subtract',
                                      scaling=1.)

        self.pipeline.add_module(module)
        self.pipeline.run_module('subtract')

        data = self.pipeline.get_data('subtract')
        assert np.allclose(data[0, 50, 50], 0.0, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.0, rtol=limit, atol=0.)
        assert data.shape == (40, 100, 100)

    def test_add_images(self):

        module = AddImagesModule(name_in='add',
                                 image_in_tags=('data1', 'data2'),
                                 image_out_tag='add',
                                 scaling=1.)

        self.pipeline.add_module(module)
        self.pipeline.run_module('add')

        data = self.pipeline.get_data('add')
        assert np.allclose(data[0, 50, 50], 0.19596827004387407, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00020058989563476133, rtol=limit, atol=0.)
        assert data.shape == (40, 100, 100)

    def test_rotate_images(self):

        module = RotateImagesModule(name_in='rotate',
                                    image_in_tag='data1',
                                    image_out_tag='rotate',
                                    angle=10.)

        self.pipeline.add_module(module)
        self.pipeline.run_module('rotate')

        data = self.pipeline.get_data('rotate')
        assert np.allclose(data[0, 50, 50], 0.09746600632363736, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010030089755226848, rtol=limit, atol=0.)
        assert data.shape == (40, 100, 100)

    def test_repeat_images(self):

        module = RepeatImagesModule(name_in='repeat',
                                    image_in_tag='data1',
                                    image_out_tag='repeat',
                                    repeat=5)

        self.pipeline.add_module(module)
        self.pipeline.run_module('repeat')

        data1 = self.pipeline.get_data('data1')
        assert data1.shape == (40, 100, 100)

        data2 = self.pipeline.get_data('repeat')
        assert data2.shape == (200, 100, 100)

        assert np.allclose(data1, data2[0:40], rtol=limit, atol=0.)
        assert np.allclose(data1, data2[40:80], rtol=limit, atol=0.)
        assert np.allclose(data1, data2[80:120], rtol=limit, atol=0.)
        assert np.allclose(data1, data2[120:160], rtol=limit, atol=0.)
        assert np.allclose(data1, data2[160:200], rtol=limit, atol=0.)
