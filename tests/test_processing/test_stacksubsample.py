import os
import warnings

import pytest
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.fitsreading import FitsReadingModule
from pynpoint.processing.stacksubset import StackAndSubsetModule, StackCubesModule, \
                                            DerotateAndStackModule, CombineTagsModule
from pynpoint.util.tests import create_config, create_star_data, remove_test_data

warnings.simplefilter('always')

limit = 1e-10


class TestStackSubset:

    def setup_class(self):

        self.test_dir = os.path.dirname(__file__) + '/'

        create_star_data(path=self.test_dir+'data')
        create_star_data(path=self.test_dir+'extra')

        create_config(self.test_dir+'PynPoint_config.ini')

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self):

        remove_test_data(self.test_dir, folders=['data', 'extra'])

    def test_read_data(self):

        module = FitsReadingModule(name_in='read1',
                                   image_tag='images',
                                   input_dir=self.test_dir+'data',
                                   overwrite=True,
                                   check=True)

        self.pipeline.add_module(module)
        self.pipeline.run_module('read1')

        data = self.pipeline.get_data('images')
        assert np.allclose(data[0, 50, 50], 0.09798413502193704, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010029494781738066, rtol=limit, atol=0.)
        assert data.shape == (40, 100, 100)

        module = FitsReadingModule(name_in='read2',
                                   image_tag='extra',
                                   input_dir=self.test_dir+'extra',
                                   overwrite=True,
                                   check=True)

        self.pipeline.add_module(module)
        self.pipeline.run_module('read2')

        extra = self.pipeline.get_data('extra')
        assert np.allclose(data, extra, rtol=limit, atol=0.)

    def test_stack_and_subset(self):

        self.pipeline.set_attribute('images', 'PARANG', np.arange(1., 41., 1.), static=False)

        module = StackAndSubsetModule(name_in='stack1',
                                      image_in_tag='images',
                                      image_out_tag='stack1',
                                      random=10,
                                      stacking=2,
                                      combine='mean',
                                      max_rotation=None)

        self.pipeline.add_module(module)
        self.pipeline.run_module('stack1')

        data = self.pipeline.get_data('stack1')
        assert np.allclose(data[0, 50, 50], 0.09816320034649725, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 9.983545774937238e-05, rtol=limit, atol=0.)
        assert data.shape == (10, 100, 100)

        data = self.pipeline.get_data('header_stack1/INDEX')
        assert np.allclose(data, np.arange(10), rtol=limit, atol=0.)
        assert data.shape == (10, )

        data = self.pipeline.get_data('header_stack1/PARANG')
        parang = [1.5, 15.5, 19.5, 23.5, 25.5, 29.5, 31.5, 35.5, 37.5, 39.5]
        assert np.allclose(data, parang, rtol=limit, atol=0.)
        assert data.shape == (10, )

    def test_stack_max_rotation(self):

        angles = np.arange(1., 41., 1.)
        angles[1:6] = 3.
        angles[39] = 50.

        self.pipeline.set_attribute('images', 'PARANG', angles, static=False)

        module = StackAndSubsetModule(name_in='stack2',
                                      image_in_tag='images',
                                      image_out_tag='stack2',
                                      random=None,
                                      stacking=2,
                                      combine='median',
                                      max_rotation=1.)

        self.pipeline.add_module(module)
        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module('stack2')

        assert len(warning) == 1
        assert warning[0].message.args[0] == 'Testing of util.module.stack_angles has been ' \
                                             'limited, please use carefully.'

        data = self.pipeline.get_data('stack2')
        assert np.allclose(data[0, 50, 50], 0.09798413502193704, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010035876175210947, rtol=limit, atol=0.)
        assert data.shape == (22, 100, 100)

        data = self.pipeline.get_data('header_stack2/INDEX')
        assert np.allclose(data, np.arange(22), rtol=limit, atol=0.)
        assert data.shape == (22, )

        data = self.pipeline.get_data('header_stack2/PARANG')
        assert data.shape == (22, )

        self.pipeline.set_attribute('images', 'PARANG', np.arange(1., 41., 1.), static=False)

    def test_stack_cube(self):

        module = StackCubesModule(name_in='stackcube',
                                  image_in_tag='images',
                                  image_out_tag='mean',
                                  combine='mean')

        self.pipeline.add_module(module)
        self.pipeline.run_module('stackcube')

        data = self.pipeline.get_data('mean')
        assert np.allclose(data[0, 50, 50], 0.09805840100024205, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010029494781738069, rtol=limit, atol=0.)
        assert data.shape == (4, 100, 100)

        attribute = self.pipeline.get_attribute('mean', 'INDEX', static=False)
        assert np.allclose(np.mean(attribute), 1.5, rtol=limit, atol=0.)
        assert attribute.shape == (4, )

        attribute = self.pipeline.get_attribute('mean', 'NFRAMES', static=False)
        assert np.allclose(np.mean(attribute), 1, rtol=limit, atol=0.)
        assert attribute.shape == (4, )

    def test_derotate_and_stack(self):

        module = DerotateAndStackModule(name_in='derotate1',
                                        image_in_tag='images',
                                        image_out_tag='derotate1',
                                        derotate=True,
                                        stack='mean',
                                        extra_rot=10.)

        self.pipeline.add_module(module)
        self.pipeline.run_module('derotate1')

        data = self.pipeline.get_data('derotate1')
        assert np.allclose(data[0, 50, 50], 0.09689679769268554, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010021671152246617, rtol=limit, atol=0.)
        assert data.shape == (1, 100, 100)

        module = DerotateAndStackModule(name_in='derotate2',
                                        image_in_tag='images',
                                        image_out_tag='derotate2',
                                        derotate=False,
                                        stack='median',
                                        extra_rot=0.)

        self.pipeline.add_module(module)
        self.pipeline.run_module('derotate2')

        data = self.pipeline.get_data('derotate2')
        assert np.allclose(data[0, 50, 50], 0.09809001768003645, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010033064394962, rtol=limit, atol=0.)
        assert data.shape == (1, 100, 100)

    def test_combine_tags(self):

        module = CombineTagsModule(image_in_tags=['images', 'extra'],
                                   check_attr=True,
                                   index_init=False,
                                   name_in='combine1',
                                   image_out_tag='combine1')

        self.pipeline.add_module(module)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module('combine1')

        assert len(warning) == 1
        assert warning[0].message.args[0] == 'The non-static keyword FILES is already used but ' \
                                             'with different values. It is advisable to only ' \
                                             'combine tags that descend from the same data set.'

        data = self.pipeline.get_data('combine1')
        assert np.allclose(data[0, 50, 50], 0.09798413502193704, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010029494781738068, rtol=limit, atol=0.)
        assert data.shape == (80, 100, 100)

        data = self.pipeline.get_data('header_combine1/INDEX')
        assert data[40] == 0
        assert data.shape == (80, )

        module = CombineTagsModule(image_in_tags=['images', 'extra'],
                                   check_attr=False,
                                   index_init=True,
                                   name_in='combine2',
                                   image_out_tag='combine2')

        self.pipeline.add_module(module)
        self.pipeline.run_module('combine2')

        data = self.pipeline.get_data('combine1')
        extra = self.pipeline.get_data('combine2')
        assert np.allclose(data, extra, rtol=limit, atol=0.)

        data = self.pipeline.get_data('header_combine2/INDEX')
        assert data[40] == 40
        assert data.shape == (80, )
