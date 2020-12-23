import os

import pytest
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.fitsreading import FitsReadingModule
from pynpoint.processing.stacksubset import StackAndSubsetModule, StackCubesModule, \
                                            DerotateAndStackModule, CombineTagsModule
from pynpoint.util.tests import create_config, create_star_data, create_ifs_data, remove_test_data


class TestStackSubset:

    def setup_class(self) -> None:

        self.limit = 1e-10
        self.test_dir = os.path.dirname(__file__) + '/'

        create_ifs_data(self.test_dir+'data_ifs')
        create_star_data(self.test_dir+'data')
        create_star_data(self.test_dir+'extra')

        create_config(self.test_dir+'PynPoint_config.ini')

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self) -> None:

        remove_test_data(self.test_dir, folders=['data_ifs', 'extra', 'data'])

    def test_read_data(self) -> None:

        module = FitsReadingModule(name_in='read1',
                                   image_tag='images',
                                   input_dir=self.test_dir+'data',
                                   overwrite=True,
                                   check=True)

        self.pipeline.add_module(module)
        self.pipeline.run_module('read1')

        data = self.pipeline.get_data('images')
        assert np.mean(data) == pytest.approx(0.08722544528764692, rel=self.limit, abs=0.)
        assert data.shape == (10, 11, 11)

        module = FitsReadingModule(name_in='read2',
                                   image_tag='extra',
                                   input_dir=self.test_dir+'extra',
                                   overwrite=True,
                                   check=True)

        self.pipeline.add_module(module)
        self.pipeline.run_module('read2')

        extra = self.pipeline.get_data('extra')
        assert data == pytest.approx(extra, rel=self.limit, abs=0.)

        module = FitsReadingModule(name_in='read_ifs',
                                   image_tag='images_ifs',
                                   input_dir=self.test_dir+'data_ifs',
                                   overwrite=True,
                                   check=True,
                                   ifs_data=True)

        self.pipeline.add_module(module)
        self.pipeline.run_module('read_ifs')
        self.pipeline.set_attribute('images_ifs', 'PARANG', np.linspace(0., 180., 10), static=False)

        data = self.pipeline.get_data('images_ifs')
        assert np.sum(data) == pytest.approx(749.8396528807369, rel=self.limit, abs=0.)
        assert data.shape == (3, 10, 21, 21)

    def test_stack_and_subset(self) -> None:

        self.pipeline.set_attribute('images', 'PARANG', np.arange(10.), static=False)

        module = StackAndSubsetModule(name_in='stack1',
                                      image_in_tag='images',
                                      image_out_tag='stack1',
                                      random=4,
                                      stacking=2,
                                      combine='mean',
                                      max_rotation=None)

        self.pipeline.add_module(module)
        self.pipeline.run_module('stack1')

        data = self.pipeline.get_data('stack1')
        assert np.mean(data) == pytest.approx(0.08758276283743936, rel=self.limit, abs=0.)
        assert data.shape == (4, 11, 11)

        data = self.pipeline.get_data('header_stack1/INDEX')
        assert data == pytest.approx(np.arange(4), rel=self.limit, abs=0.)
        assert data.shape == (4, )

        data = self.pipeline.get_data('header_stack1/PARANG')
        assert data == pytest.approx([0.5, 2.5, 6.5, 8.5], rel=self.limit, abs=0.)
        assert data.shape == (4, )

    def test_stack_max_rotation(self) -> None:

        angles = np.arange(10.)
        angles[1:6] = 3.
        angles[9] = 50.

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
        assert np.mean(data) == pytest.approx(0.08580759396987508, rel=self.limit, abs=0.)
        assert data.shape == (7, 11, 11)

        data = self.pipeline.get_data('header_stack2/INDEX')
        assert data == pytest.approx(np.arange(7), rel=self.limit, abs=0.)
        assert data.shape == (7, )

        data = self.pipeline.get_data('header_stack2/PARANG')
        assert data.shape == (7, )

        self.pipeline.set_attribute('images', 'PARANG', np.arange(10.), static=False)

    def test_stack_cube(self) -> None:

        module = StackCubesModule(name_in='stackcube',
                                  image_in_tag='images',
                                  image_out_tag='mean',
                                  combine='mean')

        self.pipeline.add_module(module)
        self.pipeline.run_module('stackcube')

        data = self.pipeline.get_data('mean')
        assert np.mean(data) == pytest.approx(0.08722544528764689, rel=self.limit, abs=0.)
        assert data.shape == (2, 11, 11)

        attribute = self.pipeline.get_attribute('mean', 'INDEX', static=False)
        assert np.mean(attribute) == pytest.approx(0.5, rel=self.limit, abs=0.)
        assert attribute.shape == (2, )

        attribute = self.pipeline.get_attribute('mean', 'NFRAMES', static=False)
        assert np.mean(attribute) == pytest.approx(1, rel=self.limit, abs=0.)
        assert attribute.shape == (2, )

    def test_derotate_and_stack(self) -> None:

        module = DerotateAndStackModule(name_in='derotate1',
                                        image_in_tag='images',
                                        image_out_tag='derotate1',
                                        derotate=True,
                                        stack='mean',
                                        extra_rot=10.)

        self.pipeline.add_module(module)
        self.pipeline.run_module('derotate1')

        data = self.pipeline.get_data('derotate1')
        assert np.mean(data) == pytest.approx(0.08709860116308817, rel=self.limit, abs=0.)
        assert data.shape == (1, 11, 11)

        module = DerotateAndStackModule(name_in='derotate2',
                                        image_in_tag='images',
                                        image_out_tag='derotate2',
                                        derotate=False,
                                        stack='median',
                                        extra_rot=0.)

        self.pipeline.add_module(module)
        self.pipeline.run_module('derotate2')

        data = self.pipeline.get_data('derotate2')
        assert np.mean(data) == pytest.approx(0.0861160094566323, rel=self.limit, abs=0.)
        assert data.shape == (1, 11, 11)

        data = self.pipeline.get_data('derotate2')
        assert np.mean(data) == pytest.approx(0.0861160094566323, rel=self.limit, abs=0.)
        assert data.shape == (1, 11, 11)

        module = DerotateAndStackModule(name_in='derotate_ifs1',
                                        image_in_tag='images_ifs',
                                        image_out_tag='derotate_ifs1',
                                        derotate=True,
                                        stack='mean',
                                        extra_rot=0.,
                                        dimension='time')

        self.pipeline.add_module(module)
        self.pipeline.run_module('derotate_ifs1')

        data = self.pipeline.get_data('derotate_ifs1')
        assert np.mean(data) == pytest.approx(0.1884438996655355, rel=self.limit, abs=0.)
        assert data.shape == (3, 1, 21, 21)

        module = DerotateAndStackModule(name_in='derotate_ifs2',
                                        image_in_tag='images_ifs',
                                        image_out_tag='derotate_ifs2',
                                        derotate=False,
                                        stack='median',
                                        extra_rot=0.,
                                        dimension='wavelength')

        self.pipeline.add_module(module)
        self.pipeline.run_module('derotate_ifs2')

        data = self.pipeline.get_data('derotate_ifs2')
        assert np.mean(data) == pytest.approx(0.055939644983170146, rel=self.limit, abs=0.)
        assert data.shape == (1, 10, 21, 21)

        module = DerotateAndStackModule(name_in='derotate_ifs3',
                                        image_in_tag='images_ifs',
                                        image_out_tag='derotate_ifs3',
                                        derotate=True,
                                        stack=None,
                                        extra_rot=0.,
                                        dimension='wavelength')

        self.pipeline.add_module(module)
        self.pipeline.run_module('derotate_ifs3')

        data = self.pipeline.get_data('derotate_ifs3')
        assert np.mean(data) == pytest.approx(0.05653316989966066, rel=self.limit, abs=0.)
        assert data.shape == (3, 10, 21, 21)

    def test_combine_tags(self) -> None:

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
        assert np.mean(data) == pytest.approx(0.0872254452876469, rel=self.limit, abs=0.)
        assert data.shape == (20, 11, 11)

        data = self.pipeline.get_data('header_combine1/INDEX')
        assert data[19] == 9
        assert data.shape == (20, )

        module = CombineTagsModule(image_in_tags=['images', 'extra'],
                                   check_attr=False,
                                   index_init=True,
                                   name_in='combine2',
                                   image_out_tag='combine2')

        self.pipeline.add_module(module)
        self.pipeline.run_module('combine2')

        data = self.pipeline.get_data('combine1')
        extra = self.pipeline.get_data('combine2')
        assert data == pytest.approx(extra, rel=self.limit, abs=0.)

        data = self.pipeline.get_data('header_combine2/INDEX')
        assert data[19] == 19
        assert data.shape == (20, )
