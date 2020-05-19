import os

import h5py
import pytest
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.fitsreading import FitsReadingModule
from pynpoint.processing.resizing import CropImagesModule, ScaleImagesModule, \
                                         AddLinesModule, RemoveLinesModule
from pynpoint.util.tests import create_config, create_star_data, remove_test_data


class TestResizing:

    def setup_class(self) -> None:

        self.limit = 1e-10
        self.test_dir = os.path.dirname(__file__) + '/'

        create_star_data(self.test_dir+'resize')
        create_config(self.test_dir+'PynPoint_config.ini')

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self) -> None:

        remove_test_data(self.test_dir, folders=['resize'])

    def test_read_data(self) -> None:

        module = FitsReadingModule(name_in='read',
                                   image_tag='read',
                                   input_dir=self.test_dir+'resize',
                                   overwrite=True,
                                   check=True)

        self.pipeline.add_module(module)
        self.pipeline.run_module('read')

        data = self.pipeline.get_data('read')
        assert np.sum(data) == pytest.approx(105.54278879805277, rel=self.limit, abs=0.)
        assert data.shape == (10, 11, 11)

    def test_crop_images(self) -> None:

        module = CropImagesModule(size=0.2,
                                  center=None,
                                  name_in='crop1',
                                  image_in_tag='read',
                                  image_out_tag='crop1')

        self.pipeline.add_module(module)
        self.pipeline.run_module('crop1')

        module = CropImagesModule(size=0.2,
                                  center=(4, 4),
                                  name_in='crop2',
                                  image_in_tag='read',
                                  image_out_tag='crop2')

        self.pipeline.add_module(module)
        self.pipeline.run_module('crop2')

        data = self.pipeline.get_data('crop1')
        assert np.sum(data) == pytest.approx(104.93318507061295, rel=self.limit, abs=0.)
        assert data.shape == (10, 9, 9)

        data = self.pipeline.get_data('crop2')
        assert np.sum(data) == pytest.approx(105.64863165433025, rel=self.limit, abs=0.)
        assert data.shape == (10, 9, 9)

    def test_scale_images(self) -> None:

        module = ScaleImagesModule(scaling=(2., 2., None),
                                   name_in='scale1',
                                   image_in_tag='read',
                                   image_out_tag='scale1')

        self.pipeline.add_module(module)
        self.pipeline.run_module('scale1')

        module = ScaleImagesModule(scaling=(None, None, 2.),
                                   name_in='scale2',
                                   image_in_tag='read',
                                   image_out_tag='scale2')

        self.pipeline.add_module(module)
        self.pipeline.run_module('scale2')

        data = self.pipeline.get_data('scale1')
        assert np.sum(data) == pytest.approx(105.54278879805277, rel=self.limit, abs=0.)
        assert data.shape == (10, 22, 22)

        data = self.pipeline.get_data('scale2')
        assert np.sum(data) == pytest.approx(211.08557759610554, rel=self.limit, abs=0.)
        assert data.shape == (10, 11, 11)

    def test_add_lines(self) -> None:

        module = AddLinesModule(lines=(2, 5, 0, 3),
                                name_in='add',
                                image_in_tag='read',
                                image_out_tag='add')

        self.pipeline.add_module(module)
        self.pipeline.run_module('add')

        data = self.pipeline.get_data('add')
        assert np.sum(data) == pytest.approx(105.54278879805275, rel=self.limit, abs=0.)
        assert data.shape == (10, 14, 18)

    def test_remove_lines(self) -> None:

        module = RemoveLinesModule(lines=(2, 5, 0, 3),
                                   name_in='remove',
                                   image_in_tag='read',
                                   image_out_tag='remove')

        self.pipeline.add_module(module)
        self.pipeline.run_module('remove')

        data = self.pipeline.get_data('remove')
        assert np.sum(data) == pytest.approx(67.49726677462391, rel=self.limit, abs=0.)
        assert data.shape == (10, 8, 4)

        with h5py.File(self.test_dir+'PynPoint_database.hdf5', 'a') as hdf_file:
            hdf_file['config'].attrs['CPU'] = 4

        self.pipeline.run_module('remove')

        data_multi = self.pipeline.get_data('remove')
        assert data == pytest.approx(data_multi, rel=self.limit, abs=0.)
        assert data.shape == data_multi.shape
