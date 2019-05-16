import os
import warnings

import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.fitsreading import FitsReadingModule
from pynpoint.processing.extract import ExtractBinaryModule
from pynpoint.processing.psfpreparation import AngleInterpolationModule
from pynpoint.util.tests import create_config, create_fake, remove_test_data

warnings.simplefilter('always')

limit = 1e-10

class TestStarAlignment:

    def setup_class(self):

        self.test_dir = os.path.dirname(__file__) + '/'

        create_fake(path=self.test_dir+'binary',
                    ndit=[20, 20, 20, 20],
                    nframes=[20, 20, 20, 20],
                    exp_no=[1, 2, 3, 4],
                    npix=(101, 101),
                    fwhm=3.,
                    x0=[50, 50, 50, 50],
                    y0=[50, 50, 50, 50],
                    angles=[[0., 25.], [25., 50.], [50., 75.], [75., 100.]],
                    sep=20.,
                    contrast=1.)

        create_config(self.test_dir+'PynPoint_config.ini')

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self):

        remove_test_data(path=self.test_dir, folders=['binary'])

    def test_read_data(self):

        module = FitsReadingModule(name_in='read',
                                   image_tag='binary',
                                   input_dir=self.test_dir+'binary',
                                   overwrite=True,
                                   check=True)

        self.pipeline.add_module(module)
        self.pipeline.run_module('read')

        data = self.pipeline.get_data('binary')
        assert np.allclose(data[0, 50, 50], 0.0986064357966972, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00019636787665654158, rtol=limit, atol=0.)
        assert data.shape == (80, 101, 101)

    def test_angle_interpolation(self):

        module = AngleInterpolationModule(name_in='angle',
                                          data_tag='binary')

        self.pipeline.add_module(module)
        self.pipeline.run_module('angle')

        data = self.pipeline.get_attribute('binary', 'PARANG', static=False)
        assert data[5] == 6.578947368421053
        assert np.allclose(np.mean(data), 50.0, rtol=limit, atol=0.)
        assert data.shape == (80, )

    def test_extract_binary(self):

        module = ExtractBinaryModule(pos_center=(50., 50.),
                                     pos_binary=(50., 70.),
                                     name_in='extract_binary',
                                     image_in_tag='binary',
                                     image_out_tag='extract_binary',
                                     image_size=0.5,
                                     search_size=0.2,
                                     filter_size=None)

        self.pipeline.add_module(module)
        self.pipeline.run_module('extract_binary')

        data = self.pipeline.get_data('extract_binary')

        assert np.allclose(data[0, 9, 9], 0.09774483733119443, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.0027700881940171283, rtol=limit, atol=0.)
        assert data.shape == (80, 19, 19)
