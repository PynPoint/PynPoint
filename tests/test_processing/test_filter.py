import os
import warnings

import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.fitsreading import FitsReadingModule
from pynpoint.processing.filter import GaussianFilterModule
from pynpoint.util.tests import create_config, remove_test_data, create_star_data

warnings.simplefilter('always')

limit = 1e-10


class TestFilter:

    def setup_class(self):

        self.test_dir = os.path.dirname(__file__) + '/'

        create_star_data(self.test_dir+'data')
        create_config(self.test_dir+'PynPoint_config.ini')

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self):

        remove_test_data(self.test_dir, folders=['data'])

    def test_read_data(self):

        module = FitsReadingModule(name_in='read',
                                   image_tag='data',
                                   input_dir=self.test_dir+'data',
                                   overwrite=True,
                                   check=True)

        self.pipeline.add_module(module)

        self.pipeline.run_module('read')

        data = self.pipeline.get_data('data')
        assert np.allclose(data[0, 50, 50], 0.09798413502193704, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010029494781738066, rtol=limit, atol=0.)
        assert data.shape == (40, 100, 100)

    def test_gaussian_filter(self):

        module = GaussianFilterModule(name_in='filter',
                                      image_in_tag='data',
                                      image_out_tag='filtered',
                                      fwhm=0.1)

        self.pipeline.add_module(module)
        self.pipeline.run_module('filter')

        data = self.pipeline.get_data('filtered')
        assert np.allclose(data[0, 50, 50], 0.0388143943049942, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010029494781738068, rtol=limit, atol=0.)
        assert data.shape == (40, 100, 100)
