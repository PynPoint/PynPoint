import os
import warnings

import pytest
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.nearreading import NearReadingModule
from pynpoint.util.tests import create_config, create_near_data, remove_test_data

warnings.simplefilter('always')

limit = 1e-6

class TestNearInitModule(object):

    def setup_class(self):

        self.test_dir = os.path.dirname(__file__) + '/'

        create_near_data(path=self.test_dir + 'near')
        create_config(self.test_dir + 'PynPoint_config.ini')

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

        self.pipeline.set_attribute('config', 'NFRAMES', 'None', static=True)
        self.pipeline.set_attribute('config', 'EXP_NO', 'ESO TPL EXPNO', static=True)
        self.pipeline.set_attribute('config', 'NDIT', 'None', static=True)
        self.pipeline.set_attribute('config', 'PARANG_START', 'None', static=True)
        self.pipeline.set_attribute('config', 'PARANG_END', 'None', static=True)
        self.pipeline.set_attribute('config', 'DITHER_X', 'None', static=True)
        self.pipeline.set_attribute('config', 'DITHER_Y', 'None', static=True)
        self.pipeline.set_attribute('config', 'PIXSCALE', 0.045, static=True)
        self.pipeline.set_attribute('config', 'MEMORY', 100, static=True)

        self.positions = ('chopa', 'chopb')

    def teardown_class(self):

        remove_test_data(self.test_dir, folders=['near'])

    def test_near_read(self):

        module = NearReadingModule(name_in='read1',
                                   input_dir=self.test_dir+'near',
                                   chopa_out_tag=self.positions[0],
                                   chopb_out_tag=self.positions[1])

        self.pipeline.add_module(module)
        self.pipeline.run_module('read1')

        for item in self.positions:

            data = self.pipeline.get_data(item)
            assert np.allclose(np.mean(data), 0.060582854, rtol=limit, atol=0.)
            assert data.shape == (20, 10, 10)

    def test_static_not_found(self):

        self.pipeline.set_attribute('config', 'DIT', 'Test', static=True)

        module = NearReadingModule(name_in='read5',
                                   input_dir=self.test_dir+'near',
                                   chopa_out_tag=self.positions[0],
                                   chopb_out_tag=self.positions[1])

        self.pipeline.add_module(module)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module('read5')

        assert len(warning) == 8
        for item in warning:
            assert item.message.args[0] == 'Static attribute DIT (=Test) not found in the FITS ' \
                                           'header.'

        self.pipeline.set_attribute('config', 'DIT', 'ESO DET SEQ1 DIT', static=True)

    def test_nonstatic_not_found(self):

        self.pipeline.set_attribute('config', 'NDIT', 'Test', static=True)

        module = NearReadingModule(name_in='read6',
                                   input_dir=self.test_dir+'near',
                                   chopa_out_tag=self.positions[0],
                                   chopb_out_tag=self.positions[1])

        self.pipeline.add_module(module)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module('read6')

        assert len(warning) == 8
        for item in warning:
            assert item.message.args[0] == 'Non-static attribute NDIT (=Test) not found in the ' \
                                           'FITS header.'
