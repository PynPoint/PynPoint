from __future__ import absolute_import

import os
import warnings

import pytest
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.nearreading import NearReadingModule
from pynpoint.util.tests import create_near_config, create_near_data, remove_test_data

warnings.simplefilter('always')

limit = 1e-6

class TestNearInitModule(object):

    def setup_class(self):

        self.test_dir = os.path.dirname(__file__) + '/'

        create_near_data(path=self.test_dir + 'near')
        create_near_config(self.test_dir + 'PynPoint_config.ini')

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

        self.positions = ('noda_chopa', 'noda_chopb', 'nodb_chopa', 'nodb_chopb')

    def teardown_class(self):

        remove_test_data(self.test_dir, folders=['near'])

    def test_near_read(self):

        module = NearReadingModule(name_in='read1',
                                   input_dir=self.test_dir+'near',
                                   noda_chopa_tag=self.positions[0],
                                   noda_chopb_tag=self.positions[1],
                                   nodb_chopa_tag=self.positions[2],
                                   nodb_chopb_tag=self.positions[3],
                                   scheme='ABBA')

        self.pipeline.add_module(module)
        self.pipeline.run_module('read1')

        for item in self.positions:

            data = self.pipeline.get_data(item)
            assert np.allclose(np.mean(data), 0.060582854, rtol=limit, atol=0.)
            assert data.shape == (10, 10, 10)

    def test_near_read_scheme(self):

        module = NearReadingModule(name_in='read2',
                                   input_dir=self.test_dir+'near',
                                   noda_chopa_tag=self.positions[0],
                                   noda_chopb_tag=self.positions[1],
                                   nodb_chopa_tag=self.positions[2],
                                   nodb_chopb_tag=self.positions[3],
                                   scheme='ABAB')

        self.pipeline.add_module(module)
        self.pipeline.run_module('read2')

        for item in self.positions:

            data = self.pipeline.get_data(item)
            assert np.allclose(np.mean(data), 0.060582854, rtol=limit, atol=0.)
            assert data.shape == (10, 10, 10)

    def test_near_read_tag_check(self):

        with pytest.raises(ValueError) as error:
            NearReadingModule(name_in='read3',
                              input_dir=self.test_dir+'near',
                              noda_chopa_tag=self.positions[0],
                              noda_chopb_tag=self.positions[0],
                              nodb_chopa_tag=self.positions[2],
                              nodb_chopb_tag=self.positions[3],
                              scheme='ABBA')

        assert str(error.value) == 'Output ports should have different name tags.'

    def test_near_read_scheme_check(self):

        with pytest.raises(ValueError) as error:
            NearReadingModule(name_in='read4',
                              input_dir=self.test_dir+'near',
                              noda_chopa_tag=self.positions[0],
                              noda_chopb_tag=self.positions[1],
                              nodb_chopa_tag=self.positions[2],
                              nodb_chopb_tag=self.positions[3],
                              scheme='test')

        assert str(error.value) == 'Nodding scheme argument should be set to \'ABBA\' or \'ABAB\'.'

    def test_static_not_found(self):

        self.pipeline.set_attribute('config', 'DIT', 'Test', static=True)

        module = NearReadingModule(name_in='read5',
                                   input_dir=self.test_dir+'near',
                                   noda_chopa_tag=self.positions[0],
                                   noda_chopb_tag=self.positions[1],
                                   nodb_chopa_tag=self.positions[2],
                                   nodb_chopb_tag=self.positions[3],
                                   scheme='ABBA')

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
                                   noda_chopa_tag=self.positions[0],
                                   noda_chopb_tag=self.positions[1],
                                   nodb_chopa_tag=self.positions[2],
                                   nodb_chopb_tag=self.positions[3],
                                   scheme='ABBA')

        self.pipeline.add_module(module)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module('read6')

        assert len(warning) == 8
        for item in warning:
            assert item.message.args[0] == 'Non-static attribute NDIT (=Test) not found in the ' \
                                           'FITS header.'
