import os
import warnings

import pytest
import numpy as np

from astropy.io import fits

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.nearreading import NearReadingModule
from pynpoint.util.tests import create_config, create_near_data, remove_test_data

warnings.simplefilter('always')

limit = 1e-6

class TestNearInitModule(object):

    def setup_class(self):

        self.test_dir = os.path.dirname(__file__) + '/'
        self.fitsfile = self.test_dir + 'near/images_1.fits'

        create_near_data(path=self.test_dir + 'near')
        create_config(self.test_dir + 'PynPoint_config.ini')

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

        self.pipeline.set_attribute('config', 'NFRAMES', 'ESO DET CHOP NCYCLES', static=True)
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

        module = NearReadingModule(name_in='read2',
                                   input_dir=self.test_dir+'near',
                                   chopa_out_tag=self.positions[0],
                                   chopb_out_tag=self.positions[1])

        self.pipeline.add_module(module)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module('read2')

        assert len(warning) == 8
        for item in warning:
            assert item.message.args[0] == 'Static attribute DIT (=Test) not found in the FITS ' \
                                           'header.'

        self.pipeline.set_attribute('config', 'DIT', 'ESO DET SEQ1 DIT', static=True)

    def test_nonstatic_not_found(self):

        self.pipeline.set_attribute('config', 'NDIT', 'Test', static=True)

        module = NearReadingModule(name_in='read3',
                                   input_dir=self.test_dir+'near',
                                   chopa_out_tag=self.positions[0],
                                   chopb_out_tag=self.positions[1])

        self.pipeline.add_module(module)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module('read3')

        assert len(warning) == 8
        for item in warning:
            assert item.message.args[0] == 'Non-static attribute NDIT (=Test) not found in the ' \
                                           'FITS header.'

        self.pipeline.set_attribute('config', 'NDIT', 'None', static=True)

    def test_check_header(self):

        with fits.open(self.fitsfile) as hdulist:
            hdulist[0].header['ESO DET CHOP ST'] = 'F'
            hdulist[0].header['ESO DET CHOP CYCSKIP'] = 1
            hdulist[0].header['ESO DET CHOP CYCSUM'] = 'T'
            hdulist.writeto(self.fitsfile, overwrite=True)

        module = NearReadingModule(name_in='read4',
                                   input_dir=self.test_dir+'near',
                                   chopa_out_tag=self.positions[0],
                                   chopb_out_tag=self.positions[1])

        self.pipeline.add_module(module)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module('read4')

        assert len(warning) == 3
        assert warning[0].message.args[0] == 'Dataset was obtained without chopping.'
        assert warning[1].message.args[0] == 'Chop cycles (1) have been skipped.'
        assert warning[2].message.args[0] == 'FITS file contains averaged images.'

        with fits.open(self.fitsfile) as hdulist:
            hdulist[0].header['ESO DET CHOP ST'] = 'T'
            hdulist[0].header['ESO DET CHOP CYCSKIP'] = 0
            hdulist[0].header['ESO DET CHOP CYCSUM'] = 'F'
            hdulist.writeto(self.fitsfile, overwrite=True)

    def test_frame_type_invalid(self):

        with fits.open(self.fitsfile) as hdulist:
            hdulist[10].header['ESO DET FRAM TYPE'] = 'Test'
            hdulist.writeto(self.fitsfile, overwrite=True)

        module = NearReadingModule(name_in='read5',
                                   input_dir=self.test_dir+'near',
                                   chopa_out_tag=self.positions[0],
                                   chopb_out_tag=self.positions[1])

        self.pipeline.add_module(module)

        with pytest.raises(ValueError) as error:
            self.pipeline.run_module('read5')

        assert str(error.value) == 'Frame type (Test) not a valid value. Expecting HCYCLE1 or ' \
                                   'HCYCLE2 as value for ESO DET FRAM TYPE.'

    def test_frame_type_missing(self):

        with fits.open(self.fitsfile) as hdulist:
            hdulist[10].header.remove('ESO DET FRAM TYPE')
            hdulist.writeto(self.fitsfile, overwrite=True)

        module = NearReadingModule(name_in='read6',
                                   input_dir=self.test_dir+'near',
                                   chopa_out_tag=self.positions[0],
                                   chopb_out_tag=self.positions[1])

        self.pipeline.add_module(module)

        with pytest.raises(ValueError) as error:
            self.pipeline.run_module('read6')

        assert str(error.value) == 'Frame type not found in the FITS header. Image number: 9.'

    def test_same_cycle(self):

        with fits.open(self.fitsfile) as hdulist:

            with pytest.warns(UserWarning) as warning:
                hdulist[10].header['ESO DET FRAM TYPE'] = 'HCYCLE1'

            assert len(warning) == 1
            assert warning[0].message.args[0] == 'Keyword name \'ESO DET FRAM TYPE\' is greater ' \
                                                 'than 8 characters or contains characters not ' \
                                                 'allowed by the FITS standard; a HIERARCH card ' \
                                                 'will be created.'

            hdulist.writeto(self.fitsfile, overwrite=True)

        module = NearReadingModule(name_in='read7',
                                   input_dir=self.test_dir+'near',
                                   chopa_out_tag=self.positions[0],
                                   chopb_out_tag=self.positions[1])

        self.pipeline.add_module(module)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module('read7')

        assert len(warning) == 2

        assert warning[0].message.args[0] == 'Previous and current chop position (HCYCLE1) are ' \
                                             'the same. Skipping the current image.'

        assert warning[1].message.args[0] == 'The number of images is not equal for chop A and ' \
                                             'chop B.'

    def test_odd_number_images(self):

        with fits.open(self.fitsfile) as hdulist:
            del hdulist[11]
            hdulist.writeto(self.fitsfile, overwrite=True)

        module = NearReadingModule(name_in='read8',
                                   input_dir=self.test_dir+'near',
                                   chopa_out_tag=self.positions[0],
                                   chopb_out_tag=self.positions[1])

        self.pipeline.add_module(module)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module('read8')

        assert len(warning) == 2

        assert warning[0].message.args[0] == f'FITS file contains odd number of images: ' \
                                             f'{self.fitsfile}'

        assert warning[1].message.args[0] == 'The number of chop cycles (5) is not equal to ' \
                                             'half the number of available HDU images (4).'
