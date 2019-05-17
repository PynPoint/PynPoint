import os
import warnings

import pytest
import numpy as np

from astropy.io import fits

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.fitsreading import FitsReadingModule
from pynpoint.util.tests import create_config, create_star_data, remove_test_data

warnings.simplefilter('always')

limit = 1e-10

class TestFitsReadingModule:

    def setup_class(self):

        self.test_dir = os.path.dirname(__file__) + '/'

        create_star_data(path=self.test_dir+'fits')
        create_config(self.test_dir+'PynPoint_config.ini')

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self):

        remove_test_data(self.test_dir, folders=['fits'], files=['filenames.dat'])

    def test_fits_reading(self):

        module = FitsReadingModule(name_in='read1',
                                   input_dir=self.test_dir+'fits',
                                   image_tag='input',
                                   overwrite=False,
                                   check=True)

        self.pipeline.add_module(module)
        self.pipeline.run_module('read1')

        data = self.pipeline.get_data('input')
        assert np.allclose(data[0, 50, 50], 0.09798413502193704, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010029494781738066, rtol=limit, atol=0.)
        assert data.shape == (40, 100, 100)

    def test_fits_reading_overwrite(self):

        module = FitsReadingModule(name_in='read2',
                                   input_dir=self.test_dir+'fits',
                                   image_tag='input',
                                   overwrite=True,
                                   check=True)

        self.pipeline.add_module(module)
        self.pipeline.run_module('read2')

        data = self.pipeline.get_data('input')
        assert np.allclose(data[0, 50, 50], 0.09798413502193704, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010029494781738066, rtol=limit, atol=0.)
        assert data.shape == (40, 100, 100)

    def test_static_not_found(self):

        self.pipeline.set_attribute('config', 'DIT', 'Test', static=True)

        module = FitsReadingModule(name_in='read3',
                                   input_dir=self.test_dir+'fits',
                                   image_tag='input',
                                   overwrite=True,
                                   check=True)

        self.pipeline.add_module(module)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module('read3')

        assert len(warning) == 4
        for item in warning:
            assert item.message.args[0] == 'Static attribute DIT (=Test) not found in the FITS ' \
                                           'header.'

        self.pipeline.set_attribute('config', 'DIT', 'ESO DET DIT', static=True)

    def test_static_changing(self):

        with fits.open(self.test_dir+'fits/image01.fits') as hdu:
            header = hdu[0].header
            header['HIERARCH ESO DET DIT'] = 0.1
            hdu.writeto(self.test_dir+'fits/image01.fits', overwrite=True)

        with fits.open(self.test_dir+'fits/image02.fits') as hdu:
            header = hdu[0].header
            header['HIERARCH ESO DET DIT'] = 0.1
            hdu.writeto(self.test_dir+'fits/image02.fits', overwrite=True)

        with fits.open(self.test_dir+'fits/image03.fits') as hdu:
            header = hdu[0].header
            header['HIERARCH ESO DET DIT'] = 0.2
            hdu.writeto(self.test_dir+'fits/image03.fits', overwrite=True)

        with fits.open(self.test_dir+'fits/image04.fits') as hdu:
            header = hdu[0].header
            header['HIERARCH ESO DET DIT'] = 0.2
            hdu.writeto(self.test_dir+'fits/image04.fits', overwrite=True)

        module = FitsReadingModule(name_in='read4',
                                   input_dir=self.test_dir+'fits',
                                   image_tag='input',
                                   overwrite=True,
                                   check=True)

        self.pipeline.add_module(module)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module('read4')

        assert len(warning) == 2
        assert warning[0].message.args[0] == f'Static attribute ESO DET DIT has changed. ' \
                                             f'Possibly the current file {self.test_dir}fits/' \
                                             f'image03.fits does not belong to the data set ' \
                                             f'\'input\'. Attribute value is updated.'

        assert warning[1].message.args[0] == f'Static attribute ESO DET DIT has changed. ' \
                                             f'Possibly the current file {self.test_dir}fits/' \
                                             f'image04.fits does not belong to the data set ' \
                                             f'\'input\'. Attribute value is updated.'

    def test_header_attribute(self):

        with fits.open(self.test_dir+'fits/image01.fits') as hdu:
            header = hdu[0].header
            header['PARANG'] = 1.0
            hdu.writeto(self.test_dir+'fits/image01.fits', overwrite=True)

        with fits.open(self.test_dir+'fits/image02.fits') as hdu:
            header = hdu[0].header
            header['PARANG'] = 2.0
            hdu.writeto(self.test_dir+'fits/image02.fits', overwrite=True)

        with fits.open(self.test_dir+'fits/image03.fits') as hdu:
            header = hdu[0].header
            header['PARANG'] = 3.0
            header['HIERARCH ESO DET DIT'] = 0.1
            hdu.writeto(self.test_dir+'fits/image03.fits', overwrite=True)

        with fits.open(self.test_dir+'fits/image04.fits') as hdu:
            header = hdu[0].header
            header['PARANG'] = 4.0
            header['HIERARCH ESO DET DIT'] = 0.1
            hdu.writeto(self.test_dir+'fits/image04.fits', overwrite=True)

        module = FitsReadingModule(name_in='read5',
                                   input_dir=self.test_dir+'fits',
                                   image_tag='input',
                                   overwrite=True,
                                   check=True)

        self.pipeline.add_module(module)
        self.pipeline.run_module('read5')

    def test_non_static_not_found(self):

        self.pipeline.set_attribute('config', 'DIT', 'None', static=True)

        for i in range(1, 5):
            with fits.open(self.test_dir+'fits/image0'+str(i)+'.fits') as hdu:
                header = hdu[0].header
                del header['HIERARCH ESO DET DIT']
                del header['HIERARCH ESO DET EXP NO']
                hdu.writeto(self.test_dir+'fits/image0'+str(i)+'.fits', overwrite=True)

        module = FitsReadingModule(name_in='read6',
                                   input_dir=self.test_dir+'fits',
                                   image_tag='input',
                                   overwrite=True,
                                   check=True)

        self.pipeline.add_module(module)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module('read6')

        assert len(warning) == 4
        for item in warning:
            assert item.message.args[0] == 'Non-static attribute EXP_NO (=ESO DET EXP NO) not ' \
                                           'found in the FITS header.'

    def test_fits_read_files(self):

        module = FitsReadingModule(name_in='read7',
                                   input_dir=None,
                                   image_tag='files',
                                   overwrite=False,
                                   check=True,
                                   filenames=[self.test_dir+'fits/image01.fits',
                                              self.test_dir+'fits/image03.fits'])

        self.pipeline.add_module(module)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module('read7')

        assert len(warning) == 2
        for item in warning:
            assert item.message.args[0] == 'Non-static attribute EXP_NO (=ESO DET EXP NO) not ' \
                                           'found in the FITS header.'

        data = self.pipeline.get_data('files')
        assert np.allclose(data[0, 50, 50], 0.09798413502193704, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010032245393723324, rtol=limit, atol=0.)
        assert data.shape == (20, 100, 100)

    def test_fits_read_textfile(self):

        with open(self.test_dir+'filenames.dat', 'w') as file_obj:
            file_obj.write(self.test_dir+'fits/image01.fits\n')
            file_obj.write(self.test_dir+'fits/image03.fits\n')

        module = FitsReadingModule(name_in='read8',
                                   input_dir=None,
                                   image_tag='files',
                                   overwrite=True,
                                   check=True,
                                   filenames=self.test_dir+'filenames.dat')

        self.pipeline.add_module(module)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module('read8')

        assert len(warning) == 2
        for item in warning:
            assert item.message.args[0] == 'Non-static attribute EXP_NO (=ESO DET EXP NO) not ' \
                                           'found in the FITS header.'

        data = self.pipeline.get_data('files')
        assert np.allclose(data[0, 50, 50], 0.09798413502193704, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010032245393723324, rtol=limit, atol=0.)
        assert data.shape == (20, 100, 100)

    def test_fits_read_files_exists(self):

        module = FitsReadingModule(name_in='read9',
                                   input_dir=None,
                                   image_tag='files',
                                   overwrite=True,
                                   check=True,
                                   filenames=['fits/image00.fits',
                                              'fits/image03.fits'])

        self.pipeline.add_module(module)

        with pytest.raises(ValueError) as error:
            self.pipeline.run_module('read9')

        assert str(error.value) == f'The file fits/image00.fits does not exist. '\
        'Please check that the path is correct.'

    def test_fits_read_textfile_exists(self):

        with open(self.test_dir+'filenames.dat', 'w') as file_obj:
            file_obj.write('fits/image00.fits\n')
            file_obj.write('fits/image03.fits\n')

        module = FitsReadingModule(name_in='read10',
                                   input_dir=None,
                                   image_tag='files',
                                   overwrite=True,
                                   check=True,
                                   filenames=self.test_dir+'filenames.dat')

        self.pipeline.add_module(module)

        with pytest.raises(ValueError) as error:
            self.pipeline.run_module('read10')

        assert str(error.value) == f'The file fits/image00.fits does not exist. '\
        'Please check that the path is correct.'
