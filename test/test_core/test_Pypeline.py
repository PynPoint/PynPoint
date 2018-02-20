import os
import warnings

import pytest
import h5py
import numpy as np

from astropy.io import fits

from PynPoint.Core import Pypeline
from PynPoint.IOmodules import FitsReadingModule, FitsWritingModule
from PynPoint.ProcessingModules import BadPixelCleaningSigmaFilterModule

warnings.simplefilter("always")

def setup_module():
    file_in = os.path.dirname(__file__) + "/images.fits"
    config_file = os.path.dirname(__file__) + "/PynPoint_config.ini"

    np.random.seed(1)
    images = np.random.normal(loc=0, scale=2e-4, size=(10, 100, 100))

    hdu = fits.PrimaryHDU()
    header = hdu.header
    header['INSTRUME'] = "IMAGER"
    header['HIERARCH ESO DET EXP NO'] = 1
    header['HIERARCH ESO DET NDIT'] = 10
    header['HIERARCH ESO INS PIXSCALE'] = 0.01
    header['HIERARCH ESO ADA POSANG'] = 10.
    header['HIERARCH ESO ADA POSANG END'] = 20.
    header['HIERARCH ESO SEQ CUMOFFSETX'] = 5.
    header['HIERARCH ESO SEQ CUMOFFSETY'] = 5.
    hdu.data = images
    hdu.writeto(file_in)

    f = open(config_file, 'w')
    f.write('[header]\n\n')
    f.write('INSTRUMENT: INSTRUME\n')
    f.write('NFRAMES: NAXIS3\n')
    f.write('EXP_NO: ESO DET EXP NO\n')
    f.write('NDIT: ESO DET NDIT\n')
    f.write('PARANG_START: ESO ADA POSANG\n')
    f.write('PARANG_END: ESO ADA POSANG END\n')
    f.write('DITHER_X: ESO SEQ CUMOFFSETX\n')
    f.write('DITHER_Y: ESO SEQ CUMOFFSETY\n\n')
    f.write('[settings]\n\n')
    f.write('PIXSCALE: 0.01\n')
    f.write('MEMORY: 100\n')
    f.write('CPU: 1')
    f.close()

def teardown_module():
    file_in = os.path.dirname(__file__) + "/images.fits"
    config_file = os.path.dirname(__file__) + "/PynPoint_config.ini"

    os.remove(file_in)
    os.remove(config_file)

class TestPypeline(object):

    def setup(self):
        self.test_dir = os.path.dirname(__file__) + '/'
        self.test_data = os.path.dirname(__file__) + '/PynPoint_database.hdf5'
        self.test_config = os.path.dirname(__file__) + '/PynPoint_config.ini'

    def test_create_instance_using_existing_database(self):
        np.random.seed(1)
        images = np.random.normal(loc=0, scale=2e-4, size=(10, 100, 100))

        h5f = h5py.File(self.test_data, "w")
        dset = h5f.create_dataset("images", data=images)
        dset.attrs['PIXSCALE'] = 0.01
        h5f.close()

        pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)
        data = pipeline.get_data("images")

        assert data[0, 0, 0] == 0.00032486907273264834
        assert np.mean(data) == 1.0506056979365338e-06
        assert pipeline.get_attribute("images", "PIXSCALE") == 0.01

        os.remove(self.test_data)

    def test_create_instance_missing_directory(self):
        dir_non_exists = self.test_dir + "none/"
        dir_exists = self.test_dir

        with pytest.raises(AssertionError):
            pipeline = Pypeline(dir_non_exists, dir_exists, dir_exists)

        with pytest.raises(AssertionError):
            pipeline = Pypeline(dir_exists, dir_non_exists, dir_non_exists)

        with pytest.raises(AssertionError):
            pipeline = Pypeline()

    def test_create_instance_new_data_base(self):
        pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

        pipeline.m_data_storage.open_connection()
        pipeline.m_data_storage.close_connection()

        del pipeline

        os.remove(self.test_data)

    def test_add_modules(self):
        pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

        # --- Reading Modules ---
        # default location
        reading = FitsReadingModule(name_in="reading",
                                    input_dir=None,
                                    image_tag="im_arr")

        pipeline.add_module(reading)

        # no default location
        reading2 = FitsReadingModule(name_in="reading2",
                                     input_dir=self.test_dir,
                                     image_tag="im_arr2")

        pipeline.add_module(reading2)

        # --- Processing Module ---
        process = BadPixelCleaningSigmaFilterModule(image_in_tag="im_arr")

        pipeline.add_module(process)

        # --- Writing Module ---
        # default location
        write = FitsWritingModule(name_in="writing",
                                  file_name="result.fits",
                                  data_tag="im_arr")

        pipeline.add_module(write)

        # no default location
        write2 = FitsWritingModule(name_in="writing2",
                                   file_name="result.fits",
                                   data_tag="im_arr",
                                   output_dir=self.test_dir)

        pipeline.add_module(write2)

        with pytest.warns(UserWarning):
            pipeline.add_module(write2)

        pipeline.run()

        os.remove(self.test_dir+"result.fits")
        os.remove(self.test_data)

        # --- Reading Module ---
        # run_module

        pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

        reading = FitsReadingModule(name_in="reading",
                                    input_dir=None,
                                    image_tag="im_arr")

        pipeline.add_module(reading)
        pipeline.run_module("reading")

        os.remove(self.test_data)

    def test_add_non_module_as_module(self):
        pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

        with pytest.raises(AssertionError):
            pipeline.add_module(None)

        os.remove(self.test_data)

    def test_run_non_valid_module_list(self):
        pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

        # --- Reading Modules ---
        reading = FitsReadingModule(name_in="reading")

        pipeline.add_module(reading)

        # --- Processing Module ---
        process = BadPixelCleaningSigmaFilterModule(name_in="filter",
                                                    image_in_tag="im_list")

        pipeline.add_module(process)

        # --- Writing Module ---
        write = FitsWritingModule(name_in="writing",
                                  file_name="result.fits",
                                  data_tag="im_list")

        pipeline.add_module(write)

        with pytest.raises(AttributeError):
            pipeline.run()

        with pytest.raises(AttributeError):
            pipeline.run_module("filter")

        with pytest.raises(AttributeError):
            pipeline.run_module("writing")

        assert pipeline.validate_pipeline_module("test") is None

        os.remove(self.test_data)

    def test_run_non_existing_module(self):
        pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

        with pytest.warns(UserWarning):
            pipeline.run_module("test")

        os.remove(self.test_data)

    def test_remove_module(self):
        pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

        # --- Reading Modules ---
        reading = FitsReadingModule(name_in="reading")

        pipeline.add_module(reading)

        # --- Processing Module ---
        process = BadPixelCleaningSigmaFilterModule(name_in="filter",
                                                    image_in_tag="im_list")

        pipeline.add_module(process)

        assert pipeline.get_module_names() == ["reading", "filter"]

        pipeline.remove_module("reading")

        assert pipeline.get_module_names() == ["filter"]

        assert pipeline.remove_module("filter") is True
        
        os.remove(self.test_data)
