from __future__ import absolute_import
from __future__ import print_function

import os
import warnings

import pytest
import h5py
import numpy as np

from astropy.io import fits

from PynPoint.Core.Pypeline import Pypeline
from PynPoint.IOmodules.FitsReading import FitsReadingModule
from PynPoint.IOmodules.FitsWriting import FitsWritingModule
from PynPoint.ProcessingModules.BadPixelCleaning import BadPixelSigmaFilterModule
from PynPoint.Util.TestTools import create_config, remove_test_data

warnings.simplefilter("always")

limit = 1e-10

class TestPypeline(object):

    def setup_class(self):
        self.test_dir = os.path.dirname(__file__) + "/"

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
        hdu.writeto(self.test_dir+"images.fits")

    def teardown_class(self):
        remove_test_data(self.test_dir, files=["images.fits"])

    def test_create_default_config(self):
        with pytest.warns(UserWarning) as warning:
            Pypeline(self.test_dir, self.test_dir, self.test_dir)

        # assert len(warning) == 2
        assert warning[0].message.args[0] == "Configuration file not found. Creating " \
                                             "PynPoint_config.ini with default values " \
                                             "in the working place."

        with open(self.test_dir+"PynPoint_config.ini") as f_obj:
            count = 0
            for _ in f_obj:
                count += 1

        assert count == 23

    def test_create_none_config(self):
        file_obj = open(self.test_dir+"PynPoint_config.ini", 'w')
        file_obj.write('[header]\n\n')
        file_obj.write('INSTRUMENT: None\n')
        file_obj.write('NFRAMES: None\n')
        file_obj.write('EXP_NO: None\n')
        file_obj.write('NDIT: None\n')
        file_obj.write('PARANG_START: ESO ADA POSANG\n')
        file_obj.write('PARANG_END: None\n')
        file_obj.write('DITHER_X: None\n')
        file_obj.write('DITHER_Y: None\n')
        file_obj.write('DIT: None\n')
        file_obj.write('LATITUDE: None\n')
        file_obj.write('LONGITUDE: None\n')
        file_obj.write('PUPIL: None\n')
        file_obj.write('DATE: None\n')
        file_obj.write('RA: None\n')
        file_obj.write('DEC: None\n\n')
        file_obj.write('[settings]\n\n')
        file_obj.write('PIXSCALE: None\n')
        file_obj.write('MEMORY: None\n')
        file_obj.write('CPU: None\n')
        file_obj.close()

        pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

        attribute = pipeline.get_attribute("config", "MEMORY", static=True)
        assert attribute == 0

        attribute = pipeline.get_attribute("config", "CPU", static=True)
        assert attribute == 0

        attribute = pipeline.get_attribute("config", "PIXSCALE", static=True)
        assert np.allclose(attribute, 0.0, rtol=limit, atol=0.)

        attribute = pipeline.get_attribute("config", "INSTRUMENT", static=True)
        assert attribute == "None"

        create_config(self.test_dir+"PynPoint_config.ini")

    def test_create_pipeline_path_missing(self):
        dir_non_exists = self.test_dir + "none/"
        dir_exists = self.test_dir

        with pytest.raises(AssertionError) as error:
            Pypeline(dir_non_exists, dir_exists, dir_exists)

        assert str(error.value) == "Input directory for _m_working_place does not exist " \
                                   "- input requested: "+self.test_dir+"none/."

        with pytest.raises(AssertionError) as error:
            Pypeline(dir_exists, dir_non_exists, dir_exists)

        assert str(error.value) == "Input directory for _m_input_place does not exist " \
                                   "- input requested: "+self.test_dir+"none/."

        with pytest.raises(AssertionError) as error:
            Pypeline(dir_exists, dir_exists, dir_non_exists)

        assert str(error.value) == "Input directory for _m_output_place does not exist " \
                                   "- input requested: "+self.test_dir+"none/."

        with pytest.raises(AssertionError) as error:
            Pypeline()

        assert str(error.value) == "Input directory for _m_working_place does not exist " \
                                   "- input requested: None."

    def test_create_pipeline_existing_database(self):
        np.random.seed(1)
        images = np.random.normal(loc=0, scale=2e-4, size=(10, 100, 100))

        h5f = h5py.File(self.test_dir+"PynPoint_database.hdf5", "w")
        dset = h5f.create_dataset("images", data=images)
        dset.attrs['PIXSCALE'] = 0.01
        h5f.close()

        pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)
        data = pipeline.get_data("images")

        assert np.allclose(data[0, 0, 0], 0.00032486907273264834, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 1.0506056979365338e-06, rtol=limit, atol=0.)
        assert pipeline.get_attribute("images", "PIXSCALE") == 0.01

        os.remove(self.test_dir+"PynPoint_database.hdf5")

    def test_create_pipeline_new_database(self):
        pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

        pipeline.m_data_storage.open_connection()
        pipeline.m_data_storage.close_connection()

        del pipeline

        os.remove(self.test_dir+"PynPoint_database.hdf5")

    def test_add_module(self):
        pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

        read = FitsReadingModule(name_in="read1", input_dir=None, image_tag="im_arr1")
        assert pipeline.add_module(read) is None

        read = FitsReadingModule(name_in="read2", input_dir=self.test_dir, image_tag="im_arr2")
        assert pipeline.add_module(read) is None

        with pytest.warns(UserWarning) as warning:
            pipeline.add_module(read)

        assert len(warning) == 1
        assert warning[0].message.args[0] == "Processing module names need to be unique. " \
                                             "Overwriting module 'read2'."

        process = BadPixelSigmaFilterModule(name_in="badpixel", image_in_tag="im_arr1")
        assert pipeline.add_module(process) is None

        write = FitsWritingModule(name_in="write1", file_name="result.fits", data_tag="im_arr1")
        assert pipeline.add_module(write) is None

        write = FitsWritingModule(name_in="write2", file_name="result.fits", data_tag="im_arr1",
                                  output_dir=self.test_dir)
        assert pipeline.add_module(write) is None

        assert pipeline.run() is None

        assert pipeline.get_module_names() == ['read1', 'read2', 'badpixel', 'write1', 'write2']

        os.remove(self.test_dir+"result.fits")
        os.remove(self.test_dir+"PynPoint_database.hdf5")

    def test_run_module(self):
        pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

        read = FitsReadingModule(name_in="read", image_tag="im_arr")
        assert pipeline.add_module(read) is None
        assert pipeline.run_module("read") is None

        os.remove(self.test_dir+"PynPoint_database.hdf5")

    def test_add_wrong_module(self):
        pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

        with pytest.raises(AssertionError) as error:
            pipeline.add_module(None)

        assert str(error.value) == "The added module is not a valid Pypeline module."

        os.remove(self.test_dir+"PynPoint_database.hdf5")

    def test_run_module_wrong_tag(self):
        pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

        read = FitsReadingModule(name_in="read")
        pipeline.add_module(read)

        write = FitsWritingModule(name_in="write", file_name="result.fits", data_tag="im_list")
        pipeline.add_module(write)

        process = BadPixelSigmaFilterModule(name_in="badpixel", image_in_tag="im_list")
        pipeline.add_module(process)

        with pytest.raises(AttributeError) as error:
            pipeline.run_module("badpixel")

        assert str(error.value) == "Pipeline module 'badpixel' is looking for data under a tag " \
                                   "which does not exist in the database."

        with pytest.raises(AttributeError) as error:
            pipeline.run_module("write")

        assert str(error.value) == "Pipeline module 'write' is looking for data under a tag " \
                                   "which does not exist in the database."

        with pytest.raises(AttributeError) as error:
            pipeline.run()

        assert str(error.value) == "Pipeline module 'write' is looking for data under a tag " \
                                   "which is not created by a previous module or does not exist " \
                                   "in the database."

        assert pipeline.validate_pipeline_module("test") is None
        assert pipeline._validate("module", "tag") == (False, None)

        os.remove(self.test_dir+"PynPoint_database.hdf5")

    def test_run_module_non_existing(self):
        pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

        with pytest.warns(UserWarning) as warning:
            pipeline.run_module("test")

        assert len(warning) == 1
        assert warning[0].message.args[0] == "Module 'test' not found."

        os.remove(self.test_dir+"PynPoint_database.hdf5")

    def test_remove_module(self):
        pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

        read = FitsReadingModule(name_in="read")
        pipeline.add_module(read)

        process = BadPixelSigmaFilterModule(name_in="badpixel")
        pipeline.add_module(process)

        assert pipeline.get_module_names() == ["read", "badpixel"]
        assert pipeline.remove_module("read")

        assert pipeline.get_module_names() == ["badpixel"]
        assert pipeline.remove_module("badpixel")

        with pytest.warns(UserWarning) as warning:
            pipeline.remove_module("test")

        assert len(warning) == 1
        assert warning[0].message.args[0] == "Module name 'test' not found in the Pypeline " \
                                             "dictionary."

        os.remove(self.test_dir+"PynPoint_database.hdf5")

    def test_get_shape(self):
        pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

        read = FitsReadingModule(name_in="read", image_tag="images")
        pipeline.add_module(read)
        pipeline.run_module("read")

        assert pipeline.get_shape("images") == (10, 100, 100)

    def test_get_tags(self):
        pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

        assert pipeline.get_tags() == "images"

    def test_set_and_get_attribute(self):
        pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

        pipeline.set_attribute("images", "PIXSCALE", 0.1, static=True)
        pipeline.set_attribute("images", "PARANG", np.arange(1., 11., 1.), static=False)

        attribute = pipeline.get_attribute("images", "PIXSCALE", static=True)
        assert np.allclose(attribute, 0.1, rtol=limit, atol=0.)

        attribute = pipeline.get_attribute("images", "PARANG", static=False)
        assert np.allclose(attribute, np.arange(1., 11., 1.), rtol=limit, atol=0.)

        pipeline.set_attribute("images", "PARANG", np.arange(10., 21., 1.), static=False)

        attribute = pipeline.get_attribute("images", "PARANG", static=False)
        assert np.allclose(attribute, np.arange(10., 21., 1.), rtol=limit, atol=0.)
