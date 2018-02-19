import os
import numpy as np
import pytest
import warnings

from PynPoint.Core import Pypeline
from PynPoint.IOmodules import FitsReadingModule, FitsWritingModule
from PynPoint.ProcessingModules import BadPixelCleaningSigmaFilterModule

warnings.simplefilter("always")


class TestPypeline(object):

    def setup(self):
        self.test_data_dir = (os.path.dirname(__file__)) + '/test_data/'

    def test_create_instance_using_existing_database(self):
        dir_in = self.test_data_dir + "init/"

        pipeline = Pypeline(dir_in,
                            dir_in,
                            dir_in)

        data = pipeline.get_data("im_arr")

        assert data[0, 0, 0] == 27.113279943585397
        assert np.mean(data) == 467.20439057377075

        attr = pipeline.get_attribute("im_arr",
                                      "num_files")

        assert attr == 47

    def test_create_instance_missing_directory(self):
        dir_non_exists = self.test_data_dir + "none/"
        dir_exists = self.test_data_dir + "init/"

        with pytest.raises(AssertionError):
            pipeline = Pypeline(dir_non_exists,
                                dir_exists,
                                dir_exists)

        with pytest.raises(AssertionError):
            pipeline = Pypeline(dir_exists,
                                dir_non_exists,
                                dir_non_exists)

        with pytest.raises(AssertionError):
            # Everything is None
            pipeline = Pypeline()

    def test_create_instance_new_data_base(self):
        dir_in = self.test_data_dir + "new/"

        pipeline = Pypeline(dir_in,
                            dir_in,
                            dir_in)

        pipeline.m_data_storage.open_connection()
        pipeline.m_data_storage.close_connection()

        del pipeline

        os.remove(dir_in + "PynPoint_database.hdf5")

    def test_add_modules(self):

        dir_in = self.test_data_dir + "modules/"

        pipeline = Pypeline(dir_in,
                            dir_in,
                            dir_in)

        # --- Reading Modules ---
        # default location
        reading = FitsReadingModule(name_in="reading")

        # no default location
        reading2 = FitsReadingModule(name_in="reading2",
                                     input_dir=dir_in,
                                     image_tag="im_arr2")

        pipeline.add_module(reading)
        pipeline.add_module(reading2)

        # --- Processing Module ---
        process = BadPixelCleaningSigmaFilterModule(image_in_tag="im_arr")

        pipeline.add_module(process)

        # --- Writing Module ---
        # default location
        write = FitsWritingModule(name_in="writing",
                                  file_name="result.fits",
                                  data_tag="im_arr")

        # no default location
        write2 = FitsWritingModule(name_in="writing2",
                                   file_name="result.fits",
                                   data_tag="im_arr",
                                   output_dir=dir_in)

        pipeline.add_module(write)

        pipeline.add_module(write2)

        with pytest.warns(UserWarning):
            pipeline.add_module(write2)

        pipeline.run()
        pipeline.run_module("reading")

        os.remove(dir_in + "PynPoint_database.hdf5")
        os.remove(dir_in + "result.fits")

    def test_add_non_module_as_module(self):
        dir_in = self.test_data_dir + "modules/"

        pipeline = Pypeline(dir_in,
                            dir_in,
                            dir_in)

        with pytest.raises(AssertionError):
            pipeline.add_module(None)

    def test_run_non_valid_module_list(self):
        dir_in = self.test_data_dir + "modules/"

        pipeline = Pypeline(dir_in,
                            dir_in,
                            dir_in)

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

        assert pipeline.validate_pipeline_module("bla") is None

        os.remove(dir_in + "PynPoint_database.hdf5")

    def test_run_non_existing_module(self):
        dir_in = self.test_data_dir + "modules/"

        pipeline = Pypeline(dir_in,
                            dir_in,
                            dir_in)

        with pytest.warns(UserWarning):
            pipeline.run_module("test")

    def test_remove_module(self):

        dir_in = self.test_data_dir + "modules/"

        pipeline = Pypeline(dir_in,
                            dir_in,
                            dir_in)

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

        assert pipeline.remove_module("none") is False
