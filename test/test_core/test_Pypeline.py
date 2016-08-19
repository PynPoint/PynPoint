import os
import numpy as np
import pytest

from PynPoint.core import Pypeline


class TestPypeline(object):

    def setup(self):
        self.test_data_dir = (os.path.dirname(__file__)) + '/test_data/'

    def test_create_instance_using_existing_database(self):
        dir = self.test_data_dir + "init/"

        pipeline = Pypeline(dir,
                            dir,
                            dir)

        data = pipeline.get_data("im_arr")

        assert data[0, 0, 0] == 27.113279943585397
        assert np.mean(data) == 467.20439057377075

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









