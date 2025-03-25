import os

import h5py
import pytest
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.hdf5reading import Hdf5ReadingModule
from pynpoint.util.tests import create_config, create_random, remove_test_data


class TestHdf5Reading:

    def setup_class(self) -> None:

        self.limit = 1e-10
        self.test_dir = os.path.dirname(__file__) + "/"

        create_random(self.test_dir + "data")
        create_config(self.test_dir + "PynPoint_config.ini")

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self) -> None:

        remove_test_data(self.test_dir, folders=["data"])

    def test_hdf5_reading(self) -> None:

        data = np.random.normal(loc=0, scale=2e-4, size=(4, 10, 10))

        with h5py.File(self.test_dir + "data/PynPoint_database.hdf5", "a") as hdf_file:
            hdf_file.create_dataset("extra", data=data)
            hdf_file.create_dataset("header_extra/PARANG", data=[1.0, 2.0, 3.0, 4.0])

        module = Hdf5ReadingModule(
            name_in="read1",
            input_filename="PynPoint_database.hdf5",
            input_dir=self.test_dir + "data",
            tag_dictionary={"images": "images"},
        )

        self.pipeline.add_module(module)
        self.pipeline.run_module("read1")

        data = self.pipeline.get_data("images")
        assert np.sum(data) == pytest.approx(
            0.007153603490533874, rel=self.limit, abs=0.0
        )
        assert data.shape == (5, 11, 11)

    def test_dictionary_none(self) -> None:

        module = Hdf5ReadingModule(
            name_in="read2",
            input_filename="PynPoint_database.hdf5",
            input_dir=self.test_dir + "data",
            tag_dictionary=None,
        )

        self.pipeline.add_module(module)
        self.pipeline.run_module("read2")

        data = self.pipeline.get_data("images")
        assert np.sum(data) == pytest.approx(
            0.007153603490533874, rel=self.limit, abs=0.0
        )
        assert data.shape == (5, 11, 11)

    def test_wrong_tag(self) -> None:

        module = Hdf5ReadingModule(
            name_in="read3",
            input_filename="PynPoint_database.hdf5",
            input_dir=self.test_dir + "data",
            tag_dictionary={"test": "test"},
        )

        self.pipeline.add_module(module)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module("read3")

        assert len(warning) == 1
        assert (
            warning[0].message.args[0]
            == "The dataset with tag name 'test' is not found in "
            "the HDF5 file."
        )

        with h5py.File(self.test_dir + "data/PynPoint_database.hdf5", "r") as hdf_file:
            assert set(hdf_file.keys()) == set(
                ["extra", "header_extra", "header_images", "images"]
            )

    def test_no_input_filename(self) -> None:

        module = Hdf5ReadingModule(
            name_in="read4",
            input_filename=None,
            input_dir=self.test_dir + "data",
            tag_dictionary=None,
        )

        self.pipeline.add_module(module)
        self.pipeline.run_module("read4")

        data = self.pipeline.get_data("images")
        assert np.sum(data) == pytest.approx(
            0.007153603490533874, rel=self.limit, abs=0.0
        )
        assert data.shape == (5, 11, 11)
