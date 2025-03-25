import os

import h5py
import pytest
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.processing.darkflat import DarkCalibrationModule, FlatCalibrationModule
from pynpoint.util.tests import create_config, remove_test_data


class TestDarkFlat:

    def setup_class(self) -> None:

        self.limit = 1e-10
        self.test_dir = os.path.dirname(__file__) + "/"

        np.random.seed(1)

        images = np.random.normal(loc=0, scale=2e-4, size=(5, 11, 11))
        dark = np.random.normal(loc=0, scale=2e-4, size=(5, 11, 11))
        flat = np.random.normal(loc=0, scale=2e-4, size=(5, 11, 11))
        crop = np.random.normal(loc=0, scale=2e-4, size=(5, 7, 7))

        with h5py.File(self.test_dir + "PynPoint_database.hdf5", "w") as hdf_file:
            hdf_file.create_dataset("images", data=images)
            hdf_file.create_dataset("dark", data=dark)
            hdf_file.create_dataset("flat", data=flat)
            hdf_file.create_dataset("crop", data=crop)

        create_config(self.test_dir + "PynPoint_config.ini")

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self) -> None:

        remove_test_data(self.test_dir)

    def test_input_data(self) -> None:

        data = self.pipeline.get_data("dark")
        assert np.sum(data) == pytest.approx(
            -2.0262345764957305e-05, rel=self.limit, abs=0.0
        )
        assert data.shape == (5, 11, 11)

        data = self.pipeline.get_data("flat")
        assert np.sum(data) == pytest.approx(
            0.0076413379497053, rel=self.limit, abs=0.0
        )
        assert data.shape == (5, 11, 11)

    def test_dark_calibration(self) -> None:

        module = DarkCalibrationModule(
            name_in="dark",
            image_in_tag="images",
            dark_in_tag="dark",
            image_out_tag="dark_cal",
        )

        self.pipeline.add_module(module)
        self.pipeline.run_module("dark")

        data = self.pipeline.get_data("dark_cal")
        assert np.sum(data) == pytest.approx(
            0.00717386583629883, rel=self.limit, abs=0.0
        )
        assert data.shape == (5, 11, 11)

    def test_flat_calibration(self) -> None:

        module = FlatCalibrationModule(
            name_in="flat",
            image_in_tag="dark_cal",
            flat_in_tag="flat",
            image_out_tag="flat_cal",
        )

        self.pipeline.add_module(module)
        self.pipeline.run_module("flat")

        data = self.pipeline.get_data("flat_cal")
        assert np.sum(data) == pytest.approx(
            0.00717439711853594, rel=self.limit, abs=0.0
        )
        assert data.shape == (5, 11, 11)

    def test_flat_crop(self) -> None:

        module = FlatCalibrationModule(
            name_in="flat_crop",
            image_in_tag="crop",
            flat_in_tag="flat",
            image_out_tag="flat_crop",
        )

        self.pipeline.add_module(module)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module("flat_crop")

        assert len(warning) == 1

        assert (
            warning[0].message.args[0]
            == "The calibration images were cropped around their "
            "center to match the shape of the science images."
        )

        data = self.pipeline.get_data("flat_crop")
        assert np.sum(data) == pytest.approx(
            -0.003242901413605404, rel=self.limit, abs=0.0
        )
        assert data.shape == (5, 7, 7)

    def test_flat_too_small(self) -> None:

        module = FlatCalibrationModule(
            name_in="flat_small",
            image_in_tag="flat",
            flat_in_tag="crop",
            image_out_tag="flat_small",
        )

        self.pipeline.add_module(module)

        with pytest.raises(ValueError) as error:
            self.pipeline.run_module("flat_small")

        assert (
            str(error.value) == "Shape of the calibration images is smaller than the "
            "science images."
        )
