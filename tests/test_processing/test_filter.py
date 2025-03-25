import os

import pytest
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.fitsreading import FitsReadingModule
from pynpoint.processing.filter import GaussianFilterModule
from pynpoint.util.tests import create_config, remove_test_data, create_star_data


class TestFilter:

    def setup_class(self) -> None:

        self.limit = 1e-10
        self.test_dir = os.path.dirname(__file__) + "/"

        create_star_data(self.test_dir + "data")
        create_config(self.test_dir + "PynPoint_config.ini")

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self) -> None:

        remove_test_data(self.test_dir, folders=["data"])

    def test_read_data(self) -> None:

        module = FitsReadingModule(
            name_in="read",
            image_tag="data",
            input_dir=self.test_dir + "data",
            overwrite=True,
            check=True,
        )

        self.pipeline.add_module(module)

        self.pipeline.run_module("read")

        data = self.pipeline.get_data("data")
        assert np.sum(data) == pytest.approx(
            105.54278879805277, rel=self.limit, abs=0.0
        )
        assert data.shape == (10, 11, 11)

    def test_gaussian_filter(self) -> None:

        module = GaussianFilterModule(
            name_in="filter", image_in_tag="data", image_out_tag="filtered", fwhm=0.1
        )

        self.pipeline.add_module(module)
        self.pipeline.run_module("filter")

        data = self.pipeline.get_data("filtered")
        assert np.sum(data) == pytest.approx(
            105.54278879805275, rel=self.limit, abs=0.0
        )
        assert data.shape == (10, 11, 11)
