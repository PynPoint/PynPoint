import os

import h5py
import pytest
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.fitsreading import FitsReadingModule
from pynpoint.processing.resizing import AddLinesModule
from pynpoint.processing.timedenoising import (
    CwtWaveletConfiguration,
    DwtWaveletConfiguration,
    WaveletTimeDenoisingModule,
    TimeNormalizationModule,
)
from pynpoint.util.tests import create_config, remove_test_data, create_star_data


class TestTimeDenoising:

    def setup_class(self) -> None:

        self.limit = 1e-10
        self.test_dir = os.path.dirname(__file__) + "/"

        create_star_data(self.test_dir + "images")
        create_config(self.test_dir + "PynPoint_config.ini")

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self) -> None:

        remove_test_data(self.test_dir, folders=["images"])

    def test_read_data(self) -> None:

        module = FitsReadingModule(
            name_in="read",
            image_tag="images",
            input_dir=self.test_dir + "images",
            overwrite=True,
            check=True,
        )

        self.pipeline.add_module(module)
        self.pipeline.run_module("read")

        data = self.pipeline.get_data("images")
        assert np.sum(data) == pytest.approx(
            105.54278879805277, rel=self.limit, abs=0.0
        )
        assert data.shape == (10, 11, 11)

    def test_wavelet_denoising_cwt_dog(self) -> None:

        cwt_config = CwtWaveletConfiguration(
            wavelet="dog", wavelet_order=2, keep_mean=False, resolution=0.5
        )

        assert cwt_config.m_wavelet == "dog"
        assert cwt_config.m_wavelet_order == 2
        assert not cwt_config.m_keep_mean
        assert cwt_config.m_resolution == 0.5

        module = WaveletTimeDenoisingModule(
            wavelet_configuration=cwt_config,
            name_in="wavelet_cwt_dog",
            image_in_tag="images",
            image_out_tag="wavelet_cwt_dog",
            padding="zero",
            median_filter=True,
            threshold_function="soft",
        )

        self.pipeline.add_module(module)
        self.pipeline.run_module("wavelet_cwt_dog")

        data = self.pipeline.get_data("wavelet_cwt_dog")
        assert np.sum(data) == pytest.approx(105.1035789572968, rel=self.limit, abs=0.0)
        assert data.shape == (10, 11, 11)

        with h5py.File(self.test_dir + "PynPoint_database.hdf5", "a") as hdf_file:
            hdf_file["config"].attrs["CPU"] = 4

        self.pipeline.run_module("wavelet_cwt_dog")

        data_multi = self.pipeline.get_data("wavelet_cwt_dog")
        assert data == pytest.approx(data_multi, rel=self.limit, abs=0.0)
        assert data.shape == data_multi.shape

    def test_wavelet_denoising_cwt_morlet(self) -> None:

        with h5py.File(self.test_dir + "PynPoint_database.hdf5", "a") as hdf_file:
            hdf_file["config"].attrs["CPU"] = 1

        cwt_config = CwtWaveletConfiguration(
            wavelet="morlet", wavelet_order=5, keep_mean=False, resolution=0.5
        )

        assert cwt_config.m_wavelet == "morlet"
        assert cwt_config.m_wavelet_order == 5
        assert not cwt_config.m_keep_mean
        assert cwt_config.m_resolution == 0.5

        module = WaveletTimeDenoisingModule(
            wavelet_configuration=cwt_config,
            name_in="wavelet_cwt_morlet",
            image_in_tag="images",
            image_out_tag="wavelet_cwt_morlet",
            padding="mirror",
            median_filter=False,
            threshold_function="hard",
        )

        self.pipeline.add_module(module)
        self.pipeline.run_module("wavelet_cwt_morlet")

        data = self.pipeline.get_data("wavelet_cwt_morlet")
        assert np.sum(data) == pytest.approx(
            104.86262840716438, rel=self.limit, abs=0.0
        )
        assert data.shape == (10, 11, 11)

        data = self.pipeline.get_attribute(
            "wavelet_cwt_morlet", "NFRAMES", static=False
        )
        assert data[0] == data[1] == 5

    def test_wavelet_denoising_dwt(self) -> None:

        dwt_config = DwtWaveletConfiguration(wavelet="db8")

        assert dwt_config.m_wavelet == "db8"

        module = WaveletTimeDenoisingModule(
            wavelet_configuration=dwt_config,
            name_in="wavelet_dwt",
            image_in_tag="images",
            image_out_tag="wavelet_dwt",
            padding="zero",
            median_filter=True,
            threshold_function="soft",
        )

        self.pipeline.add_module(module)
        self.pipeline.run_module("wavelet_dwt")

        data = self.pipeline.get_data("wavelet_dwt")
        assert np.sum(data) == pytest.approx(
            105.54278879805277, rel=self.limit, abs=0.0
        )
        assert data.shape == (10, 11, 11)

    def test_time_normalization(self) -> None:

        module = TimeNormalizationModule(
            name_in="timenorm", image_in_tag="images", image_out_tag="timenorm"
        )

        self.pipeline.add_module(module)
        self.pipeline.run_module("timenorm")

        data = self.pipeline.get_data("timenorm")
        assert np.sum(data) == pytest.approx(56.443663773873, rel=self.limit, abs=0.0)
        assert data.shape == (10, 11, 11)

    def test_wavelet_denoising_even_size(self) -> None:

        module = AddLinesModule(
            name_in="add",
            image_in_tag="images",
            image_out_tag="images_even",
            lines=(1, 0, 1, 0),
        )

        self.pipeline.add_module(module)
        self.pipeline.run_module("add")

        data = self.pipeline.get_data("images_even")
        assert np.sum(data) == pytest.approx(
            105.54278879805275, rel=self.limit, abs=0.0
        )
        assert data.shape == (10, 12, 12)

        cwt_config = CwtWaveletConfiguration(
            wavelet="dog", wavelet_order=2, keep_mean=False, resolution=0.5
        )

        assert cwt_config.m_wavelet == "dog"
        assert cwt_config.m_wavelet_order == 2
        assert not cwt_config.m_keep_mean
        assert cwt_config.m_resolution == 0.5

        module = WaveletTimeDenoisingModule(
            wavelet_configuration=cwt_config,
            name_in="wavelet_even_1",
            image_in_tag="images_even",
            image_out_tag="wavelet_even_1",
            padding="zero",
            median_filter=True,
            threshold_function="soft",
        )

        self.pipeline.add_module(module)
        self.pipeline.run_module("wavelet_even_1")

        data = self.pipeline.get_data("wavelet_even_1")
        assert np.sum(data) == pytest.approx(105.1035789572968, rel=self.limit, abs=0.0)
        assert data.shape == (10, 12, 12)

        module = WaveletTimeDenoisingModule(
            wavelet_configuration=cwt_config,
            name_in="wavelet_even_2",
            image_in_tag="images_even",
            image_out_tag="wavelet_even_2",
            padding="mirror",
            median_filter=True,
            threshold_function="soft",
        )

        self.pipeline.add_module(module)
        self.pipeline.run_module("wavelet_even_2")

        data = self.pipeline.get_data("wavelet_even_2")
        assert np.sum(data) == pytest.approx(
            105.06809820408587, rel=self.limit, abs=0.0
        )
        assert data.shape == (10, 12, 12)

        data = self.pipeline.get_attribute("images", "NFRAMES", static=False)
        assert data == pytest.approx([5, 5], rel=self.limit, abs=0.0)

        data = self.pipeline.get_attribute("wavelet_even_1", "NFRAMES", static=False)
        assert data == pytest.approx([5, 5], rel=self.limit, abs=0.0)

        data = self.pipeline.get_attribute("wavelet_even_2", "NFRAMES", static=False)
        assert data == pytest.approx([5, 5], rel=self.limit, abs=0.0)
