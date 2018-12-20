import os
import warnings

import h5py
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.fitsreading import FitsReadingModule
from pynpoint.processing.timedenoising import CwtWaveletConfiguration, DwtWaveletConfiguration, \
                                              WaveletTimeDenoisingModule, TimeNormalizationModule
from pynpoint.util.tests import create_config, remove_test_data, create_star_data

warnings.simplefilter("always")

limit = 1e-10

class TestTimeDenoising(object):

    def setup_class(self):

        self.test_dir = os.path.dirname(__file__) + "/"

        create_star_data(path=self.test_dir+"images",
                         npix_x=20,
                         npix_y=20,
                         x0=[10, 10, 10, 10],
                         y0=[10, 10, 10, 10],
                         parang_start=[0., 25., 50., 75.],
                         parang_end=[25., 50., 75., 100.],
                         exp_no=[1, 2, 3, 4])

        create_config(self.test_dir+"PynPoint_config.ini")

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self):

        remove_test_data(self.test_dir, folders=["images"])

    def test_read_data(self):

        read = FitsReadingModule(name_in="read",
                                 image_tag="images",
                                 input_dir=self.test_dir+"images",
                                 overwrite=True,
                                 check=True)

        self.pipeline.add_module(read)

        self.pipeline.run_module("read")

        data = self.pipeline.get_data("images")
        assert np.allclose(data[0, 10, 10], 0.09799496683489618, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.0025020285041348557, rtol=limit, atol=0.)
        assert data.shape == (40, 20, 20)

    def test_wavelet_denoising_cwt_dog(self):

        cwt_config = CwtWaveletConfiguration(wavelet="dog",
                                             wavelet_order=2,
                                             keep_mean=False,
                                             resolution=0.5)

        assert cwt_config.m_wavelet == "dog"
        assert np.allclose(cwt_config.m_wavelet_order, 2, rtol=limit, atol=0.)
        assert not cwt_config.m_keep_mean
        assert np.allclose(cwt_config.m_resolution, 0.5, rtol=limit, atol=0.)

        wavelet_cwt = WaveletTimeDenoisingModule(wavelet_configuration=cwt_config,
                                                 name_in="wavelet_cwt_dog",
                                                 image_in_tag="images",
                                                 image_out_tag="wavelet_cwt_dog",
                                                 padding="zero",
                                                 median_filter=True,
                                                 threshold_function="soft")

        self.pipeline.add_module(wavelet_cwt)
        self.pipeline.run_module("wavelet_cwt_dog")

        data = self.pipeline.get_data("wavelet_cwt_dog")
        assert np.allclose(data[0, 10, 10], 0.09805577173716859, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.002502083112599873, rtol=limit, atol=0.)
        assert data.shape == (40, 20, 20)

        database = h5py.File(self.test_dir+'PynPoint_database.hdf5', 'a')
        database['config'].attrs['CPU'] = 4

        self.pipeline.run_module("wavelet_cwt_dog")

        data_multi = self.pipeline.get_data("wavelet_cwt_dog")
        assert np.allclose(data, data_multi, rtol=limit, atol=0.)
        assert data.shape == data_multi.shape

    def test_wavelet_denoising_cwt_morlet(self):

        database = h5py.File(self.test_dir+'PynPoint_database.hdf5', 'a')
        database['config'].attrs['CPU'] = 1

        cwt_config = CwtWaveletConfiguration(wavelet="morlet",
                                             wavelet_order=5,
                                             keep_mean=False,
                                             resolution=0.5)

        assert cwt_config.m_wavelet == "morlet"
        assert np.allclose(cwt_config.m_wavelet_order, 5, rtol=limit, atol=0.)
        assert not cwt_config.m_keep_mean
        assert np.allclose(cwt_config.m_resolution, 0.5, rtol=limit, atol=0.)

        wavelet_cwt = WaveletTimeDenoisingModule(wavelet_configuration=cwt_config,
                                                 name_in="wavelet_cwt_morlet",
                                                 image_in_tag="images",
                                                 image_out_tag="wavelet_cwt_morlet",
                                                 padding="mirror",
                                                 median_filter=False,
                                                 threshold_function="hard")

        self.pipeline.add_module(wavelet_cwt)
        self.pipeline.run_module("wavelet_cwt_morlet")

        data = self.pipeline.get_data("wavelet_cwt_morlet")
        assert np.allclose(data[0, 10, 10], 0.09805577173716859, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.0025019409784314286, rtol=limit, atol=0.)
        assert data.shape == (40, 20, 20)

    def test_wavelet_denoising_dwt(self):

        dwt_config = DwtWaveletConfiguration(wavelet="db8")

        assert dwt_config.m_wavelet == "db8"

        wavelet_dwt = WaveletTimeDenoisingModule(wavelet_configuration=dwt_config,
                                                 name_in="wavelet_dwt",
                                                 image_in_tag="images",
                                                 image_out_tag="wavelet_dwt",
                                                 padding="zero",
                                                 median_filter=True,
                                                 threshold_function="soft")

        self.pipeline.add_module(wavelet_dwt)
        self.pipeline.run_module("wavelet_dwt")

        data = self.pipeline.get_data("wavelet_dwt")
        assert np.allclose(data[0, 10, 10], 0.09650639476873678, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.0024998798596330475, rtol=limit, atol=0.)
        assert data.shape == (40, 20, 20)

    def test_time_normalization(self):

        timenorm = TimeNormalizationModule(name_in="timenorm",
                                           image_in_tag="images",
                                           image_out_tag="timenorm")

        self.pipeline.add_module(timenorm)
        self.pipeline.run_module("timenorm")

        data = self.pipeline.get_data("timenorm")
        assert np.allclose(data[0, 10, 10], 0.09793500165714215, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.0024483409033199985, rtol=limit, atol=0.)
        assert data.shape == (40, 20, 20)
