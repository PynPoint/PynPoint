import os
import warnings

import h5py
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.fitsreading import FitsReadingModule
from pynpoint.processing.resizing import AddLinesModule
from pynpoint.processing.timedenoising import CwtWaveletConfiguration, DwtWaveletConfiguration, \
                                              WaveletTimeDenoisingModule, TimeNormalizationModule
from pynpoint.util.tests import create_config, remove_test_data, create_star_data

warnings.simplefilter('always')

limit = 1e-10


class TestTimeDenoising:

    def setup_class(self):

        self.test_dir = os.path.dirname(__file__) + '/'

        create_star_data(path=self.test_dir+'images',
                         npix_x=20,
                         npix_y=20,
                         x0=[10, 10, 10, 10],
                         y0=[10, 10, 10, 10],
                         parang_start=[0., 25., 50., 75.],
                         parang_end=[25., 50., 75., 100.],
                         exp_no=[1, 2, 3, 4])

        create_config(self.test_dir+'PynPoint_config.ini')

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self):

        remove_test_data(self.test_dir, folders=['images'])

    def test_read_data(self):

        module = FitsReadingModule(name_in='read',
                                   image_tag='images',
                                   input_dir=self.test_dir+'images',
                                   overwrite=True,
                                   check=True)

        self.pipeline.add_module(module)
        self.pipeline.run_module('read')

        data = self.pipeline.get_data('images')
        assert np.allclose(data[0, 10, 10], 0.09799496683489618, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.0025020285041348557, rtol=limit, atol=0.)
        assert data.shape == (40, 20, 20)

    def test_wavelet_denoising_cwt_dog(self):

        cwt_config = CwtWaveletConfiguration(wavelet='dog',
                                             wavelet_order=2,
                                             keep_mean=False,
                                             resolution=0.5)

        assert cwt_config.m_wavelet == 'dog'
        assert np.allclose(cwt_config.m_wavelet_order, 2, rtol=limit, atol=0.)
        assert not cwt_config.m_keep_mean
        assert np.allclose(cwt_config.m_resolution, 0.5, rtol=limit, atol=0.)

        module = WaveletTimeDenoisingModule(wavelet_configuration=cwt_config,
                                            name_in='wavelet_cwt_dog',
                                            image_in_tag='images',
                                            image_out_tag='wavelet_cwt_dog',
                                            padding='zero',
                                            median_filter=True,
                                            threshold_function='soft')

        self.pipeline.add_module(module)
        self.pipeline.run_module('wavelet_cwt_dog')

        data = self.pipeline.get_data('wavelet_cwt_dog')
        assert np.allclose(data[0, 10, 10], 0.09805577173716859, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.002502083112599873, rtol=limit, atol=0.)
        assert data.shape == (40, 20, 20)

        with h5py.File(self.test_dir+'PynPoint_database.hdf5', 'a') as hdf_file:
            hdf_file['config'].attrs['CPU'] = 4

        self.pipeline.run_module('wavelet_cwt_dog')

        data_multi = self.pipeline.get_data('wavelet_cwt_dog')
        assert np.allclose(data, data_multi, rtol=limit, atol=0.)
        assert data.shape == data_multi.shape

    def test_wavelet_denoising_cwt_morlet(self):

        with h5py.File(self.test_dir+'PynPoint_database.hdf5', 'a') as hdf_file:
            hdf_file['config'].attrs['CPU'] = 1

        cwt_config = CwtWaveletConfiguration(wavelet='morlet',
                                             wavelet_order=5,
                                             keep_mean=False,
                                             resolution=0.5)

        assert cwt_config.m_wavelet == 'morlet'
        assert np.allclose(cwt_config.m_wavelet_order, 5, rtol=limit, atol=0.)
        assert not cwt_config.m_keep_mean
        assert np.allclose(cwt_config.m_resolution, 0.5, rtol=limit, atol=0.)

        module = WaveletTimeDenoisingModule(wavelet_configuration=cwt_config,
                                            name_in='wavelet_cwt_morlet',
                                            image_in_tag='images',
                                            image_out_tag='wavelet_cwt_morlet',
                                            padding='mirror',
                                            median_filter=False,
                                            threshold_function='hard')

        self.pipeline.add_module(module)
        self.pipeline.run_module('wavelet_cwt_morlet')

        data = self.pipeline.get_data('wavelet_cwt_morlet')
        assert np.allclose(data[0, 10, 10], 0.09805577173716859, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.0025019409784314286, rtol=limit, atol=0.)
        assert data.shape == (40, 20, 20)

        data = self.pipeline.get_attribute('wavelet_cwt_morlet', 'NFRAMES', static=False)
        assert np.allclose(data, [10, 10, 10, 10], rtol=limit, atol=0.)

    def test_wavelet_denoising_dwt(self):

        dwt_config = DwtWaveletConfiguration(wavelet='db8')

        assert dwt_config.m_wavelet == 'db8'

        module = WaveletTimeDenoisingModule(wavelet_configuration=dwt_config,
                                            name_in='wavelet_dwt',
                                            image_in_tag='images',
                                            image_out_tag='wavelet_dwt',
                                            padding='zero',
                                            median_filter=True,
                                            threshold_function='soft')

        self.pipeline.add_module(module)
        self.pipeline.run_module('wavelet_dwt')

        data = self.pipeline.get_data('wavelet_dwt')
        assert np.allclose(data[0, 10, 10], 0.09650639476873678, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.0024998798596330475, rtol=limit, atol=0.)
        assert data.shape == (40, 20, 20)

    def test_time_normalization(self):

        module = TimeNormalizationModule(name_in='timenorm',
                                         image_in_tag='images',
                                         image_out_tag='timenorm')

        self.pipeline.add_module(module)
        self.pipeline.run_module('timenorm')

        data = self.pipeline.get_data('timenorm')
        assert np.allclose(data[0, 10, 10], 0.09793500165714215, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.0024483409033199985, rtol=limit, atol=0.)
        assert data.shape == (40, 20, 20)

    def test_wavelet_denoising_odd_size(self):

        module = AddLinesModule(name_in='add',
                                image_in_tag='images',
                                image_out_tag='images_odd',
                                lines=(1, 0, 1, 0))

        self.pipeline.add_module(module)
        self.pipeline.run_module('add')

        data = self.pipeline.get_data('images_odd')
        assert np.allclose(data[0, 10, 10], 0.05294085050174391, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.002269413609192613, rtol=limit, atol=0.)
        assert data.shape == (40, 21, 21)

        cwt_config = CwtWaveletConfiguration(wavelet='dog',
                                             wavelet_order=2,
                                             keep_mean=False,
                                             resolution=0.5)

        assert cwt_config.m_wavelet == 'dog'
        assert np.allclose(cwt_config.m_wavelet_order, 2, rtol=limit, atol=0.)
        assert not cwt_config.m_keep_mean
        assert np.allclose(cwt_config.m_resolution, 0.5, rtol=limit, atol=0.)

        module = WaveletTimeDenoisingModule(wavelet_configuration=cwt_config,
                                            name_in='wavelet_odd_1',
                                            image_in_tag='images_odd',
                                            image_out_tag='wavelet_odd_1',
                                            padding='zero',
                                            median_filter=True,
                                            threshold_function='soft')

        self.pipeline.add_module(module)
        self.pipeline.run_module('wavelet_odd_1')

        data = self.pipeline.get_data('wavelet_odd_1')
        assert np.allclose(data[0, 10, 10], 0.0529782051386938, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.0022694631406801565, rtol=limit, atol=0.)
        assert data.shape == (40, 21, 21)

        module = WaveletTimeDenoisingModule(wavelet_configuration=cwt_config,
                                            name_in='wavelet_odd_2',
                                            image_in_tag='images_odd',
                                            image_out_tag='wavelet_odd_2',
                                            padding='mirror',
                                            median_filter=True,
                                            threshold_function='soft')

        self.pipeline.add_module(module)
        self.pipeline.run_module('wavelet_odd_2')

        data = self.pipeline.get_data('wavelet_odd_2')
        assert np.allclose(data[0, 10, 10], 0.05297146283932275, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.0022694809842930034, rtol=limit, atol=0.)
        assert data.shape == (40, 21, 21)

        data = self.pipeline.get_attribute('images', 'NFRAMES', static=False)
        assert np.allclose(data, [10, 10, 10, 10], rtol=limit, atol=0.)

        data = self.pipeline.get_attribute('wavelet_odd_1', 'NFRAMES', static=False)
        assert np.allclose(data, [10, 10, 10, 10], rtol=limit, atol=0.)

        data = self.pipeline.get_attribute('wavelet_odd_2', 'NFRAMES', static=False)
        assert np.allclose(data, [10, 10, 10, 10], rtol=limit, atol=0.)
