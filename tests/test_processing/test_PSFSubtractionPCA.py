import os
import warnings

import h5py
import numpy as np

from PynPoint.Core.Pypeline import Pypeline
from PynPoint.IOmodules.FitsReading import FitsReadingModule
from PynPoint.ProcessingModules.PSFpreparation import AngleInterpolationModule
from PynPoint.ProcessingModules.PSFSubtractionPCA import PcaPsfSubtractionModule
from PynPoint.Util.TestTools import create_config, create_fake, remove_test_data

warnings.simplefilter("always")

limit = 1e-10

class TestPSFSubtractionPCA(object):

    def setup_class(self):

        self.test_dir = os.path.dirname(__file__) + "/"

        create_fake(path=self.test_dir+"pca",
                    ndit=[20, 20, 20, 20],
                    nframes=[20, 20, 20, 20],
                    exp_no=[1, 2, 3, 4],
                    npix=(100, 100),
                    fwhm=3.,
                    x0=[50, 50, 50, 50],
                    y0=[50, 50, 50, 50],
                    angles=[[0., 25.], [25., 50.], [50., 75.], [75., 100.]],
                    sep=10.,
                    contrast=3e-3)

        create_config(self.test_dir+"PynPoint_config.ini")

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self):

        remove_test_data(self.test_dir, folders=["pca"])

    def test_read_data(self):

        read = FitsReadingModule(name_in="read",
                                 image_tag="read",
                                 input_dir=self.test_dir+"pca")

        self.pipeline.add_module(read)
        self.pipeline.run_module("read")

        data = self.pipeline.get_data("read")
        assert np.allclose(data[0, 50, 50], 0.09798413502193708, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010063896953157961, rtol=limit, atol=0.)
        assert data.shape == (80, 100, 100)

    def test_angle_interpolation(self):

        angle = AngleInterpolationModule(name_in="angle",
                                         data_tag="read")

        self.pipeline.add_module(angle)
        self.pipeline.run_module("angle")

        data = self.pipeline.get_data("header_read/PARANG")
        assert np.allclose(data[0], 0., rtol=limit, atol=0.)
        assert np.allclose(data[15], 19.736842105263158, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 50.0, rtol=limit, atol=0.)
        assert data.shape == (80, )

    def test_psf_subtraction_pca_single(self):

        pca = PcaPsfSubtractionModule(pca_numbers=np.arange(1, 21, 1),
                                      name_in="pca_single",
                                      images_in_tag="read",
                                      reference_in_tag="read",
                                      res_mean_tag="res_mean_single",
                                      res_median_tag="res_median_single",
                                      res_arr_out_tag="res_arr_single",
                                      res_rot_mean_clip_tag="res_clip_single",
                                      basis_out_tag="basis_single",
                                      extra_rot=-15.,
                                      verbose=True)

        self.pipeline.add_module(pca)
        self.pipeline.run_module("pca_single")

        data = self.pipeline.get_data("res_mean_single")
        assert np.allclose(data[4, 50, 50], 1.947810457180298e-06, rtol=limit, atol=0.)
        assert np.allclose(data[4, 59, 46], 0.00016087655925993273, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 2.6959819771522928e-08, rtol=limit, atol=0.)
        assert data.shape == (20, 100, 100)

        data = self.pipeline.get_data("res_median_single")
        assert np.allclose(data[4, 50, 50], -2.223389676715259e-06, rtol=limit, atol=0.)
        assert np.allclose(data[4, 59, 46], 0.00015493570876347953, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), -2.4142571236920345e-08, rtol=limit, atol=0.)
        assert data.shape == (20, 100, 100)

        data = self.pipeline.get_data("res_clip_single")
        assert np.allclose(data[4, 50, 50], 2.2828813434810948e-06, rtol=limit, atol=0.)
        assert np.allclose(data[4, 59, 46], 1.0816254290076103e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 1.8805674720019969e-06, rtol=limit, atol=0.)
        assert data.shape == (20, 100, 100)

        data = self.pipeline.get_data("res_arr_single5")
        assert np.allclose(data[0, 50, 50], -0.00010775091764735749, rtol=limit, atol=0.)
        assert np.allclose(data[0, 59, 46], 0.0001732810184783699, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 3.184676024912723e-08, rtol=limit, atol=0.)
        assert data.shape == (80, 100, 100)

        data = self.pipeline.get_data("basis_single")
        assert np.allclose(data[0, 50, 50], -0.005866797940467074, rtol=limit, atol=0.)
        assert np.allclose(data[0, 59, 46], 0.0010154680995154122, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), -1.593245396350998e-05, rtol=limit, atol=0.)
        assert data.shape == (20, 100, 100)

    def test_psf_subtraction_pca_multi(self):

        database = h5py.File(self.test_dir+'PynPoint_database.hdf5', 'a')
        database['config'].attrs['CPU'] = 4
        database.close()

        pca = PcaPsfSubtractionModule(pca_numbers=np.arange(1, 21, 1),
                                      name_in="pca_multi",
                                      images_in_tag="read",
                                      reference_in_tag="read",
                                      res_mean_tag="res_mean_multi",
                                      res_median_tag="res_median_multi",
                                      res_arr_out_tag=None,
                                      res_rot_mean_clip_tag=None,
                                      basis_out_tag="basis_multi",
                                      extra_rot=-15.,
                                      verbose=True)

        self.pipeline.add_module(pca)
        self.pipeline.run_module("pca_multi")

        data_single = self.pipeline.get_data("res_mean_single")
        data_multi = self.pipeline.get_data("res_mean_multi")
        assert np.allclose(data_single, data_multi, rtol=1e-6, atol=0.)
        assert data_single.shape == data_multi.shape

        data_single = self.pipeline.get_data("res_median_single")
        data_multi = self.pipeline.get_data("res_median_multi")
        assert np.allclose(data_single, data_multi, rtol=1e-6, atol=0.)
        assert data_single.shape == data_multi.shape

        data_single = self.pipeline.get_data("basis_single")
        data_multi = self.pipeline.get_data("basis_multi")
        assert np.allclose(data_single, data_multi, rtol=1e-5, atol=0.)
        assert data_single.shape == data_multi.shape
