import os
import warnings

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

    def test_psf_subtraction_pca(self):

        pca = PcaPsfSubtractionModule(pca_numbers=(5, ),
                                      name_in="pca",
                                      images_in_tag="read",
                                      reference_in_tag="read",
                                      res_mean_tag="res_mean",
                                      res_median_tag="res_median",
                                      res_arr_out_tag="res_arr",
                                      res_rot_mean_clip_tag="res_clip",
                                      basis_out_tag="basis",
                                      extra_rot=-15.,
                                      verbose=True)

        self.pipeline.add_module(pca)
        self.pipeline.run_module("pca")

        data = self.pipeline.get_data("res_mean")
        assert np.allclose(data[0, 50, 50], 1.947810457180298e-06, rtol=limit, atol=0.)
        assert np.allclose(data[0, 59, 46], 0.00016087655925993273, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 3.184676024912574e-08, rtol=limit, atol=0.)
        assert data.shape == (1, 100, 100)

        data = self.pipeline.get_data("res_median")
        assert np.allclose(data[0, 50, 50], -2.223389676715259e-06, rtol=limit, atol=0.)
        assert np.allclose(data[0, 59, 46], 0.00015493570876347953, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 1.250907785757355e-07, rtol=limit, atol=0.)
        assert data.shape == (1, 100, 100)

        data = self.pipeline.get_data("res_clip")
        assert np.allclose(data[0, 50, 50], 2.2828813434810948e-06, rtol=limit, atol=0.)
        assert np.allclose(data[0, 59, 46], 1.0816254290076103e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 2.052077475694807e-06, rtol=limit, atol=0.)
        assert data.shape == (1, 100, 100)

        data = self.pipeline.get_data("res_arr5")
        assert np.allclose(data[0, 50, 50], -0.00010775091764735749, rtol=limit, atol=0.)
        assert np.allclose(data[0, 59, 46], 0.0001732810184783699, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 3.184676024912564e-08, rtol=limit, atol=0.)
        assert data.shape == (80, 100, 100)

        data = self.pipeline.get_data("basis")
        assert np.allclose(data[0, 50, 50], -0.005866797940467074, rtol=limit, atol=0.)
        assert np.allclose(data[0, 59, 46], 0.0010154680995154122, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), -4.708475279640416e-05, rtol=limit, atol=0.)
        assert data.shape == (5, 100, 100)
