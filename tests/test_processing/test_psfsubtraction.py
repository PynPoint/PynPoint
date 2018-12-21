import os
import warnings

import h5py
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.fitsreading import FitsReadingModule
from pynpoint.processing.psfpreparation import AngleInterpolationModule, PSFpreparationModule
from pynpoint.processing.psfsubtraction import PcaPsfSubtractionModule
from pynpoint.util.tests import create_config, create_fake, remove_test_data

warnings.simplefilter("always")

limit = 1e-10

class TestPSFSubtractionPCA(object):

    def setup_class(self):

        self.test_dir = os.path.dirname(__file__) + "/"

        create_fake(path=self.test_dir+"science",
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

        create_fake(path=self.test_dir+"reference",
                    ndit=[10, 10, 10, 10],
                    nframes=[10, 10, 10, 10],
                    exp_no=[1, 2, 3, 4],
                    npix=(100, 100),
                    fwhm=3.,
                    x0=[50, 50, 50, 50],
                    y0=[50, 50, 50, 50],
                    angles=[[0., 25.], [25., 50.], [50., 75.], [75., 100.]],
                    sep=None,
                    contrast=None)

        create_config(self.test_dir+"PynPoint_config.ini")

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self):

        remove_test_data(self.test_dir, folders=["science", "reference"])

    def test_read_data(self):

        read = FitsReadingModule(name_in="read1",
                                 image_tag="science",
                                 input_dir=self.test_dir+"science")

        self.pipeline.add_module(read)
        self.pipeline.run_module("read1")

        data = self.pipeline.get_data("science")
        assert np.allclose(data[0, 50, 50], 0.09798413502193708, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010063896953157961, rtol=limit, atol=0.)
        assert data.shape == (80, 100, 100)

        read = FitsReadingModule(name_in="read2",
                                 image_tag="reference",
                                 input_dir=self.test_dir+"reference")

        self.pipeline.add_module(read)
        self.pipeline.run_module("read2")

        data = self.pipeline.get_data("reference")
        assert np.allclose(data[0, 50, 50], 0.09798413502193708, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010029494781738066, rtol=limit, atol=0.)
        assert data.shape == (40, 100, 100)

    def test_angle_interpolation(self):

        angle = AngleInterpolationModule(name_in="angle",
                                         data_tag="science")

        self.pipeline.add_module(angle)
        self.pipeline.run_module("angle")

        data = self.pipeline.get_data("header_science/PARANG")
        assert np.allclose(data[0], 0., rtol=limit, atol=0.)
        assert np.allclose(data[15], 19.736842105263158, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 50.0, rtol=limit, atol=0.)
        assert data.shape == (80, )

    def test_psf_preparation(self):

        prep = PSFpreparationModule(name_in="prep1",
                                    image_in_tag="science",
                                    image_out_tag="science_prep",
                                    mask_out_tag=None,
                                    norm=False,
                                    resize=None,
                                    cent_size=0.2,
                                    edge_size=1.0)

        self.pipeline.add_module(prep)
        self.pipeline.run_module("prep1")

        data = self.pipeline.get_data("science_prep")
        assert np.allclose(data[0, 0, 0], 0.0, rtol=limit, atol=0.)
        assert np.allclose(data[0, 25, 25], 2.0926464668090656e-05, rtol=limit, atol=0.)
        assert np.allclose(data[0, 99, 99], 0.0, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 4.534001223501053e-07, rtol=limit, atol=0.)
        assert data.shape == (80, 100, 100)

        prep = PSFpreparationModule(name_in="prep2",
                                    image_in_tag="reference",
                                    image_out_tag="reference_prep",
                                    mask_out_tag=None,
                                    norm=False,
                                    resize=None,
                                    cent_size=0.2,
                                    edge_size=1.0)

        self.pipeline.add_module(prep)
        self.pipeline.run_module("prep2")

        data = self.pipeline.get_data("reference_prep")
        assert np.allclose(data[0, 0, 0], 0.0, rtol=limit, atol=0.)
        assert np.allclose(data[0, 25, 25], 2.0926464668090656e-05, rtol=limit, atol=0.)
        assert np.allclose(data[0, 99, 99], 0.0, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 1.227592050148539e-07, rtol=limit, atol=0.)
        assert data.shape == (40, 100, 100)

    def test_psf_subtraction_pca_single(self):

        pca = PcaPsfSubtractionModule(pca_numbers=np.arange(1, 21, 1),
                                      name_in="pca_single",
                                      images_in_tag="science",
                                      reference_in_tag="science",
                                      res_mean_tag="res_mean_single",
                                      res_median_tag="res_median_single",
                                      res_weighted_tag="res_weighted_single",
                                      res_rot_mean_clip_tag="res_clip_single",
                                      res_arr_out_tag="res_arr_single",
                                      basis_out_tag="basis_single",
                                      extra_rot=-15.,
                                      subtract_mean=True)

        self.pipeline.add_module(pca)
        self.pipeline.run_module("pca_single")

        data = self.pipeline.get_data("res_mean_single")
        assert np.allclose(np.mean(data), 2.6959819771522928e-08, rtol=limit, atol=0.)
        assert data.shape == (20, 100, 100)

        data = self.pipeline.get_data("res_median_single")
        assert np.allclose(np.mean(data), -2.4142571236920345e-08, rtol=limit, atol=0.)
        assert data.shape == (20, 100, 100)

        data = self.pipeline.get_data("res_weighted_single")
        assert np.allclose(np.mean(data), -5.293559651636843e-09, rtol=limit, atol=0.)
        assert data.shape == (20, 100, 100)

        data = self.pipeline.get_data("res_clip_single")
        assert np.allclose(np.mean(data), 2.6199554737979536e-08, rtol=limit, atol=0.)
        assert data.shape == (20, 100, 100)

        data = self.pipeline.get_data("res_arr_single5")
        assert np.allclose(np.mean(data), 3.184676024912723e-08, rtol=limit, atol=0.)
        assert data.shape == (80, 100, 100)

        data = self.pipeline.get_data("basis_single")
        assert np.allclose(np.mean(data), -1.593245396350998e-05, rtol=limit, atol=0.)
        assert data.shape == (20, 100, 100)

    def test_psf_subtraction_no_mean(self):

        pca = PcaPsfSubtractionModule(pca_numbers=np.arange(1, 21, 1),
                                      name_in="pca_no_mean",
                                      images_in_tag="science",
                                      reference_in_tag="science",
                                      res_mean_tag="res_mean_no_mean",
                                      res_median_tag=None,
                                      res_weighted_tag=None,
                                      res_rot_mean_clip_tag=None,
                                      res_arr_out_tag=None,
                                      basis_out_tag="basis_no_mean",
                                      extra_rot=0.,
                                      subtract_mean=False)

        self.pipeline.add_module(pca)
        self.pipeline.run_module("pca_no_mean")

        data = self.pipeline.get_data("res_mean_no_mean")
        assert np.allclose(np.mean(data), 2.413203757426239e-08, rtol=limit, atol=0.)
        assert data.shape == (20, 100, 100)

        data = self.pipeline.get_data("basis_no_mean")
        assert np.allclose(np.mean(data), 7.4728664805632875e-06, rtol=limit, atol=0.)
        assert data.shape == (20, 100, 100)

    def test_psf_subtraction_ref(self):

        pca = PcaPsfSubtractionModule(pca_numbers=np.arange(1, 21, 1),
                                      name_in="pca_ref",
                                      images_in_tag="science",
                                      reference_in_tag="reference",
                                      res_mean_tag="res_mean_ref",
                                      res_median_tag=None,
                                      res_weighted_tag=None,
                                      res_rot_mean_clip_tag=None,
                                      res_arr_out_tag=None,
                                      basis_out_tag="basis_ref",
                                      extra_rot=0.,
                                      subtract_mean=True)

        self.pipeline.add_module(pca)
        self.pipeline.run_module("pca_ref")

        data = self.pipeline.get_data("res_mean_ref")
        assert np.allclose(np.mean(data), 1.1662201512335965e-08, rtol=limit, atol=0.)
        assert data.shape == (20, 100, 100)

        data = self.pipeline.get_data("basis_ref")
        assert np.allclose(np.mean(data), -1.6780507257603104e-05, rtol=limit, atol=0.)
        assert data.shape == (20, 100, 100)

    def test_psf_subtraction_ref_no_mean(self):

        pca = PcaPsfSubtractionModule(pca_numbers=np.arange(1, 21, 1),
                                      name_in="pca_ref_no_mean",
                                      images_in_tag="science",
                                      reference_in_tag="reference",
                                      res_mean_tag="res_mean_ref_no_mean",
                                      res_median_tag=None,
                                      res_weighted_tag=None,
                                      res_rot_mean_clip_tag=None,
                                      res_arr_out_tag=None,
                                      basis_out_tag="basis_ref_no_mean",
                                      extra_rot=0.,
                                      subtract_mean=False)

        self.pipeline.add_module(pca)
        self.pipeline.run_module("pca_ref_no_mean")

        data = self.pipeline.get_data("res_mean_ref_no_mean")
        assert np.allclose(np.mean(data), 3.7029738044199534e-07, rtol=limit, atol=0.)
        assert data.shape == (20, 100, 100)

        data = self.pipeline.get_data("basis_ref_no_mean")
        assert np.allclose(np.mean(data), 2.3755682312090375e-05, rtol=limit, atol=0.)
        assert data.shape == (20, 100, 100)

    def test_psf_subtraction_pca_single_mask(self):

        pca = PcaPsfSubtractionModule(pca_numbers=np.arange(1, 21, 1),
                                      name_in="pca_single_mask",
                                      images_in_tag="science_prep",
                                      reference_in_tag="science_prep",
                                      res_mean_tag="res_mean_single_mask",
                                      res_median_tag="res_median_single_mask",
                                      res_weighted_tag="res_weighted_single_mask",
                                      res_rot_mean_clip_tag="res_clip_single_mask",
                                      res_arr_out_tag="res_arr_single_mask",
                                      basis_out_tag="basis_single_mask",
                                      extra_rot=-15.,
                                      subtract_mean=True)

        self.pipeline.add_module(pca)
        self.pipeline.run_module("pca_single_mask")

        data = self.pipeline.get_data("res_mean_single_mask")
        assert np.allclose(np.mean(data), -1.6536519510012155e-09, rtol=limit, atol=0.)
        assert data.shape == (20, 100, 100)

        data = self.pipeline.get_data("res_median_single_mask")
        assert np.allclose(np.mean(data), 5.6094356668078245e-08, rtol=limit, atol=0.)
        assert data.shape == (20, 100, 100)

        data = self.pipeline.get_data("res_weighted_single_mask")
        assert np.allclose(np.mean(data), 4.7079857263662695e-08, rtol=limit, atol=0.)
        assert data.shape == (20, 100, 100)

        data = self.pipeline.get_data("res_clip_single_mask")
        assert np.allclose(np.mean(data), -4.875856901892831e-10, rtol=limit, atol=0.)
        assert data.shape == (20, 100, 100)

        data = self.pipeline.get_data("res_arr_single_mask5")
        assert np.allclose(np.mean(data), -1.700674890172441e-09, rtol=limit, atol=0.)
        assert data.shape == (80, 100, 100)

        data = self.pipeline.get_data("basis_single_mask")
        assert np.allclose(np.mean(data), 5.584100479595007e-06, rtol=limit, atol=0.)
        assert data.shape == (20, 100, 100)

    def test_psf_subtraction_no_mean_mask(self):

        pca = PcaPsfSubtractionModule(pca_numbers=np.arange(1, 21, 1),
                                      name_in="pca_no_mean_mask",
                                      images_in_tag="science_prep",
                                      reference_in_tag="science_prep",
                                      res_mean_tag="res_mean_no_mean_mask",
                                      res_median_tag=None,
                                      res_weighted_tag=None,
                                      res_rot_mean_clip_tag=None,
                                      res_arr_out_tag=None,
                                      basis_out_tag="basis_no_mean_mask",
                                      extra_rot=0.,
                                      subtract_mean=False)

        self.pipeline.add_module(pca)
        self.pipeline.run_module("pca_no_mean_mask")

        data = self.pipeline.get_data("res_mean_no_mean_mask")
        assert np.allclose(np.mean(data), -1.0905008724474168e-09, rtol=limit, atol=0.)
        assert data.shape == (20, 100, 100)

        data = self.pipeline.get_data("basis_no_mean_mask")
        assert np.allclose(np.sum(np.abs(data)), 1025.2018448288406, rtol=limit, atol=0.)
        assert data.shape == (20, 100, 100)

    def test_psf_subtraction_ref_mask(self):

        pca = PcaPsfSubtractionModule(pca_numbers=np.arange(1, 21, 1),
                                      name_in="pca_ref_mask",
                                      images_in_tag="science_prep",
                                      reference_in_tag="reference_prep",
                                      res_mean_tag="res_mean_ref_mask",
                                      res_median_tag=None,
                                      res_weighted_tag=None,
                                      res_rot_mean_clip_tag=None,
                                      res_arr_out_tag=None,
                                      basis_out_tag="basis_ref_mask",
                                      extra_rot=0.,
                                      subtract_mean=True)

        self.pipeline.add_module(pca)
        self.pipeline.run_module("pca_ref_mask")

        data = self.pipeline.get_data("res_mean_ref_mask")
        assert np.allclose(np.mean(data), -9.962692629500833e-10, rtol=limit, atol=0.)
        assert data.shape == (20, 100, 100)

        data = self.pipeline.get_data("basis_ref_mask")
        assert np.allclose(np.mean(data), -2.3165670099810983e-05, rtol=limit, atol=0.)
        assert data.shape == (20, 100, 100)

    def test_psf_subtraction_ref_no_mean_mask(self):

        pca = PcaPsfSubtractionModule(pca_numbers=np.arange(1, 21, 1),
                                      name_in="pca_ref_no_mean_mask",
                                      images_in_tag="science_prep",
                                      reference_in_tag="reference_prep",
                                      res_mean_tag="res_mean_ref_no_mean_mask",
                                      res_median_tag=None,
                                      res_weighted_tag=None,
                                      res_rot_mean_clip_tag=None,
                                      res_arr_out_tag=None,
                                      basis_out_tag="basis_ref_no_mean_mask",
                                      extra_rot=0.,
                                      subtract_mean=False)

        self.pipeline.add_module(pca)
        self.pipeline.run_module("pca_ref_no_mean_mask")

        data = self.pipeline.get_data("res_mean_ref_no_mean_mask")
        assert np.allclose(np.mean(data), 3.848255803450399e-07, rtol=limit, atol=0.)
        assert data.shape == (20, 100, 100)

        data = self.pipeline.get_data("basis_ref_no_mean_mask")
        assert np.allclose(np.sum(np.abs(data)), 1026.3329224435665, rtol=limit, atol=0.)
        assert data.shape == (20, 100, 100)

    def test_psf_subtraction_pca_multi(self):

        database = h5py.File(self.test_dir+'PynPoint_database.hdf5', 'a')
        database['config'].attrs['CPU'] = 4

        pca = PcaPsfSubtractionModule(pca_numbers=np.arange(1, 21, 1),
                                      name_in="pca_multi",
                                      images_in_tag="science",
                                      reference_in_tag="science",
                                      res_mean_tag="res_mean_multi",
                                      res_median_tag="res_median_multi",
                                      res_weighted_tag="res_weighted_multi",
                                      res_rot_mean_clip_tag="res_clip_multi",
                                      res_arr_out_tag=None,
                                      basis_out_tag="basis_multi",
                                      extra_rot=-15.,
                                      subtract_mean=True)

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

        data_single = self.pipeline.get_data("res_weighted_single")
        data_multi = self.pipeline.get_data("res_weighted_multi")
        assert np.allclose(data_single, data_multi, rtol=1e-6, atol=0.)
        assert data_single.shape == data_multi.shape

        data_single = self.pipeline.get_data("basis_single")
        data_multi = self.pipeline.get_data("basis_multi")
        assert np.allclose(data_single, data_multi, rtol=1e-5, atol=0.)
        assert data_single.shape == data_multi.shape

    def test_psf_subtraction_pca_multi_mask(self):

        database = h5py.File(self.test_dir+'PynPoint_database.hdf5', 'a')
        database['config'].attrs['CPU'] = 4

        pca = PcaPsfSubtractionModule(pca_numbers=np.arange(1, 21, 1),
                                      name_in="pca_multi_mask",
                                      images_in_tag="science_prep",
                                      reference_in_tag="science_prep",
                                      res_mean_tag="res_mean_multi_mask",
                                      res_median_tag="res_median_multi_mask",
                                      res_weighted_tag="res_weighted_multi_mask",
                                      res_rot_mean_clip_tag="res_clip_multi_mask",
                                      res_arr_out_tag=None,
                                      basis_out_tag="basis_multi_mask",
                                      extra_rot=-15.,
                                      subtract_mean=True)

        self.pipeline.add_module(pca)
        self.pipeline.run_module("pca_multi_mask")

        data_single = self.pipeline.get_data("res_mean_single_mask")
        data_multi = self.pipeline.get_data("res_mean_multi_mask")
        assert np.allclose(data_single[data_single > 1e-12], data_multi[data_multi > 1e-12], rtol=1e-6, atol=0.)
        assert data_single.shape == data_multi.shape

        data_single = self.pipeline.get_data("res_median_single_mask")
        data_multi = self.pipeline.get_data("res_median_multi_mask")
        assert np.allclose(data_single[data_single > 1e-12], data_multi[data_multi > 1e-12], rtol=1e-6, atol=0.)
        assert data_single.shape == data_multi.shape

        data_single = self.pipeline.get_data("res_weighted_single_mask")
        data_multi = self.pipeline.get_data("res_weighted_multi_mask")
        assert np.allclose(data_single[data_single > 1e-12], data_multi[data_multi > 1e-12], rtol=1e-6, atol=0.)
        assert data_single.shape == data_multi.shape

        data_single = self.pipeline.get_data("basis_single_mask")
        data_multi = self.pipeline.get_data("basis_multi_mask")
        assert np.allclose(data_single, data_multi, rtol=1e-5, atol=0.)
        assert data_single.shape == data_multi.shape
