import os
import h5py

import pytest
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.fitsreading import FitsReadingModule
from pynpoint.processing.psfsubtraction import PcaPsfSubtractionModule
from pynpoint.util.tests import create_config, create_ifs_data, remove_test_data


class TestPsfSubtractionSdi:

    def setup_class(self) -> None:

        self.limit = 1e-5
        self.test_dir = os.path.dirname(__file__) + '/'

        create_ifs_data(self.test_dir+'science')
        create_config(self.test_dir+'PynPoint_config.ini')

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self) -> None:

        remove_test_data(self.test_dir, folders=['science'])

    def test_read_data(self) -> None:

        module = FitsReadingModule(name_in='read',
                                   image_tag='science',
                                   input_dir=self.test_dir+'science',
                                   ifs_data=True)

        self.pipeline.add_module(module)
        self.pipeline.run_module('read')

        data = self.pipeline.get_data('science')
        assert np.sum(data) == pytest.approx(749.8396528807368, rel=self.limit, abs=0.)
        assert data.shape == (3, 10, 21, 21)

        self.pipeline.set_attribute('science', 'WAVELENGTH', [1., 1.1, 1.2], static=False)
        self.pipeline.set_attribute('science', 'PARANG', np.linspace(0., 180., 10), static=False)

    def test_psf_subtraction_sdi(self) -> None:

        processing_types = ['ADI', 'SDI+ADI', 'ADI+SDI']

        expected = [[-0.176152493909826, -0.7938155399702668, 19.552033067005578, -0.21617058715490922],
                    [-0.004568154679096975, -0.08621264803633322, 2.2901225325010888, -0.010269745733878437],
                    [0.008630501634061892, -0.05776205365084376, -0.4285370289350482, 0.0058856438951644455]]

        shape_expc = [(2, 3, 21, 21), (2, 2, 3, 21, 21), (1, 1, 3, 21, 21)]

        pca_numbers = [range(1, 3), (range(1, 3), range(1, 3)), ([1], [1])]

        for i, p_type in enumerate(processing_types):

            module = PcaPsfSubtractionModule(pca_numbers=pca_numbers[i],
                                             name_in='pca_single_sdi_'+p_type,
                                             images_in_tag='science',
                                             reference_in_tag='science',
                                             res_mean_tag='res_mean_single_sdi_'+p_type,
                                             res_median_tag='res_median_single_sdi_'+p_type,
                                             res_weighted_tag='res_weighted_single_sdi_'+p_type,
                                             res_rot_mean_clip_tag='res_clip_single_sdi_'+p_type,
                                             res_arr_out_tag='res_arr_single_sdi_'+p_type,
                                             basis_out_tag='basis_single_sdi_'+p_type,
                                             extra_rot=0.,
                                             subtract_mean=True,
                                             processing_type=p_type)

            self.pipeline.add_module(module)
            self.pipeline.run_module('pca_single_sdi_'+p_type)

            data = self.pipeline.get_data('res_mean_single_sdi_'+p_type)
#            assert np.sum(data) == pytest.approx(expected[i][0], rel=self.limit, abs=0.)
            assert data.shape == shape_expc[i]

            data = self.pipeline.get_data('res_median_single_sdi_'+p_type)
#            assert np.sum(data) == pytest.approx(expected[i][1], rel=self.limit, abs=0.)
            assert data.shape == shape_expc[i]

            data = self.pipeline.get_data('res_weighted_single_sdi_'+p_type)
#            assert np.sum(data) == pytest.approx(expected[i][2], rel=self.limit, abs=0.)
            assert data.shape == shape_expc[i]

            data = self.pipeline.get_data('res_clip_single_sdi_'+p_type)
#            assert np.sum(data) == pytest.approx(expected[i][3], rel=self.limit, abs=0.)
            assert data.shape == shape_expc[i]

            # data = self.pipeline.get_data('basis_single_sdi_'+p_type)
            # assert np.sum(data) == pytest.approx(-1.3886119555248766, rel=self.limit, abs=0.)
            # assert data.shape == (5, 30, 30)

    def test_psf_subtraction_sdi_multi(self) -> None:

        with h5py.File(self.test_dir+'PynPoint_database.hdf5', 'a') as hdf_file:
            hdf_file['config'].attrs['CPU'] = 4

        processing_types = ['SDI', 'ADI+SDI']

        pca_numbers = [range(1, 3), (range(1, 3), range(1, 3))]

        expected = [[-0.0044942456603888695, 0.02613693149969979, -0.15045543311096457, -0.008432530081399985],
                    [-0.0094093643053501, -0.08171546066331437, 0.560810054788774, -0.014527353460544753]]

        shape_expc = [(2, 3, 21, 21), (2, 2, 3, 21, 21)]

        for i, p_type in enumerate(processing_types):

            module = PcaPsfSubtractionModule(pca_numbers=pca_numbers[i],
                                             name_in='pca_multi_sdi_'+p_type,
                                             images_in_tag='science',
                                             reference_in_tag='science',
                                             res_mean_tag='res_mean_multi_sdi_'+p_type,
                                             res_median_tag='res_median_multi_sdi_'+p_type,
                                             res_weighted_tag='res_weighted_multi_sdi_'+p_type,
                                             res_rot_mean_clip_tag='res_clip_multi_sdi_'+p_type,
                                             res_arr_out_tag=None,
                                             basis_out_tag=None,
                                             extra_rot=0.,
                                             subtract_mean=True,
                                             processing_type=p_type)

            self.pipeline.add_module(module)
            self.pipeline.run_module('pca_multi_sdi_'+p_type)

            data = self.pipeline.get_data('res_mean_multi_sdi_'+p_type)
#            assert np.sum(data) == pytest.approx(expected[i][0], rel=self.limit, abs=0.)
            assert data.shape == shape_expc[i]

            data = self.pipeline.get_data('res_median_multi_sdi_'+p_type)
#            assert np.sum(data) == pytest.approx(expected[i][1], rel=self.limit, abs=0.)
            assert data.shape == shape_expc[i]

            data = self.pipeline.get_data('res_weighted_multi_sdi_'+p_type)
#            assert np.sum(data) == pytest.approx(expected[i][2], rel=self.limit, abs=0.)
            assert data.shape == shape_expc[i]

            data = self.pipeline.get_data('res_clip_multi_sdi_'+p_type)
#            assert np.sum(data) == pytest.approx(expected[i][3], rel=self.limit, abs=0.)
            assert data.shape == shape_expc[i]