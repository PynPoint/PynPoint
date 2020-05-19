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

        processing_types = ['ADI', 'SDI+ADI', 'Tsaa', 'Tsap']

        expected = [[-0.17615249390982402, -0.7938155399702641, 19.552033067005553, 3.850324378983753e-08],
                    [-0.000643226513845853, -0.026507248922765857, 1.0739620435800206, -3.772431121370883e-08],
                    [0.13478375100474496, 0.5715672449895544, -1.0533917918029834, -4.612469664581486e-08],
                    [0.03054619904150832, 0.051145815221822294, 0.027898160417622917, 0.029999347063636557]]

        shape_expc = [(2, 3, 21, 21), (2, 2, 3, 21, 21), (1, 21, 21), (1, 1, 21, 21)]

        pca_numbers = [(range(1, 3), range(1, 3)), (range(1, 3), range(1, 3)), [1], ([1], [1])]

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
            assert np.sum(data) == pytest.approx(expected[i][0], rel=self.limit, abs=0.)
            assert data.shape == shape_expc[i]

            data = self.pipeline.get_data('res_median_single_sdi_'+p_type)
            assert np.sum(data) == pytest.approx(expected[i][1], rel=self.limit, abs=0.)
            assert data.shape == shape_expc[i]

            data = self.pipeline.get_data('res_weighted_single_sdi_'+p_type)
            assert np.sum(data) == pytest.approx(expected[i][2], rel=self.limit, abs=0.)
            assert data.shape == shape_expc[i]

            data = self.pipeline.get_data('res_clip_single_sdi_'+p_type)
#            assert np.sum(data) == pytest.approx(expected[i][3], rel=self.limit, abs=0.)
            assert data.shape == shape_expc[i]

            # data = self.pipeline.get_data('basis_single_sdi_'+p_type)
            # assert np.sum(data) == pytest.approx(-1.3886119555248766, rel=self.limit, abs=0.)
            # assert data.shape == (5, 30, 30)

            if p_type == 'Tsaa':
                data = self.pipeline.get_data(f'res_arr_single_sdi_{p_type}1')
#                assert np.sum(data) == pytest.approx(-1.3886119555248766, rel=self.limit, abs=0.)
                assert data.shape == (10, 21, 21)

    def test_psf_subtraction_sdi_multi(self) -> None:

        with h5py.File(self.test_dir+'PynPoint_database.hdf5', 'a') as hdf_file:
            hdf_file['config'].attrs['CPU'] = 4

        processing_types = ['SDI', 'Tasp', 'Tnan']

        expected = [[-0.005048156279988383, -0.04357166929029911, 0.06870265135913348, -0.009994248054555663],
                    [0.20578060679082302, 0.03699430525568057, 0.14681169595467616, 0.20794580732735243],
                    [50.089890230823386, 49.699563934004516, 50.41008576279893, 50.07842404732254]]

        shape_expc = [(2, 3, 21, 21), (2, 2, 21, 21), (2, 21, 21)]

        for i, p_type in enumerate(processing_types):

            module = PcaPsfSubtractionModule(pca_numbers=(range(1, 3), range(1, 3)),
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
            assert np.sum(data) == pytest.approx(expected[i][0], rel=self.limit, abs=0.)
            assert data.shape == shape_expc[i]

            data = self.pipeline.get_data('res_median_multi_sdi_'+p_type)
            assert np.sum(data) == pytest.approx(expected[i][1], rel=self.limit, abs=0.)
            assert data.shape == shape_expc[i]

            data = self.pipeline.get_data('res_weighted_multi_sdi_'+p_type)
            assert np.sum(data) == pytest.approx(expected[i][2], rel=self.limit, abs=0.)
            assert data.shape == shape_expc[i]

            data = self.pipeline.get_data('res_clip_multi_sdi_'+p_type)
#            assert np.sum(data) == pytest.approx(expected[i][3], rel=self.limit, abs=0.)
            assert data.shape == shape_expc[i]
