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

        # processing_types = ['ADI', 'SDI', 'SDI+ADI', 'ADI+SDI', 'Tsaa']
        #
        # expected = [[-3.9661462067317356e-08, 9.813240333505746e-08, 6.920690625734549e-08, -3.792473707685877e-08],
        #             [2.1744099321374343e-08, -1.956821429551539e-07, -3.106340537276925e-07, 8.690272190373355e-08],
        #             [-6.743135285330971e-08, -3.835617222375879e-07, 6.258907827506765e-07, -3.315712845815245e-08],
        #             [-4.608125341486133e-08, -1.014224025773705e-07, -6.027023567648257e-07, -1.1293200783123714e-08],
        #             [-5.472341323448938e-07, -2.0368478114120324e-06, -1.620138615639234e-07, -3.928677468945487e-07]]
        #
        # shape_expc = [(2, 6, 30, 30),
        #               (2, 6, 30, 30),
        #               (2, 2, 6, 30, 30),
        #               (2, 2, 6, 30, 30),
        #               (1, 30, 30)]
        #
        # pca_numbers = [(range(1, 3), range(1, 3)),
        #                (range(1, 3), range(1, 3)),
        #                (range(1, 3), range(1, 3)),
        #                (range(1, 3), range(1, 3)),
        #                [1]]

        processing_types = ['ADI', 'SDI+ADI', 'Tsaa']

        expected = [[-0.17615249390982402, -0.7938155399702641, 19.552033067005553, 3.850324378983753e-08],
                    [-0.000643226513845853, -0.026507248922765857, 1.0739620435800206, -3.772431121370883e-08],
                    [0.13478375100474496, 0.5715672449895544, -1.0533917918029834, -4.612469664581486e-08]]

        shape_expc = [(2, 3, 21, 21), (2, 2, 3, 21, 21), (1, 21, 21)]

        pca_numbers = [(range(1, 3), range(1, 3)), (range(1, 3), range(1, 3)), [1]]

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
            # assert np.sum(data) == pytest.approx(expected[i][3], rel=self.limit, abs=0.)
            assert data.shape == shape_expc[i]

            # data = self.pipeline.get_data('basis_single_sdi_'+p_type)
            # assert np.sum(data) == pytest.approx(-1.3886119555248766, rel=self.limit, abs=0.)
            # assert data.shape == (5, 30, 30)

            if p_type == 'Tsaa':
                data = self.pipeline.get_data(f'res_arr_single_sdi_{p_type}1')
                assert np.sum(data) == pytest.approx(-1.3886119555248766, rel=self.limit, abs=0.)
                assert data.shape == (10, 21, 21)

    def test_psf_subtraction_sdi_multi(self) -> None:

        with h5py.File(self.test_dir+'PynPoint_database.hdf5', 'a') as hdf_file:
            hdf_file['config'].attrs['CPU'] = 4

        # processing_types = ['ADI', 'SDI', 'SDI+ADI', 'ADI+SDI', 'Tnan']
        #
        # expected = [[-3.96614620673174e-08, 9.813240333503754e-08, 6.920690625734128e-08, -3.7924737076858164e-08],
        #             [2.1744099321366555e-08, -1.9568214295495464e-07, -3.1063405372801503e-07, 8.690272190365003e-08],
        #             [-6.743135285332267e-08, -3.835617222377436e-07, 6.258907828194748e-07, -3.315712845816095e-08],
        #             [-4.6081253414983635e-08, -1.0142240257765332e-07, -6.027023520146822e-07, -1.1293200783270142e-08],
        #             [0.0011152669134962224, 0.0011030610345340278, 0.001114351549402792, 0.0011150859312946666]]
        #
        # shape_expc = [(2, 6, 30, 30), (2, 6, 30, 30), (2, 2, 6, 30, 30), (2, 2, 6, 30, 30), (2, 30, 30)]

        processing_types = ['SDI', 'Tasp', 'Tnan']

        expected = [[-0.005048156279988383, -0.04357166929029911, 0.06870265135913348, 6.610732512298102e-09],
                    [0.20578060679082302, 0.03699430525568057, 0.14681169595467616, 2.092483992124109e-07],
                    [50.089890230823386, 49.699563934004516, 50.41008576279893, 0.0011147858788599398]]

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
            # assert np.sum(data) == pytest.approx(expected[i][3], rel=self.limit, abs=0.)
            assert data.shape == shape_expc[i]
