import os
import warnings
import h5py

import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.fitsreading import FitsReadingModule
#from pynpoint.processing.psfpreparation import AngleInterpolationModule, PSFpreparationModule
from pynpoint.processing.psfsubtraction import PcaPsfSubtractionModule
from pynpoint.util.tests import create_config, create_ifs_fake, remove_test_data

warnings.simplefilter('always')

limit = 1e-5


class TestPsfSubtractionSdi:

    def setup_class(self):

        self.test_dir = os.path.dirname(__file__) + '/'

        create_ifs_fake(path=self.test_dir+'science')

        create_config(self.test_dir+'PynPoint_config.ini')

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self):

        remove_test_data(self.test_dir, folders=['science', ])

    def test_read_data(self):

        read = FitsReadingModule(name_in='read',
                                 image_tag='science',
                                 input_dir=self.test_dir+'science',
                                 ifs_data=True)

        self.pipeline.add_module(read)
        self.pipeline.run_module('read')

        self.pipeline.set_attribute('science', 'WAVELENGTH',
                                    tuple([0.953 + i*0.0190526315789474 for i in range(6)]),
                                    static=False)

        self.pipeline.set_attribute('science', 'PARANG',
                                    tuple([i for i in np.linspace(0., 100., 10)]),
                                    static=False)

# The PSFpreparationModule does currently not support SDI
#    def test_angle_interpolation(self):
#
#        angle = AngleInterpolationModule(name_in='angle',
#                                         data_tag='science')
#
#        self.pipeline.add_module(angle)
#        self.pipeline.run_module('angle')
#
#        data = self.pipeline.get_data('header_science/PARANG')
#        assert np.allclose(data[0], 0., rtol=limit, atol=0.)
#        assert np.allclose(data[91], 78.94736842105263, rtol=limit, atol=0.)
#        assert np.allclose(np.mean(data), 50.0, rtol=limit, atol=0.)
#        assert data.shape == (120, )
#
#    def test_psf_preparation(self):
#
#        prep = PSFpreparationModule(name_in='prep',
#                                    image_in_tag='science',
#                                    image_out_tag='science_prep',
#                                    mask_out_tag=None,
#                                    norm=False,
#                                    resize=None,
#                                    cent_size=0.2,
#                                    edge_size=1.0)
#
#        self.pipeline.add_module(prep)
#        self.pipeline.run_module('prep')
#
#        data = self.pipeline.get_data('science_prep')
#        assert np.allclose(data[0, 0, 0], 0.0, rtol=limit, atol=0.)
#        assert np.allclose(data[0, 8, 8], 0.0003621069828250913, rtol=limit, atol=0.)
#        assert np.allclose(data[0, 29, 29], 0.0, rtol=limit, atol=0.)
#        assert np.allclose(np.mean(data), 4.16290438829479e-06, rtol=limit, atol=0.)
#        assert data.shape == (120, 30, 30)

    def test_psf_subtraction_pca_sdi(self):

#        processing_types = ['ADI', 'SDI', 'SDI+ADI', 'ADI+SDI', 'Tsaa']
#
#        expected = [[-3.9661462067317356e-08, 9.813240333505746e-08, 6.920690625734549e-08, -3.792473707685877e-08],
#                    [2.1744099321374343e-08, -1.956821429551539e-07, -3.106340537276925e-07, 8.690272190373355e-08],
#                    [-6.743135285330971e-08, -3.835617222375879e-07, 6.258907827506765e-07, -3.315712845815245e-08],
#                    [-4.608125341486133e-08, -1.014224025773705e-07, -6.027023567648257e-07, -1.1293200783123714e-08],
#                    [-5.472341323448938e-07, -2.0368478114120324e-06, -1.620138615639234e-07, -3.928677468945487e-07]]
#
#        shape_expc = [(2, 6, 30, 30),
#                      (2, 6, 30, 30),
#                      (2, 2, 6, 30, 30),
#                      (2, 2, 6, 30, 30),
#                      (1, 30, 30)]
#        
#        pca_numbers = [(range(1, 3), range(1, 3)),
#                       (range(1, 3), range(1, 3)),
#                       (range(1, 3), range(1, 3)),
#                       (range(1, 3), range(1, 3)),
#                       [1]]
        
        
        processing_types = ['ADI', 'SDI+ADI', 'Tsaa']

        expected = [[5.5244158736150073e-08, -3.892085139880991e-07, -8.12624165698584e-07, 3.850324378983753e-08],
                    [-2.4088885146364674e-08, -1.6202057980072862e-07, -5.106228277062057e-07, -3.772431121370883e-08],
                    [-4.612469664581462e-08, -1.4716306329639351e-06, -3.6780528183706394e-07, -4.612469664581486e-08]]

        shape_expc = [(2, 6, 30, 30),
                      (2, 2, 6, 30, 30),
                      (1, 30, 30)]
        
        pca_numbers = [(range(1, 3), range(1, 3)),
                       (range(1, 3), range(1, 3)),
                       [1]]

        # change sience to sience_prep after available
        for i, p_type in enumerate(processing_types):
            pca = PcaPsfSubtractionModule(pca_numbers=pca_numbers[i],
                                          name_in='pca_single_sdi_' + p_type,
                                          images_in_tag='science',
                                          reference_in_tag='science',
                                          res_mean_tag='res_mean_single_sdi_' + p_type,
                                          res_median_tag='res_median_single_sdi_' + p_type,
                                          res_weighted_tag='res_weighted_single_sdi_' + p_type,
                                          res_rot_mean_clip_tag='res_clip_single_sdi_' + p_type,
                                          res_arr_out_tag='res_arr_single_sdi_' + p_type,
                                          basis_out_tag='basis_single_sdi_' + p_type,
                                          extra_rot=-15.,
                                          subtract_mean=True,
                                          processing_type=p_type)

            self.pipeline.add_module(pca)
            self.pipeline.run_module('pca_single_sdi_' + p_type)

            data = self.pipeline.get_data('res_mean_single_sdi_' + p_type)
            assert np.allclose(np.mean(data), expected[i][0], rtol=limit, atol=0.)
            assert data.shape == shape_expc[i]

            data = self.pipeline.get_data('res_median_single_sdi_' + p_type)
            assert np.allclose(np.mean(data), expected[i][1], rtol=limit, atol=0.)
            assert data.shape == shape_expc[i]

            data = self.pipeline.get_data('res_weighted_single_sdi_' + p_type)
            assert np.allclose(np.mean(data), expected[i][2], rtol=limit, atol=0.)
            assert data.shape == shape_expc[i]

            data = self.pipeline.get_data('res_clip_single_sdi_' + p_type)
#            assert np.allclose(np.mean(data), expected[i][3], rtol=limit, atol=0.)
#            assert data.shape == shape_expc[i]
            
            if p_type == 'Tsaa':
                data = self.pipeline.get_data('res_arr_single_sdi_' + p_type + '1')
                assert np.allclose(np.mean(data),8.994101776941848e-07, rtol=limit, atol=0.)
                assert data.shape == (10, 30, 30)

#                data = self.pipeline.get_data('basis_single_sdi_' + p_type)
#                assert np.allclose(np.mean(data), expected[i][5], rtol=limit, atol=0.)
#                assert data.shape == (5, 30, 30)

    def test_multi_psf_subtraction_pca_sdi(self):
        
        with h5py.File(self.test_dir+'PynPoint_database.hdf5', 'a') as hdf_file:
            hdf_file['config'].attrs['CPU'] = 4
            
#        processing_types = ['ADI', 'SDI', 'SDI+ADI', 'ADI+SDI', 'Tnan']
#
#        expected = [[-3.96614620673174e-08, 9.813240333503754e-08, 6.920690625734128e-08, -3.7924737076858164e-08],
#                    [2.1744099321366555e-08, -1.9568214295495464e-07, -3.1063405372801503e-07, 8.690272190365003e-08],
#                    [-6.743135285332267e-08, -3.835617222377436e-07, 6.258907828194748e-07, -3.315712845816095e-08],
#                    [-4.6081253414983635e-08, -1.0142240257765332e-07, -6.027023520146822e-07, -1.1293200783270142e-08],
#                    [0.0011152669134962224, 0.0011030610345340278, 0.001114351549402792, 0.0011150859312946666]]
#
#        shape_expc = [(2, 6, 30, 30),
#                      (2, 6, 30, 30),
#                      (2, 2, 6, 30, 30),
#                      (2, 2, 6, 30, 30),
#                      (2, 30, 30)]
        
        processing_types = ['SDI', 'Tasp', 'Tnan']

        expected = [[1.0608921728272943e-08, -2.1214914674705469e-07, -1.1167444221274486e-06, 6.610732512298102e-09],
                    [2.1107129270565494e-07, -3.214247336073621e-11, 9.622357525155962e-08, 2.092483992124109e-07],
                    [0.0011147593991561037, 0.0011012846214997752, 0.001107264743685533, 0.0011147858788599398]]

        shape_expc = [(2, 6, 30, 30),
                      (2, 2, 30, 30),
                      (2, 30, 30)]

        # change sience to sience_prep after available
        for i, p_type in enumerate(processing_types):
            pca = PcaPsfSubtractionModule(pca_numbers=(range(1, 3), range(1, 3)),
                                          name_in='pca_multi_sdi_' + p_type,
                                          images_in_tag='science',
                                          reference_in_tag='science',
                                          res_mean_tag='res_mean_multi_sdi_' + p_type,
                                          res_median_tag='res_median_multi_sdi_' + p_type,
                                          res_weighted_tag='res_weighted_multi_sdi_' + p_type,
                                          res_rot_mean_clip_tag='res_clip_multi_sdi_' + p_type,
                                          res_arr_out_tag=None,
                                          basis_out_tag=None,
                                          extra_rot=-15.,
                                          subtract_mean=True,
                                          processing_type=p_type)

            self.pipeline.add_module(pca)
            self.pipeline.run_module('pca_multi_sdi_' + p_type)

            data = self.pipeline.get_data('res_mean_multi_sdi_' + p_type)
            assert np.allclose(np.mean(data), expected[i][0], rtol=limit, atol=0.)
            assert data.shape == shape_expc[i]

            data = self.pipeline.get_data('res_median_multi_sdi_' + p_type)
            assert np.allclose(np.mean(data), expected[i][1], rtol=limit, atol=0.)
            assert data.shape == shape_expc[i]

            data = self.pipeline.get_data('res_weighted_multi_sdi_' + p_type)
            assert np.allclose(np.mean(data), expected[i][2], rtol=limit, atol=0.)
            assert data.shape == shape_expc[i]

            data = self.pipeline.get_data('res_clip_multi_sdi_' + p_type)
#            assert np.allclose(np.mean(data), expected[i][3], rtol=limit, atol=0.)
#            assert data.shape == shape_expc[i]