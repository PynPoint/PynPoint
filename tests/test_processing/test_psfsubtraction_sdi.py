import os
import warnings

import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.fitsreading import FitsReadingModule
#from pynpoint.processing.psfpreparation import AngleInterpolationModule, PSFpreparationModule
from pynpoint.processing.psfsubtraction import PcaPsfSubtractionModule
from pynpoint.util.tests import create_config, create_ifs_fake, remove_test_data

warnings.simplefilter('always')

limit = 1e-10


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
                                    tuple([i for i in np.linspace(0., 100., 20)]),
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

        processing_types = ['SDI', 'SDI+ADI', 'ADI+SDI']

        expected = [[2.1744099321374343e-08, -1.956821429551539e-07, -3.106340537276925e-07, 8.690272190373355e-08],
                    [-6.743135285330971e-08, -3.835617222375879e-07, 6.258907827506765e-07, -3.315712845815245e-08],
                    [-4.608125341486133e-08, -1.014224025773705e-07, -6.027023567648257e-07, -1.1293200783123714e-08]]

        shape_expc = [(2, 6, 30, 30),
                      (2, 2, 6, 30, 30),
                      (2, 2, 6, 30, 30)]

        # change sience to sience_prep after available
        for i, p_type in enumerate(processing_types):
            pca = PcaPsfSubtractionModule(pca_numbers=(range(1, 3), range(1, 3)),
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
            assert np.allclose(np.mean(data), expected[i][3], rtol=limit, atol=0.)
            assert data.shape == shape_expc[i]

#            # res array outs are not supported yet
#            data = self.pipeline.get_data('res_arr_single_sdi_' + p_type)
#            assert np.allclose(np.mean(data), expected[i][4], rtol=limit, atol=0.)
#            assert data.shape == (120, 30, 30)

#            # res basis outs are not supported yet
#            data = self.pipeline.get_data('basis_single_sdi_' + p_type)
#            assert np.allclose(np.mean(data), expected[i][5], rtol=limit, atol=0.)
#            assert data.shape == (5, 30, 30)