import os
import warnings

import pytest
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.fitsreading import FitsReadingModule
from pynpoint.processing.psfpreparation import PSFpreparationModule, AngleInterpolationModule, \
                                               AngleCalculationModule, SDIpreparationModule
from pynpoint.util.tests import create_config, create_star_data, remove_test_data

warnings.simplefilter('always')

limit = 1e-10


class TestPsfPreparation:

    def setup_class(self):

        self.test_dir = os.path.dirname(__file__) + '/'

        create_star_data(path=self.test_dir+'prep')
        create_config(self.test_dir+'PynPoint_config.ini')

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self):

        remove_test_data(self.test_dir, folders=['prep'])

    def test_read_data(self):

        module = FitsReadingModule(name_in='read',
                                   image_tag='read',
                                   input_dir=self.test_dir+'prep')

        self.pipeline.add_module(module)
        self.pipeline.run_module('read')

        data = self.pipeline.get_data('read')
        assert np.allclose(data[0, 25, 25], 2.0926464668090656e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010029494781738066, rtol=limit, atol=0.)
        assert data.shape == (40, 100, 100)

    def test_angle_interpolation(self):

        module = AngleInterpolationModule(name_in='angle1',
                                          data_tag='read')

        self.pipeline.add_module(module)
        self.pipeline.run_module('angle1')

        data = self.pipeline.get_data('header_read/PARANG')
        assert np.allclose(data[0], 0., rtol=limit, atol=0.)
        assert np.allclose(data[15], 7.777777777777778, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 10.0, rtol=limit, atol=0.)
        assert data.shape == (40, )

    def test_angle_calculation(self):

        self.pipeline.set_attribute('read', 'LATITUDE', -25.)
        self.pipeline.set_attribute('read', 'LONGITUDE', -70.)
        self.pipeline.set_attribute('read', 'DIT', 1.)

        self.pipeline.set_attribute('read', 'RA', (90., 90., 90., 90.), static=False)
        self.pipeline.set_attribute('read', 'DEC', (-51., -51., -51., -51.), static=False)
        self.pipeline.set_attribute('read', 'PUPIL', (90., 90., 90., 90.), static=False)

        date = ('2012-12-01T07:09:00.0000', '2012-12-01T07:09:01.0000',
                '2012-12-01T07:09:02.0000', '2012-12-01T07:09:03.0000')

        self.pipeline.set_attribute('read', 'DATE', date, static=False)

        module = AngleCalculationModule(instrument='NACO',
                                        name_in='angle2',
                                        data_tag='read')

        self.pipeline.add_module(module)
        self.pipeline.run_module('angle2')

        data = self.pipeline.get_data('header_read/PARANG')
        assert np.allclose(data[0], -55.04109770947442, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), -54.99858360618869, rtol=limit, atol=0.)
        assert data.shape == (40, )

        self.pipeline.set_attribute('read', 'RA', (60000.0, 60000.0, 60000.0, 60000.0), static=False)

        self.pipeline.set_attribute('read', 'DEC', (-510000., -510000., -510000., -510000.), static=False)

        module = AngleCalculationModule(instrument='SPHERE/IRDIS',
                                        name_in='angle3',
                                        data_tag='read')

        self.pipeline.add_module(module)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module('angle3')

        assert len(warning) == 2

        assert warning[0].message.args[0] == 'For SPHERE data it is recommended to use the ' \
                                             'header keyword \'ESO INS4 DROT2 RA\' to specify ' \
                                             'the object\'s right ascension. The input will be ' \
                                             'parsed accordingly. Using the regular \'RA\' '\
                                             'keyword will lead to wrong parallactic angles.' \

        assert warning[1].message.args[0] == 'For SPHERE data it is recommended to use the ' \
                                             'header keyword \'ESO INS4 DROT2 DEC\' to specify ' \
                                             'the object\'s declination. The input will be ' \
                                             'parsed accordingly. Using the regular \'DEC\' '\
                                             'keyword will lead to wrong parallactic angles.' \

        data = self.pipeline.get_data('header_read/PARANG')
        assert np.allclose(data[0], 170.39102715170227, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 170.46341123194824, rtol=limit, atol=0.)
        assert data.shape == (40, )

        module = AngleCalculationModule(instrument='SPHERE/IFS',
                                        name_in='angle4',
                                        data_tag='read')

        self.pipeline.add_module(module)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module('angle4')

        assert len(warning) == 3

        assert warning[0].message.args[0] == 'AngleCalculationModule has not been tested for ' \
                                             'SPHERE/IFS data.'

        assert warning[1].message.args[0] == 'For SPHERE data it is recommended to use the ' \
                                             'header keyword \'ESO INS4 DROT2 RA\' to specify ' \
                                             'the object\'s right ascension. The input will be ' \
                                             'parsed accordingly. Using the regular \'RA\' '\
                                             'keyword will lead to wrong parallactic angles.' \

        assert warning[2].message.args[0] == 'For SPHERE data it is recommended to use the ' \
                                             'header keyword \'ESO INS4 DROT2 DEC\' to specify ' \
                                             'the object\'s declination. The input will be ' \
                                             'parsed accordingly. Using the regular \'DEC\' '\
                                             'keyword will lead to wrong parallactic angles.' \

        data = self.pipeline.get_data('header_read/PARANG')
        assert np.allclose(data[0], -89.12897284829768, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), -89.02755918786514, rtol=limit, atol=0.)
        assert data.shape == (40, )

    def test_angle_interpolation_mismatch(self):

        self.pipeline.set_attribute('read', 'NDIT', [9, 9, 9, 9], static=False)

        module = AngleInterpolationModule(name_in='angle5',
                                          data_tag='read')

        self.pipeline.add_module(module)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module('angle5')

        assert len(warning) == 1
        assert warning[0].message.args[0] == 'There is a mismatch between the NDIT and NFRAMES ' \
                                             'values. The parallactic angles are calculated ' \
                                             'with a linear interpolation by using NFRAMES ' \
                                             'steps. A frame selection should be applied ' \
                                             'after the parallactic angles are calculated.'

        data = self.pipeline.get_data('header_read/PARANG')
        assert np.allclose(data[0], 0., rtol=limit, atol=0.)
        assert np.allclose(data[15], 7.777777777777778, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 10.0, rtol=limit, atol=0.)
        assert data.shape == (40, )

    def test_psf_preparation_norm_mask(self):

        module = PSFpreparationModule(name_in='prep1',
                                      image_in_tag='read',
                                      image_out_tag='prep1',
                                      mask_out_tag='mask1',
                                      norm=True,
                                      cent_size=0.1,
                                      edge_size=1.0)

        self.pipeline.add_module(module)
        self.pipeline.run_module('prep1')

        data = self.pipeline.get_data('prep1')
        assert np.allclose(data[0, 0, 0], 0., rtol=limit, atol=0.)
        assert np.allclose(data[0, 99, 99], 0., rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.0001690382058762809, rtol=limit, atol=0.)
        assert data.shape == (40, 100, 100)

        data = self.pipeline.get_data('mask1')
        assert np.allclose(data[0, 0], 0., rtol=limit, atol=0.)
        assert np.allclose(data[99, 99], 0., rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.4268, rtol=limit, atol=0.)
        assert data.shape == (100, 100)

    def test_psf_preparation_none(self):

        module = PSFpreparationModule(name_in='prep2',
                                      image_in_tag='read',
                                      image_out_tag='prep2',
                                      mask_out_tag='mask2',
                                      norm=False,
                                      cent_size=None,
                                      edge_size=None)

        self.pipeline.add_module(module)
        self.pipeline.run_module('prep2')

        data = self.pipeline.get_data('prep2')
        assert np.allclose(data[0, 0, 0], 0.00032486907273264834, rtol=limit, atol=0.)
        assert np.allclose(data[0, 25, 25], 2.0926464668090656e-05, rtol=limit, atol=0.)
        assert np.allclose(data[0, 99, 99], -0.000287573978535779, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010029494781738066, rtol=limit, atol=0.)
        assert data.shape == (40, 100, 100)

    def test_psf_preparation_no_mask_out(self):

        module = PSFpreparationModule(name_in='prep3',
                                      image_in_tag='read',
                                      image_out_tag='prep3',
                                      mask_out_tag=None,
                                      norm=False,
                                      cent_size=None,
                                      edge_size=None)

        self.pipeline.add_module(module)
        self.pipeline.run_module('prep3')

        data = self.pipeline.get_data('prep3')
        assert np.allclose(data[0, 0, 0], 0.00032486907273264834, rtol=limit, atol=0.)
        assert np.allclose(data[0, 25, 25], 2.0926464668090656e-05, rtol=limit, atol=0.)
        assert np.allclose(data[0, 99, 99], -0.000287573978535779, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010029494781738066, rtol=limit, atol=0.)
        assert data.shape == (40, 100, 100)

    def test_sdi_preparation(self):

        module = SDIpreparationModule(name_in='sdi',
                                      wavelength=(0.65, 0.6),
                                      width=(0.1, 0.5),
                                      image_in_tag='read',
                                      image_out_tag='sdi')

        self.pipeline.add_module(module)
        self.pipeline.run_module('sdi')

        data = self.pipeline.get_data('sdi')
        assert np.allclose(data[0, 25, 25], -2.6648118007008814e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 2.0042892634995876e-05, rtol=limit, atol=0.)
        assert data.shape == (40, 100, 100)

        attribute = self.pipeline.get_attribute('sdi', 'History: SDIpreparationModule')
        assert attribute == '(line, continuum) = (0.65, 0.6)'
