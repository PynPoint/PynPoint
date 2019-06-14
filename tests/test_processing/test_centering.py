import os
import warnings

import pytest
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.fitsreading import FitsReadingModule
from pynpoint.processing.centering import StarAlignmentModule, ShiftImagesModule, \
                                          StarCenteringModule, WaffleCenteringModule, \
                                          FitCenterModule
from pynpoint.processing.extract import StarExtractionModule
from pynpoint.util.tests import create_config, create_star_data, create_waffle_data, \
                                remove_test_data

warnings.simplefilter('always')

limit = 1e-10

class TestCentering:

    def setup_class(self):

        self.test_dir = os.path.dirname(__file__) + '/'

        create_star_data(path=self.test_dir+'dither',
                         npix_x=100,
                         npix_y=100,
                         x0=[25, 75, 75, 25],
                         y0=[75, 75, 25, 25],
                         parang_start=[0., 25., 50., 75.],
                         parang_end=[25., 50., 75., 100.],
                         exp_no=[1, 2, 3, 4])

        create_star_data(path=self.test_dir+'star_odd',
                         npix_x=101,
                         npix_y=101,
                         x0=[50],
                         y0=[50],
                         parang_start=[0.],
                         parang_end=[25.],
                         exp_no=[1],
                         noise=False)

        create_star_data(path=self.test_dir+'star_even',
                         npix_x=100,
                         npix_y=100,
                         x0=[49.5],
                         y0=[49.5],
                         parang_start=[0.],
                         parang_end=[25.],
                         exp_no=[1],
                         noise=False)

        create_waffle_data(path=self.test_dir+'waffle_odd',
                           npix=101,
                           x_spot=[20., 20., 80., 80.],
                           y_spot=[20., 80., 80., 20.])

        create_waffle_data(path=self.test_dir+'waffle_even',
                           npix=100,
                           x_spot=[20., 20., 79., 79.],
                           y_spot=[20., 79., 79., 20.])

        create_config(self.test_dir+'PynPoint_config.ini')

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self):

        remove_test_data(path=self.test_dir,
                         folders=['dither', 'star_odd', 'star_even', 'waffle_odd', 'waffle_even'])

    def test_read_data(self):

        module = FitsReadingModule(name_in='read1',
                                   image_tag='dither',
                                   input_dir=self.test_dir+'dither',
                                   overwrite=True,
                                   check=True)

        self.pipeline.add_module(module)

        module = FitsReadingModule(name_in='read2',
                                   image_tag='waffle_odd',
                                   input_dir=self.test_dir+'waffle_odd',
                                   overwrite=True,
                                   check=True)

        self.pipeline.add_module(module)

        module = FitsReadingModule(name_in='read3',
                                   image_tag='waffle_even',
                                   input_dir=self.test_dir+'waffle_even',
                                   overwrite=True,
                                   check=True)

        self.pipeline.add_module(module)

        module = FitsReadingModule(name_in='read4',
                                   image_tag='star_odd',
                                   input_dir=self.test_dir+'star_odd',
                                   overwrite=True,
                                   check=True)

        self.pipeline.add_module(module)

        read = FitsReadingModule(name_in='read5',
                                 image_tag='star_even',
                                 input_dir=self.test_dir+'star_even',
                                 overwrite=True,
                                 check=True)

        self.pipeline.add_module(read)

        self.pipeline.run_module('read1')
        self.pipeline.run_module('read2')
        self.pipeline.run_module('read3')
        self.pipeline.run_module('read4')
        self.pipeline.run_module('read5')

        data = self.pipeline.get_data('dither')
        assert np.allclose(data[0, 75, 25], 0.09812948027289994, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010029494781738066, rtol=limit, atol=0.)
        assert data.shape == (40, 100, 100)

        data = self.pipeline.get_data('waffle_odd')
        assert np.allclose(data[0, 20, 20], 0.09806026673451182, rtol=1e-4, atol=0.)
        assert np.allclose(data[0, 20, 80], 0.09806026673451182, rtol=1e-4, atol=0.)
        assert np.allclose(data[0, 80, 20], 0.09806026673451182, rtol=1e-4, atol=0.)
        assert np.allclose(data[0, 80, 80], 0.09806026673451182, rtol=1e-4, atol=0.)
        assert np.allclose(np.mean(data), 0.0003921184197627874, rtol=1e-4, atol=0.)
        assert data.shape == (1, 101, 101)

        data = self.pipeline.get_data('waffle_even')
        assert np.allclose(data[0, 20, 20], 0.09806026673451182, rtol=1e-4, atol=0.)
        assert np.allclose(data[0, 20, 79], 0.09806026673451182, rtol=1e-4, atol=0.)
        assert np.allclose(data[0, 79, 20], 0.09806026673451182, rtol=1e-4, atol=0.)
        assert np.allclose(data[0, 79, 79], 0.09806026673451182, rtol=1e-4, atol=0.)
        assert np.allclose(np.mean(data), 0.00040000000000001953, rtol=1e-4, atol=0.)
        assert data.shape == (1, 100, 100)

        data = self.pipeline.get_data('star_odd')
        assert np.allclose(data[0, 50, 50], 0.09806026673451182, rtol=1e-4, atol=0.)
        assert np.allclose(np.mean(data), 9.80296049406969e-05, rtol=1e-4, atol=0.)
        assert data.shape == (10, 101, 101)

        data = self.pipeline.get_data('star_even')
        assert np.allclose(data[0, 49, 49], 0.08406157361512759, rtol=1e-4, atol=0.)
        assert np.allclose(data[0, 49, 50], 0.08406157361512759, rtol=1e-4, atol=0.)
        assert np.allclose(data[0, 50, 49], 0.08406157361512759, rtol=1e-4, atol=0.)
        assert np.allclose(data[0, 50, 50], 0.08406157361512759, rtol=1e-4, atol=0.)
        assert np.allclose(np.mean(data), 9.99999999999951e-05, rtol=1e-4, atol=0.)
        assert data.shape == (10, 100, 100)

    def test_star_extract(self):

        module = StarExtractionModule(name_in='extract1',
                                      image_in_tag='dither',
                                      image_out_tag='extract1',
                                      index_out_tag='index',
                                      image_size=1.0,
                                      fwhm_star=0.1,
                                      position=None)

        self.pipeline.add_module(module)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module('extract1')

        assert len(warning) == 1
        assert warning[0].message.args[0] == 'The new dataset that is stored under the tag name ' \
                                             '\'index\' is empty.'

        data = self.pipeline.get_data('extract1')

        assert np.allclose(data[0, 19, 19], 0.09812948027289994, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.0006578482216906739, rtol=limit, atol=0.)
        assert data.shape == (40, 39, 39)

        attr = self.pipeline.get_attribute('extract1', 'STAR_POSITION', static=False)
        assert attr[10, 0] == attr[10, 1] == 75

    def test_star_align(self):

        module = StarAlignmentModule(name_in='align',
                                     image_in_tag='extract1',
                                     ref_image_in_tag=None,
                                     image_out_tag='align',
                                     accuracy=10,
                                     resize=2.)

        self.pipeline.add_module(module)
        self.pipeline.run_module('align')

        data = self.pipeline.get_data('align')
        assert np.allclose(data[0, 39, 39], 0.023556628129942758, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00016446205542266837, rtol=limit, atol=0.)
        assert data.shape == (40, 78, 78)

    def test_shift_images_spline(self):

        module = ShiftImagesModule(shift_xy=(6., 4.),
                                   interpolation='spline',
                                   name_in='shift1',
                                   image_in_tag='align',
                                   image_out_tag='shift')

        self.pipeline.add_module(module)
        self.pipeline.run_module('shift1')

        data = self.pipeline.get_data('shift')
        assert np.allclose(data[0, 43, 45], 0.023556628129942764, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00016430682224782259, rtol=limit, atol=0.)
        assert data.shape == (40, 78, 78)

    def test_shift_images_fft(self):

        module = ShiftImagesModule(shift_xy=(6., 4.),
                                   interpolation='fft',
                                   name_in='shift2',
                                   image_in_tag='align',
                                   image_out_tag='shift_fft')

        self.pipeline.add_module(module)
        self.pipeline.run_module('shift2')

        data = self.pipeline.get_data('shift_fft')
        assert np.allclose(data[0, 43, 45], 0.023556628129942764, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00016446205542266847, rtol=limit, atol=0.)
        assert data.shape == (40, 78, 78)

    def test_star_center_full(self):

        with pytest.warns(DeprecationWarning) as warning:
            module = StarCenteringModule(name_in='center1',
                                         image_in_tag='shift',
                                         image_out_tag='center',
                                         mask_out_tag='mask',
                                         fit_out_tag=None,
                                         method='full',
                                         interpolation='spline',
                                         radius=0.05,
                                         sign='positive',
                                         model='gaussian',
                                         guess=(6., 4., 3., 3., 1., 0., 0.))

            self.pipeline.add_module(module)

        assert len(warning) == 1
        assert warning[0].message.args[0] == 'The StarCenteringModule will be deprecated in a ' \
                                             'future release. Please use the FitCenterModule ' \
                                             'and ShiftImagesModule instead.'

        self.pipeline.run_module('center1')

        data = self.pipeline.get_data('center')
        assert np.allclose(data[0, 39, 39], 0.02356308097293422, rtol=1e-4, atol=0.)
        assert np.allclose(np.mean(data), 0.000164306294368963, rtol=1e-4, atol=0.)
        assert data.shape == (40, 78, 78)

        data = self.pipeline.get_data('mask')
        assert np.allclose(data[0, 43, 45], 0.023556628129942764, rtol=limit, atol=0.)
        assert np.allclose(data[0, 43, 55], 0.0, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010827527282995304, rtol=limit, atol=0.)
        assert data.shape == (40, 78, 78)

    def test_star_center_mean(self):

        with pytest.warns(DeprecationWarning) as warning:
            module = StarCenteringModule(name_in='center2',
                                         image_in_tag='shift',
                                         image_out_tag='center',
                                         mask_out_tag=None,
                                         fit_out_tag=None,
                                         method='mean',
                                         interpolation='bilinear',
                                         radius=0.05,
                                         sign='positive',
                                         model='gaussian',
                                         guess=(6., 4., 3., 3., 1., 0., 0.))

            self.pipeline.add_module(module)

        assert len(warning) == 1
        assert warning[0].message.args[0] == 'The StarCenteringModule will be deprecated in a ' \
                                             'future release. Please use the FitCenterModule ' \
                                             'and ShiftImagesModule instead.'

        self.pipeline.run_module('center2')

        data = self.pipeline.get_data('center')
        assert np.allclose(data[0, 39, 39], 0.023556482678860322, rtol=1e-4, atol=0.)
        assert np.allclose(np.mean(data), 0.00016430629447868552, rtol=1e-4, atol=0.)
        assert data.shape == (40, 78, 78)

    def test_star_center_moffat(self):

        with pytest.warns(DeprecationWarning) as warning:
            module = StarCenteringModule(name_in='center3',
                                         image_in_tag='shift',
                                         image_out_tag='center',
                                         mask_out_tag=None,
                                         fit_out_tag='center_fit',
                                         method='mean',
                                         interpolation='spline',
                                         radius=0.05,
                                         sign='positive',
                                         model='moffat',
                                         guess=(6., 4., 3., 3., 1., 0., 0., 1.))

            self.pipeline.add_module(module)

        assert len(warning) == 1
        assert warning[0].message.args[0] == 'The StarCenteringModule will be deprecated in a ' \
                                             'future release. Please use the FitCenterModule ' \
                                             'and ShiftImagesModule instead.'

        with pytest.warns(RuntimeWarning) as warning:
            self.pipeline.run_module('center3')

        assert len(warning) == 4
        assert warning[0].message.args[0] == 'invalid value encountered in sqrt'
        assert warning[1].message.args[0] == 'invalid value encountered in sqrt'
        assert warning[2].message.args[0] == 'invalid value encountered in sqrt'
        assert warning[3].message.args[0] == 'invalid value encountered in sqrt'

        data = self.pipeline.get_data('center')
        assert np.allclose(data[0, 39, 39], 0.023556482678860322, rtol=1e-4, atol=0.)
        assert np.allclose(np.mean(data), 0.00016430629447868552, rtol=1e-4, atol=0.)
        assert data.shape == (40, 78, 78)

    def test_waffle_center_odd(self):

        module = WaffleCenteringModule(size=2.,
                                       center=(50, 50),
                                       name_in='waffle_odd',
                                       image_in_tag='star_odd',
                                       center_in_tag='waffle_odd',
                                       image_out_tag='center_odd',
                                       radius=42.5,
                                       pattern='x',
                                       sigma=0.135)

        self.pipeline.add_module(module)
        self.pipeline.run_module('waffle_odd')

        data = self.pipeline.get_data('star_odd')
        assert np.allclose(data[0, 50, 50], 0.09806026673451182, rtol=limit, atol=0.)

        data = self.pipeline.get_data('center_odd')
        assert np.allclose(data[0, 37, 37], 0.0980602667345118, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00017777777777778643, rtol=limit, atol=0.)
        assert data.shape == (10, 75, 75)

        attribute = self.pipeline.get_attribute('center_odd', 'History: WaffleCenteringModule')
        assert attribute == '[x, y] = [50.0, 50.0]'

    def test_waffle_center_even(self):

        module = WaffleCenteringModule(size=2.,
                                       center=(50, 50),
                                       name_in='waffle_even',
                                       image_in_tag='star_even',
                                       center_in_tag='waffle_even',
                                       image_out_tag='center_even',
                                       radius=42.5,
                                       pattern='x',
                                       sigma=0.135)

        self.pipeline.add_module(module)
        self.pipeline.run_module('waffle_even')

        data = self.pipeline.get_data('star_even')
        assert np.allclose(data[0, 49, 49], 0.08406157361512759, rtol=limit, atol=0.)
        assert np.allclose(data[0, 49, 50], 0.08406157361512759, rtol=limit, atol=0.)
        assert np.allclose(data[0, 50, 49], 0.08406157361512759, rtol=limit, atol=0.)
        assert np.allclose(data[0, 50, 50], 0.08406157361512759, rtol=limit, atol=0.)

        data = self.pipeline.get_data('center_even')
        assert np.allclose(data[0, 37, 37], 0.09778822940550569, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00017777777777778643, rtol=limit, atol=0.)
        assert data.shape == (10, 75, 75)

        attribute = self.pipeline.get_attribute('center_even', 'History: WaffleCenteringModule')
        assert attribute == '[x, y] = [49.5, 49.5]'

    def test_fit_center_full(self):

        module = FitCenterModule(name_in='fit1',
                                 image_in_tag='shift',
                                 fit_out_tag='fit_full',
                                 mask_out_tag='mask',
                                 method='full',
                                 radius=0.05,
                                 sign='positive',
                                 model='gaussian',
                                 guess=(6., 4., 3., 3., 0.01, 0., 0.))

        self.pipeline.add_module(module)
        self.pipeline.run_module('fit1')

        data = self.pipeline.get_data('fit_full')
        assert np.allclose(np.mean(data[:, 0]), 5.999068486622676, rtol=1e-3, atol=0.)
        assert np.allclose(np.mean(data[:, 2]), 4.000055166165185, rtol=1e-3, atol=0.)
        assert np.allclose(np.mean(data[:, 4]), 0.08106141046470318, rtol=1e-3, atol=0.)
        assert np.allclose(np.mean(data[:, 6]), 0.0810026137349896, rtol=1e-3, atol=0.)
        assert np.allclose(np.mean(data[:, 8]), 0.024462594420743763, rtol=1e-3, atol=0.)
        assert np.allclose(np.mean(data[:, 12]), 3.0281141786814477e-05, rtol=1e-3, atol=0.)
        assert data.shape == (40, 14)

        data = self.pipeline.get_data('mask')
        assert np.allclose(data[0, 43, 45], 0.023556628129942764, rtol=limit, atol=0.)
        assert np.allclose(data[0, 43, 55], 0.0, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010827527282995305, rtol=limit, atol=0.)
        assert data.shape == (40, 78, 78)

    def test_fit_center_mean(self):

        module = FitCenterModule(name_in='fit2',
                                 image_in_tag='shift',
                                 fit_out_tag='fit_mean',
                                 mask_out_tag=None,
                                 method='mean',
                                 radius=0.05,
                                 sign='positive',
                                 model='moffat',
                                 guess=(6., 4., 3., 3., 0.01, 0., 0., 1.))

        self.pipeline.add_module(module)
        self.pipeline.run_module('fit2')

        data = self.pipeline.get_data('fit_mean')
        assert np.allclose(np.mean(data[:, 0]), 5.999072568941366, rtol=1e-3, atol=0.)
        assert np.allclose(np.mean(data[:, 2]), 4.000051869708742, rtol=1e-3, atol=0.)
        assert np.allclose(np.mean(data[:, 4]), 0.08384036587023312, rtol=1e-3, atol=0.)
        assert np.allclose(np.mean(data[:, 6]), 0.08379313488754872, rtol=1e-3, atol=0.)
        assert np.allclose(np.mean(data[:, 8]), 0.025631328037795074, rtol=1e-3, atol=0.)
        assert np.allclose(np.mean(data[:, 12]), -0.0011275279023032867, rtol=1e-3, atol=0.)
        assert data.shape == (40, 16)

    def test_shift_images_tag(self):

        module = ShiftImagesModule(shift_xy='fit_full',
                                   interpolation='spline',
                                   name_in='shift3',
                                   image_in_tag='shift',
                                   image_out_tag='shift_tag_1')

        self.pipeline.add_module(module)
        self.pipeline.run_module('shift3')

        data = self.pipeline.get_data('shift_tag_1')
        assert np.allclose(data[0, 39, 39], 0.023563080974545528, rtol=1e-6, atol=0.)
        assert np.allclose(np.mean(data), 0.0001643062943690491, rtol=1e-6, atol=0.)
        assert data.shape == (40, 78, 78)

    def test_shift_images_tag_mean(self):

        module = ShiftImagesModule(shift_xy='fit_mean',
                                   interpolation='spline',
                                   name_in='shift4',
                                   image_in_tag='shift',
                                   image_out_tag='shift_tag_2')

        self.pipeline.add_module(module)
        self.pipeline.run_module('shift4')
        data = self.pipeline.get_data('shift_tag_2')
        assert np.allclose(data[0, 20, 31], 5.348337712000518e-05, rtol=1e-6, atol=0.)
        assert np.allclose(np.mean(data), 0.00016430318227546225, rtol=1e-6, atol=0.)
        assert data.shape == (40, 78, 78)
