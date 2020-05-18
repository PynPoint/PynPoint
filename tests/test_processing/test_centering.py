import os

import pytest
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.fitsreading import FitsReadingModule
from pynpoint.processing.centering import StarAlignmentModule, ShiftImagesModule, \
                                          WaffleCenteringModule, FitCenterModule
from pynpoint.processing.extract import StarExtractionModule
from pynpoint.processing.resizing import AddLinesModule
from pynpoint.util.tests import create_config, create_star_data, create_waffle_data, \
                                remove_test_data


class TestCentering:

    def setup_class(self) -> None:

        self.limit = 1e-10
        self.test_dir = os.path.dirname(__file__) + '/'

        create_star_data(self.test_dir+'star')
        create_waffle_data(self.test_dir+'waffle')
        create_config(self.test_dir+'PynPoint_config.ini')

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self) -> None:

        remove_test_data(path=self.test_dir,
                         folders=['star', 'waffle'])

    def test_read_data(self) -> None:

        module = FitsReadingModule(name_in='read1',
                                   image_tag='star',
                                   input_dir=self.test_dir+'star',
                                   overwrite=True,
                                   check=True)

        self.pipeline.add_module(module)

        module = FitsReadingModule(name_in='read2',
                                   image_tag='waffle',
                                   input_dir=self.test_dir+'waffle',
                                   overwrite=True,
                                   check=True)

        self.pipeline.add_module(module)

        self.pipeline.run_module('read1')
        self.pipeline.run_module('read2')

        data = self.pipeline.get_data('star')
        assert np.sum(data) == pytest.approx(105.54278879805277, rel=self.limit, abs=0.)
        assert data.shape == (10, 11, 11)

        data = self.pipeline.get_data('waffle')
        assert np.sum(data) == pytest.approx(4.000000000000196, rel=self.limit, abs=0.)
        assert data.shape == (1, 101, 101)

    def test_star_extract(self) -> None:

        module = StarExtractionModule(name_in='extract1',
                                      image_in_tag='star',
                                      image_out_tag='extract1',
                                      index_out_tag='index',
                                      image_size=0.2,
                                      fwhm_star=0.1,
                                      position=None)

        self.pipeline.add_module(module)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module('extract1')

        assert len(warning) == 1

        assert warning[0].message.args[0] == 'The new dataset that is stored under the tag name ' \
                                             '\'index\' is empty.'

        data = self.pipeline.get_data('extract1')
        assert np.sum(data) == pytest.approx(104.93318507061295, rel=self.limit, abs=0.)
        assert data.shape == (10, 9, 9)

        attr = self.pipeline.get_attribute('extract1', 'STAR_POSITION', static=False)
        assert np.sum(attr) == pytest.approx(100, rel=self.limit, abs=0.)
        assert attr.shape == (10, 2)

    def test_star_align(self) -> None:

        module = StarAlignmentModule(name_in='align1',
                                     image_in_tag='extract1',
                                     ref_image_in_tag=None,
                                     image_out_tag='align1',
                                     accuracy=10,
                                     resize=2.,
                                     num_references=10,
                                     subframe=None)

        self.pipeline.add_module(module)
        self.pipeline.run_module('align1')

        data = self.pipeline.get_data('align1')
        assert np.sum(data) == pytest.approx(104.70747423205349, rel=self.limit, abs=0.)
        assert data.shape == (10, 18, 18)

    def test_star_align_subframe(self) -> None:

        module = StarAlignmentModule(name_in='align2',
                                     image_in_tag='extract1',
                                     ref_image_in_tag=None,
                                     image_out_tag='align2',
                                     accuracy=10,
                                     resize=None,
                                     num_references=10,
                                     subframe=0.1)

        self.pipeline.add_module(module)
        self.pipeline.run_module('align2')

        data = self.pipeline.get_data('align2')
        assert np.sum(data) == pytest.approx(104.39031104541652, rel=self.limit, abs=0.)
        assert data.shape == (10, 9, 9)

    def test_star_align_ref(self) -> None:

        module = StarAlignmentModule(name_in='align3',
                                     image_in_tag='extract1',
                                     ref_image_in_tag='align2',
                                     image_out_tag='align3',
                                     accuracy=10,
                                     resize=None,
                                     num_references=10,
                                     subframe=None)

        self.pipeline.add_module(module)
        self.pipeline.run_module('align3')

        data = self.pipeline.get_data('align3')
        assert np.sum(data) == pytest.approx(104.46997194330757, rel=self.limit, abs=0.)
        assert data.shape == (10, 9, 9)

    def test_star_align_number_ref(self) -> None:

        module = StarAlignmentModule(name_in='align4',
                                     image_in_tag='extract1',
                                     ref_image_in_tag='align2',
                                     image_out_tag='align4',
                                     accuracy=10,
                                     resize=None,
                                     num_references=20,
                                     subframe=None)

        self.pipeline.add_module(module)

        with pytest.warns(UserWarning) as warning:
            self.pipeline.run_module('align4')

        assert len(warning) == 1

        assert warning[0].message.args[0] == 'Number of available images (10) is smaller than ' \
                                             'num_references (20). Using all available images ' \
                                             'instead.'

        data = self.pipeline.get_data('align4')
        assert np.sum(data) == pytest.approx(104.46997194330757, rel=self.limit, abs=0.)
        assert data.shape == (10, 9, 9)

    def test_shift_images_spline(self) -> None:

        module = ShiftImagesModule(shift_xy=(1., 2.),
                                   interpolation='spline',
                                   name_in='shift1',
                                   image_in_tag='align1',
                                   image_out_tag='shift')

        self.pipeline.add_module(module)
        self.pipeline.run_module('shift1')

        data = self.pipeline.get_data('shift')
        assert np.sum(data) == pytest.approx(104.20425101355242, rel=self.limit, abs=0.)
        assert data.shape == (10, 18, 18)

    def test_shift_images_fft(self) -> None:

        module = ShiftImagesModule(shift_xy=(1., 2.),
                                   interpolation='fft',
                                   name_in='shift2',
                                   image_in_tag='align1',
                                   image_out_tag='shift_fft')

        self.pipeline.add_module(module)
        self.pipeline.run_module('shift2')

        data = self.pipeline.get_data('shift_fft')
        assert np.sum(data) == pytest.approx(104.70747423205349, rel=self.limit, abs=0.)
        assert data.shape == (10, 18, 18)

    def test_waffle_center_odd(self) -> None:

        module = AddLinesModule(name_in='add',
                                image_in_tag='star',
                                image_out_tag='star_add',
                                lines=(45, 45, 45, 45))

        self.pipeline.add_module(module)
        self.pipeline.run_module('add')

        data = self.pipeline.get_data('star_add')
        assert np.sum(data) == pytest.approx(105.54278879805278, rel=self.limit, abs=0.)
        assert data.shape == (10, 101, 101)

        module = WaffleCenteringModule(size=0.2,
                                       center=(50, 50),
                                       name_in='waffle',
                                       image_in_tag='star_add',
                                       center_in_tag='waffle',
                                       image_out_tag='center',
                                       radius=35.,
                                       pattern='x',
                                       sigma=0.05)

        self.pipeline.add_module(module)
        self.pipeline.run_module('waffle')

        data = self.pipeline.get_data('center')
        assert np.sum(data) == pytest.approx(104.93318507061295, rel=self.limit, abs=0.)
        assert data.shape == (10, 9, 9)

        attr = self.pipeline.get_attribute('center', 'History: WaffleCenteringModule')
        assert attr == '[x, y] = [50.0, 50.0]'

    def test_waffle_center_even(self) -> None:

        module = AddLinesModule(name_in='add1',
                                image_in_tag='star_add',
                                image_out_tag='star_even',
                                lines=(0, 1, 0, 1))

        self.pipeline.add_module(module)
        self.pipeline.run_module('add1')

        data = self.pipeline.get_data('star_even')
        assert np.sum(data) == pytest.approx(105.54278879805275, rel=self.limit, abs=0.)
        assert data.shape == (10, 102, 102)

        module = AddLinesModule(name_in='add2',
                                image_in_tag='waffle',
                                image_out_tag='waffle_even',
                                lines=(0, 1, 0, 1))

        self.pipeline.add_module(module)
        self.pipeline.run_module('add2')

        data = self.pipeline.get_data('waffle_even')
        assert np.sum(data) == pytest.approx(4.000000000000195, rel=self.limit, abs=0.)
        assert data.shape == (1, 102, 102)

        module = ShiftImagesModule(shift_xy=(0.5, 0.5),
                                   interpolation='spline',
                                   name_in='shift3',
                                   image_in_tag='star_even',
                                   image_out_tag='star_shift')

        self.pipeline.add_module(module)
        self.pipeline.run_module('shift3')

        data = self.pipeline.get_data('star_shift')
        assert np.sum(data) == pytest.approx(105.54278879805274, rel=self.limit, abs=0.)
        assert data.shape == (10, 102, 102)

        module = ShiftImagesModule(shift_xy=(0.5, 0.5),
                                   interpolation='spline',
                                   name_in='shift4',
                                   image_in_tag='waffle_even',
                                   image_out_tag='waffle_shift')

        self.pipeline.add_module(module)
        self.pipeline.run_module('shift4')

        data = self.pipeline.get_data('waffle_shift')
        assert np.sum(data) == pytest.approx(4.000000000000194, rel=self.limit, abs=0.)
        assert data.shape == (1, 102, 102)

        module = WaffleCenteringModule(size=0.2,
                                       center=(50, 50),
                                       name_in='waffle_even',
                                       image_in_tag='star_shift',
                                       center_in_tag='waffle_shift',
                                       image_out_tag='center_even',
                                       radius=35.,
                                       pattern='x',
                                       sigma=0.05)

        self.pipeline.add_module(module)
        self.pipeline.run_module('waffle_even')

        data = self.pipeline.get_data('center_even')
        assert np.sum(data) == pytest.approx(105.22695036281449, rel=self.limit, abs=0.)
        assert data.shape == (10, 9, 9)

        attr = self.pipeline.get_attribute('center_even', 'History: WaffleCenteringModule')
        assert attr == '[x, y] = [50.5, 50.5]'

    def test_fit_center_full(self) -> None:

        module = FitCenterModule(name_in='fit1',
                                 image_in_tag='shift',
                                 fit_out_tag='fit_full',
                                 mask_out_tag='mask',
                                 method='full',
                                 radius=0.05,
                                 sign='positive',
                                 model='gaussian',
                                 guess=(1., 2., 3., 3., 0.01, 0., 0.))

        self.pipeline.add_module(module)
        self.pipeline.run_module('fit1')

        data = self.pipeline.get_data('fit_full')
        assert np.mean(data[:, 0]) == pytest.approx(0.97, rel=1e-2, abs=0.)
        assert np.mean(data[:, 2]) == pytest.approx(2.07, rel=1e-2, abs=0.)
        assert np.mean(data[:, 4]) == pytest.approx(0.37, rel=1e-2, abs=0.)
        assert np.mean(data[:, 6]) == pytest.approx(0.32, rel=1e-2, abs=0.)
        assert np.mean(data[:, 8]) == pytest.approx(21.69, rel=1e-2, abs=0.)
        assert data.shape == (10, 14)

        data = self.pipeline.get_data('mask')
        assert np.sum(data) == pytest.approx(67.43156481961213, rel=self.limit, abs=0.)
        assert data.shape == (10, 18, 18)

    def test_fit_center_mean(self) -> None:

        module = FitCenterModule(name_in='fit2',
                                 image_in_tag='shift',
                                 fit_out_tag='fit_mean',
                                 mask_out_tag=None,
                                 method='mean',
                                 radius=0.1,
                                 sign='positive',
                                 model='moffat',
                                 guess=(1., 2., 3., 3., 0.01, 0., 0., 1.))

        self.pipeline.add_module(module)
        self.pipeline.run_module('fit2')

        data = self.pipeline.get_data('fit_mean')
        assert np.mean(data[:, 0]) == pytest.approx(0.94, rel=1e-2, abs=0.)
        assert np.mean(data[:, 2]) == pytest.approx(2.06, rel=1e-2, abs=0.)
        assert np.mean(data[:, 4]) == pytest.approx(0.08, rel=1e-2, abs=0.)
        assert np.mean(data[:, 6]) == pytest.approx(0.08, rel=1e-2, abs=0.)
        assert np.mean(data[:, 8]) == pytest.approx(0.24, rel=1e-2, abs=0.)
        assert data.shape == (10, 16)

    def test_shift_images_tag(self) -> None:

        module = ShiftImagesModule(shift_xy='fit_full',
                                   interpolation='spline',
                                   name_in='shift5',
                                   image_in_tag='shift',
                                   image_out_tag='shift_tag_1')

        self.pipeline.add_module(module)
        self.pipeline.run_module('shift5')

        data = self.pipeline.get_data('shift_tag_1')
        assert np.sum(data) == pytest.approx(104.11552920880959, rel=1e-6, abs=0.)
        assert data.shape == (10, 18, 18)

    def test_shift_images_tag_mean(self) -> None:

        module = ShiftImagesModule(shift_xy='fit_mean',
                                   interpolation='spline',
                                   name_in='shift6',
                                   image_in_tag='shift',
                                   image_out_tag='shift_tag_2')

        self.pipeline.add_module(module)
        self.pipeline.run_module('shift6')

        data = self.pipeline.get_data('shift_tag_2')
        assert np.sum(data) == pytest.approx(103.42285579230325, rel=1e-6, abs=0.)
        assert data.shape == (10, 18, 18)
