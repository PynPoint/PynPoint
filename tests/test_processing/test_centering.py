import os
import warnings

import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.fitsreading import FitsReadingModule
from pynpoint.processing.centering import StarExtractionModule, StarAlignmentModule, \
                                          ShiftImagesModule, StarCenteringModule, \
                                          WaffleCenteringModule
from pynpoint.util.tests import create_config, create_star_data, create_waffle_data, \
                                remove_test_data

warnings.simplefilter("always")

limit = 1e-10

class TestStarAlignment(object):

    def setup_class(self):

        self.test_dir = os.path.dirname(__file__) + "/"

        create_star_data(path=self.test_dir+"dither",
                         npix_x=100,
                         npix_y=100,
                         x0=[25, 75, 75, 25],
                         y0=[75, 75, 25, 25],
                         parang_start=[0., 25., 50., 75.],
                         parang_end=[25., 50., 75., 100.],
                         exp_no=[1, 2, 3, 4])

        create_star_data(path=self.test_dir+"star_odd",
                         npix_x=101,
                         npix_y=101,
                         x0=[50],
                         y0=[50],
                         parang_start=[0.],
                         parang_end=[25.],
                         exp_no=[1],
                         noise=False)

        create_star_data(path=self.test_dir+"star_even",
                         npix_x=100,
                         npix_y=100,
                         x0=[49.5],
                         y0=[49.5],
                         parang_start=[0.],
                         parang_end=[25.],
                         exp_no=[1],
                         noise=False)

        create_waffle_data(path=self.test_dir+"waffle_odd",
                           npix=101,
                           x_waffle=[20., 20., 80., 80.],
                           y_waffle=[20., 80., 80., 20.])

        create_waffle_data(path=self.test_dir+"waffle_even",
                           npix=100,
                           x_waffle=[20., 20., 79., 79.],
                           y_waffle=[20., 79., 79., 20.])

        create_config(self.test_dir+"PynPoint_config.ini")

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self):

        remove_test_data(self.test_dir, folders=["dither", "star_odd", "star_even", "waffle_odd", "waffle_even"])

    def test_read_data(self):

        read = FitsReadingModule(name_in="read1",
                                 image_tag="dither",
                                 input_dir=self.test_dir+"dither",
                                 overwrite=True,
                                 check=True)

        self.pipeline.add_module(read)

        read = FitsReadingModule(name_in="read2",
                                 image_tag="waffle_odd",
                                 input_dir=self.test_dir+"waffle_odd",
                                 overwrite=True,
                                 check=True)

        self.pipeline.add_module(read)

        read = FitsReadingModule(name_in="read3",
                                 image_tag="waffle_even",
                                 input_dir=self.test_dir+"waffle_even",
                                 overwrite=True,
                                 check=True)

        self.pipeline.add_module(read)

        read = FitsReadingModule(name_in="read4",
                                 image_tag="star_odd",
                                 input_dir=self.test_dir+"star_odd",
                                 overwrite=True,
                                 check=True)

        self.pipeline.add_module(read)

        read = FitsReadingModule(name_in="read5",
                                 image_tag="star_even",
                                 input_dir=self.test_dir+"star_even",
                                 overwrite=True,
                                 check=True)

        self.pipeline.add_module(read)

        self.pipeline.run_module("read1")
        self.pipeline.run_module("read2")
        self.pipeline.run_module("read3")
        self.pipeline.run_module("read4")
        self.pipeline.run_module("read5")

        data = self.pipeline.get_data("dither")
        assert np.allclose(data[0, 75, 25], 0.09812948027289994, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010029494781738066, rtol=limit, atol=0.)
        assert data.shape == (40, 100, 100)

        data = self.pipeline.get_data("waffle_odd")
        assert np.allclose(data[0, 20, 20], 0.09806026673451182, rtol=1e-4, atol=0.)
        assert np.allclose(data[0, 20, 80], 0.09806026673451182, rtol=1e-4, atol=0.)
        assert np.allclose(data[0, 80, 20], 0.09806026673451182, rtol=1e-4, atol=0.)
        assert np.allclose(data[0, 80, 80], 0.09806026673451182, rtol=1e-4, atol=0.)
        assert np.allclose(np.mean(data), 0.0003921184197627874, rtol=1e-4, atol=0.)
        assert data.shape == (1, 101, 101)

        data = self.pipeline.get_data("waffle_even")
        assert np.allclose(data[0, 20, 20], 0.09806026673451182, rtol=1e-4, atol=0.)
        assert np.allclose(data[0, 20, 79], 0.09806026673451182, rtol=1e-4, atol=0.)
        assert np.allclose(data[0, 79, 20], 0.09806026673451182, rtol=1e-4, atol=0.)
        assert np.allclose(data[0, 79, 79], 0.09806026673451182, rtol=1e-4, atol=0.)
        assert np.allclose(np.mean(data), 0.00040000000000001953, rtol=1e-4, atol=0.)
        assert data.shape == (1, 100, 100)

        data = self.pipeline.get_data("star_odd")
        assert np.allclose(data[0, 50, 50], 0.09806026673451182, rtol=1e-4, atol=0.)
        assert np.allclose(np.mean(data), 9.80296049406969e-05, rtol=1e-4, atol=0.)
        assert data.shape == (10, 101, 101)

        data = self.pipeline.get_data("star_even")
        assert np.allclose(data[0, 49, 49], 0.08406157361512759, rtol=1e-4, atol=0.)
        assert np.allclose(data[0, 49, 50], 0.08406157361512759, rtol=1e-4, atol=0.)
        assert np.allclose(data[0, 50, 49], 0.08406157361512759, rtol=1e-4, atol=0.)
        assert np.allclose(data[0, 50, 50], 0.08406157361512759, rtol=1e-4, atol=0.)
        assert np.allclose(np.mean(data), 9.99999999999951e-05, rtol=1e-4, atol=0.)
        assert data.shape == (10, 100, 100)

    def test_star_extract_full(self):

        extract = StarExtractionModule(name_in="extract1",
                                       image_in_tag="dither",
                                       image_out_tag="extract1",
                                       image_size=1.0,
                                       fwhm_star=0.1,
                                       position=None)

        self.pipeline.add_module(extract)
        self.pipeline.run_module("extract1")

        data = self.pipeline.get_data("extract1")
        assert np.allclose(data[0, 19, 19], 0.09812948027289994, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.0006578482216906739, rtol=limit, atol=0.)
        assert data.shape == (40, 39, 39)

        data = self.pipeline.get_data("header_extract1/STAR_POSITION")
        assert data[10, 0] == data[10, 1] == 75

    def test_star_extract_subframe(self):

        top_left = np.array([25, 75, 0.5])
        top_left = np.broadcast_to(top_left, (10, 3))

        top_right = np.array([75, 75, 0.5])
        top_right = np.broadcast_to(top_right, (10, 3))

        bottom_right = np.array([75, 25, 0.5])
        bottom_right = np.broadcast_to(bottom_right, (10, 3))

        bottom_left = np.array([25, 25, 0.5])
        bottom_left = np.broadcast_to(bottom_left, (10, 3))

        position = np.concatenate((top_left, top_right, bottom_right, bottom_left))

        extract = StarExtractionModule(name_in="extract2",
                                       image_in_tag="dither",
                                       image_out_tag="extract2",
                                       image_size=1.0,
                                       fwhm_star=0.1,
                                       position=position)

        self.pipeline.add_module(extract)
        self.pipeline.run_module("extract2")

        data = self.pipeline.get_data("extract2")
        assert np.allclose(data[0, 19, 19], 0.09812948027289994, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.0006578482216906739, rtol=limit, atol=0.)
        assert data.shape == (40, 39, 39)

        data = self.pipeline.get_data("header_extract2/STAR_POSITION")
        assert data[10, 0] == data[10, 1] == 75

    def test_star_align(self):

        align = StarAlignmentModule(name_in="align",
                                    image_in_tag="extract1",
                                    ref_image_in_tag="extract2",
                                    image_out_tag="align",
                                    accuracy=10,
                                    resize=2)

        self.pipeline.add_module(align)
        self.pipeline.run_module("align")

        data = self.pipeline.get_data("align")
        assert np.allclose(data[0, 39, 39], 0.023556628129942758, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00016446205542266837, rtol=limit, atol=0.)
        assert data.shape == (40, 78, 78)

    def test_shift_images_spline(self):

        shift = ShiftImagesModule(shift_xy=(6., 4.),
                                  interpolation="spline",
                                  name_in="shift1",
                                  image_in_tag="align",
                                  image_out_tag="shift")

        self.pipeline.add_module(shift)
        self.pipeline.run_module("shift1")

        data = self.pipeline.get_data("shift")
        assert np.allclose(data[0, 43, 45], 0.023556628129942764, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00016430682224782259, rtol=limit, atol=0.)
        assert data.shape == (40, 78, 78)

    def test_shift_images_fft(self):

        shift = ShiftImagesModule(shift_xy=(6., 4.),
                                  interpolation="spline",
                                  name_in="shift2",
                                  image_in_tag="align",
                                  image_out_tag="fft")

        self.pipeline.add_module(shift)
        self.pipeline.run_module("shift2")

        data = self.pipeline.get_data("fft")
        assert np.allclose(data[0, 43, 45], 0.023556628129942764, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00016430682224782259, rtol=limit, atol=0.)
        assert data.shape == (40, 78, 78)

    def test_star_center_full(self):

        center = StarCenteringModule(name_in="center1",
                                     image_in_tag="shift",
                                     image_out_tag="center",
                                     mask_out_tag="mask",
                                     fit_out_tag="center_fit",
                                     method="full",
                                     interpolation="spline",
                                     radius=0.05,
                                     sign="positive",
                                     guess=(6., 4., 1., 1., 1., 0.))

        self.pipeline.add_module(center)
        self.pipeline.run_module("center1")

        data = self.pipeline.get_data("center")
        assert np.allclose(data[0, 39, 39], 0.023563039729627436, rtol=1e-4, atol=0.)
        assert np.allclose(np.mean(data), 0.00016430629447868552, rtol=1e-4, atol=0.)
        assert data.shape == (40, 78, 78)

        data = self.pipeline.get_data("mask")
        assert np.allclose(data[0, 43, 45], 0.023556628129942764, rtol=limit, atol=0.)
        assert np.allclose(data[0, 43, 55], 0.0, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010827527282995304, rtol=limit, atol=0.)
        assert data.shape == (40, 78, 78)

    def test_star_center_mean(self):

        center = StarCenteringModule(name_in="center2",
                                     image_in_tag="shift",
                                     image_out_tag="center",
                                     mask_out_tag=None,
                                     fit_out_tag="center_fit",
                                     method="mean",
                                     interpolation="bilinear",
                                     radius=0.05,
                                     sign="positive",
                                     guess=(6., 4., 1., 1., 1., 0.))

        self.pipeline.add_module(center)
        self.pipeline.run_module("center2")

        data = self.pipeline.get_data("center")
        assert np.allclose(data[0, 39, 39], 0.023556482678860322, rtol=1e-4, atol=0.)
        assert np.allclose(np.mean(data), 0.00016430629447868552, rtol=1e-4, atol=0.)
        assert data.shape == (40, 78, 78)

    def test_waffle_center_odd(self):

        waffle = WaffleCenteringModule(size=2.,
                                       center=(50, 50),
                                       name_in="waffle_odd",
                                       image_in_tag="star_odd",
                                       center_in_tag="waffle_odd",
                                       image_out_tag="center_odd",
                                       radius=42.5,
                                       pattern="x",
                                       sigma=0.135)

        self.pipeline.add_module(waffle)
        self.pipeline.run_module("waffle_odd")

        data = self.pipeline.get_data("star_odd")
        assert np.allclose(data[0, 50, 50], 0.09806026673451182, rtol=limit, atol=0.)

        data = self.pipeline.get_data("center_odd")
        assert np.allclose(data[0, 37, 37], 0.0980602667345118, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00017777777777778643, rtol=limit, atol=0.)
        assert data.shape == (10, 75, 75)

        attribute = self.pipeline.get_attribute("center_odd", "History: Waffle centering")
        assert attribute == "position [x, y] = [50.0, 50.0]"

    def test_waffle_center_even(self):

        waffle = WaffleCenteringModule(size=2.,
                                       center=(50, 50),
                                       name_in="waffle_even",
                                       image_in_tag="star_even",
                                       center_in_tag="waffle_even",
                                       image_out_tag="center_even",
                                       radius=42.5,
                                       pattern="x",
                                       sigma=0.135)

        self.pipeline.add_module(waffle)
        self.pipeline.run_module("waffle_even")

        data = self.pipeline.get_data("star_even")
        assert np.allclose(data[0, 49, 49], 0.08406157361512759, rtol=limit, atol=0.)
        assert np.allclose(data[0, 49, 50], 0.08406157361512759, rtol=limit, atol=0.)
        assert np.allclose(data[0, 50, 49], 0.08406157361512759, rtol=limit, atol=0.)
        assert np.allclose(data[0, 50, 50], 0.08406157361512759, rtol=limit, atol=0.)

        data = self.pipeline.get_data("center_even")
        assert np.allclose(data[0, 37, 37], 0.09778822940550569, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00017777777777778643, rtol=limit, atol=0.)
        assert data.shape == (10, 75, 75)

        attribute = self.pipeline.get_attribute("center_even", "History: Waffle centering")
        assert attribute == "position [x, y] = [49.5, 49.5]"
