import os
import warnings

import numpy as np

from PynPoint.Core.Pypeline import Pypeline
from PynPoint.IOmodules.FitsReading import FitsReadingModule
from PynPoint.ProcessingModules.StarAlignment import StarExtractionModule, \
                                                     StarAlignmentModule, \
                                                     ShiftImagesModule, \
                                                     StarCenteringModule, \
                                                     WaffleCenteringModule
from PynPoint.Util.TestTools import create_config, create_star_data, create_waffle_data

warnings.simplefilter("always")

limit = 1e-10

class TestStarAlignment(object):

    def setup_class(self):

        self.test_dir = os.path.dirname(__file__) + "/"

        os.makedirs(self.test_dir+"dither")
        os.makedirs(self.test_dir+"star")
        os.makedirs(self.test_dir+"waffle")

        create_star_data(path=self.test_dir+"dither",
                         npix_x=100,
                         npix_y=100,
                         x0=[25, 75, 75, 25],
                         y0=[75, 75, 25, 25],
                         parang_start=[0., 25., 50., 75.],
                         parang_end=[25., 50., 75., 100.],
                         exp_no=[1, 2, 3, 4])

        create_star_data(path=self.test_dir+"star",
                         npix_x=100,
                         npix_y=100,
                         x0=[50],
                         y0=[50],
                         parang_start=[0.],
                         parang_end=[25.],
                         exp_no=[1])

        create_waffle_data(path=self.test_dir+"waffle")

        create_config(self.test_dir+"PynPoint_config.ini")

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self):

        for i in range(4):
            os.remove(self.test_dir+'dither/image'+str(i+1).zfill(2)+'.fits')
        os.remove(self.test_dir+'star/image01.fits')
        os.remove(self.test_dir+'waffle/image01.fits')

        os.remove(self.test_dir+'PynPoint_database.hdf5')
        os.remove(self.test_dir+'PynPoint_config.ini')

        os.rmdir(self.test_dir+"dither")
        os.rmdir(self.test_dir+"star")
        os.rmdir(self.test_dir+"waffle")

    def test_read_data(self):

        read = FitsReadingModule(name_in="read1",
                                 image_tag="dither",
                                 input_dir=self.test_dir+"dither",
                                 overwrite=True,
                                 check=True)

        self.pipeline.add_module(read)

        read = FitsReadingModule(name_in="read2",
                                 image_tag="waffle",
                                 input_dir=self.test_dir+"waffle",
                                 overwrite=True,
                                 check=True)

        self.pipeline.add_module(read)

        read = FitsReadingModule(name_in="read3",
                                 image_tag="star",
                                 input_dir=self.test_dir+"star",
                                 overwrite=True,
                                 check=True)

        self.pipeline.add_module(read)

        self.pipeline.run_module("read1")
        self.pipeline.run_module("read2")
        self.pipeline.run_module("read3")

        data = self.pipeline.get_data("dither")
        assert np.allclose(data[0, 75, 25], 0.09812948027289994, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010029494781738066, rtol=limit, atol=0.)
        assert data.shape == (40, 100, 100)

        data = self.pipeline.get_data("star")
        assert np.allclose(data[0, 50, 50], 0.09798413502193704, rtol=1e-4, atol=0.)
        assert np.allclose(np.mean(data), 0.00010105060569794142, rtol=1e-4, atol=0.)
        assert data.shape == (10, 100, 100)

        data = self.pipeline.get_data("waffle")
        assert np.allclose(data[0, 20, 20], 0.0976950339382612, rtol=1e-4, atol=0.)
        assert np.allclose(data[0, 20, 80], 0.09801520849082404, rtol=1e-4, atol=0.)
        assert np.allclose(data[0, 80, 20], 0.09805336380096273, rtol=1e-4, atol=0.)
        assert np.allclose(data[0, 80, 80], 0.09787928233000306, rtol=1e-4, atol=0.)
        assert np.allclose(np.mean(data), 0.00039799422656987266, rtol=1e-4, atol=0.)
        assert data.shape == (1, 100, 100)

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

    def test_shift_images(self):

        shift = ShiftImagesModule(shift_xy=(6., 4.),
                                  interpolation="spline",
                                  name_in="shift",
                                  image_in_tag="align",
                                  image_out_tag="shift")

        self.pipeline.add_module(shift)

        self.pipeline.run_module("shift")

        data = self.pipeline.get_data("shift")
        assert np.allclose(data[0, 43, 45], 0.023556628129942764, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00016430682224782259, rtol=limit, atol=0.)
        assert data.shape == (40, 78, 78)

    def test_star_center(self):

        center = StarCenteringModule(name_in="center",
                                     image_in_tag="shift",
                                     image_out_tag="center1",
                                     mask_out_tag="mask",
                                     fit_out_tag="center_fit",
                                     method="full",
                                     interpolation="spline",
                                     radius=0.05,
                                     sign="positive",
                                     guess=(6., 4., 1., 1., 1., 0.))

        self.pipeline.add_module(center)

        self.pipeline.run_module("center")

        data = self.pipeline.get_data("center1")
        assert np.allclose(data[0, 39, 39], 0.023563039729627436, rtol=1e-4, atol=0.)
        assert np.allclose(np.mean(data), 0.00016430629447868552, rtol=1e-4, atol=0.)
        assert data.shape == (40, 78, 78)

        data = self.pipeline.get_data("mask")
        assert np.allclose(data[0, 43, 45], 0.023556628129942764, rtol=limit, atol=0.)
        assert np.allclose(data[0, 43, 55], 0.0, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010827527282995304, rtol=limit, atol=0.)
        assert data.shape == (40, 78, 78)

    def test_waffle_center(self):

        waffle = WaffleCenteringModule(size=2.,
                                       center=(50, 50),
                                       name_in="waffle",
                                       image_in_tag="star",
                                       center_in_tag="waffle",
                                       image_out_tag="center2",
                                       radius=42.5,
                                       pattern="x",
                                       sigma=5.)

        self.pipeline.add_module(waffle)

        self.pipeline.run_module("waffle")

        data = self.pipeline.get_data("waffle_add")
        assert np.allclose(data[0, 50, 50], 0.09798413502193704, rtol=1e-4, atol=0.)
        assert np.allclose(np.mean(data), 9.905950955586846e-05, rtol=1e-4, atol=0.)
        assert data.shape == (10, 101, 101)

        data = self.pipeline.get_data("waffle_shift")
        assert np.allclose(data[0, 49, 49], 0.09798421722910174, rtol=1e-4, atol=0.)
        assert np.allclose(np.mean(data), 9.896729989589664e-05, rtol=1e-4, atol=0.)
        assert data.shape == (10, 101, 101)

        data = self.pipeline.get_data("center2")
        assert np.allclose(data[0, 36, 36], 0.09798421722910174, rtol=1e-4, atol=0.)
        assert np.allclose(np.mean(data), 0.00017948513276825522, rtol=1e-4, atol=0.)
        assert data.shape == (10, 75, 75)
