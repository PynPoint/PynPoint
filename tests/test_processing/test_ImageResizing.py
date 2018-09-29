import os
import warnings

import numpy as np

from PynPoint.Core.Pypeline import Pypeline
from PynPoint.IOmodules.FitsReading import FitsReadingModule
from PynPoint.ProcessingModules.ImageResizing import CropImagesModule, \
                                                     ScaleImagesModule, \
                                                     AddLinesModule, \
                                                     RemoveLinesModule
from PynPoint.Util.TestTools import create_config, create_star_data, remove_test_data

warnings.simplefilter("always")

limit = 1e-10

class TestImageResizing(object):

    def setup_class(self):

        self.test_dir = os.path.dirname(__file__) + "/"

        create_star_data(path=self.test_dir+"images",
                         npix_x=20,
                         npix_y=20,
                         x0=[10, 10, 10, 10],
                         y0=[10, 10, 10, 10],
                         parang_start=[0., 25., 50., 75.],
                         parang_end=[25., 50., 75., 100.],
                         exp_no=[1, 2, 3, 4])

        create_config(self.test_dir+"PynPoint_config.ini")

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self):

        remove_test_data(self.test_dir, folders=["images"])

    def test_read_data(self):

        read = FitsReadingModule(name_in="read",
                                 image_tag="images",
                                 input_dir=self.test_dir+"images",
                                 overwrite=True,
                                 check=True)

        self.pipeline.add_module(read)

        self.pipeline.run_module("read")

        data = self.pipeline.get_data("images")
        assert np.allclose(data[0, 10, 10], 0.09799496683489618, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.0025020285041348557, rtol=limit, atol=0.)
        assert data.shape == (40, 20, 20)

    def test_crop_images(self):

        crop = CropImagesModule(size=0.3,
                                center=None,
                                name_in="crop1",
                                image_in_tag="images",
                                image_out_tag="crop1")

        self.pipeline.add_module(crop)

        crop = CropImagesModule(size=0.3,
                                center=(6, 6),
                                name_in="crop2",
                                image_in_tag="images",
                                image_out_tag="crop2")

        self.pipeline.add_module(crop)

        self.pipeline.run_module("crop1")
        self.pipeline.run_module("crop2")

        data = self.pipeline.get_data("crop1")
        assert np.allclose(data[0, 7, 7], 0.09799496683489618, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.005921620406208569, rtol=limit, atol=0.)
        assert data.shape == (40, 13, 13)

        data = self.pipeline.get_data("crop2")
        assert np.allclose(data[0, 7, 7], 0.0005067239693313918, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.005664905759498735, rtol=limit, atol=0.)
        assert data.shape == (40, 13, 13)

    def test_scale_images(self):

        scale = ScaleImagesModule(scaling=(2., None),
                                  name_in="scale1",
                                  image_in_tag="images",
                                  image_out_tag="scale1")

        self.pipeline.add_module(scale)

        scale = ScaleImagesModule(scaling=(None, 2.),
                                  name_in="scale2",
                                  image_in_tag="images",
                                  image_out_tag="scale2")

        self.pipeline.add_module(scale)

        self.pipeline.run_module("scale1")
        self.pipeline.run_module("scale2")

        data = self.pipeline.get_data("scale1")
        assert np.allclose(data[0, 20, 20], 0.02362264611391428, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.000625507126033714, rtol=limit, atol=0.)
        assert data.shape == (40, 40, 40)

        data = self.pipeline.get_data("scale2")
        assert np.allclose(data[0, 10, 10], 0.19598993366979467, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.005004057008269711, rtol=limit, atol=0.)
        assert data.shape == (40, 20, 20)

    def test_add_lines(self):

        add = AddLinesModule(lines=(2, 5, 0, 9),
                             name_in="add",
                             image_in_tag="images",
                             image_out_tag="add")

        self.pipeline.add_module(add)
        self.pipeline.run_module("add")

        data = self.pipeline.get_data("add")
        assert np.allclose(data[0, 12, 12], 0.028528539751565975, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.0012781754810395176, rtol=limit, atol=0.)
        assert data.shape == (40, 29, 27)

    def test_remove_lines(self):

        remove = RemoveLinesModule(lines=(2, 5, 0, 9),
                                   name_in="remove",
                                   image_in_tag="images",
                                   image_out_tag="remove")

        self.pipeline.add_module(remove)
        self.pipeline.run_module("remove")

        data = self.pipeline.get_data("remove")
        assert np.allclose(data[0, 10, 10], 0.02882041378895293, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.004595775333521642, rtol=limit, atol=0.)
        assert data.shape == (40, 11, 13)
