import os
import warnings

import numpy as np

from PynPoint.Core.Pypeline import Pypeline
from PynPoint.IOmodules.FitsReading import FitsReadingModule
from PynPoint.ProcessingModules.DetectionLimits import ContrastCurveModule
from PynPoint.ProcessingModules.PSFpreparation import AngleInterpolationModule
from PynPoint.Util.TestTools import create_config, create_star_data, remove_test_data

warnings.simplefilter("always")

limit = 1e-10

class TestDetectionLimits(object):

    def setup_class(self):

        self.test_dir = os.path.dirname(__file__) + "/"

        create_star_data(path=self.test_dir+"limits")
        create_config(self.test_dir+"PynPoint_config.ini")

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self):

        remove_test_data(self.test_dir, folders=["limits"])

    def test_read_data(self):

        read = FitsReadingModule(name_in="read",
                                 image_tag="read",
                                 input_dir=self.test_dir+"limits")

        self.pipeline.add_module(read)
        self.pipeline.run_module("read")

        data = self.pipeline.get_data("read")
        assert np.allclose(data[0, 10, 10], 0.00012958496246258364, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010029494781738066, rtol=limit, atol=0.)
        assert data.shape == (40, 100, 100)

    def test_angle_interpolation(self):

        angle = AngleInterpolationModule(name_in="angle",
                                         data_tag="read")

        self.pipeline.add_module(angle)
        self.pipeline.run_module("angle")

        data = self.pipeline.get_data("header_read/PARANG")
        assert data[5] == 2.7777777777777777
        assert np.allclose(np.mean(data), 10.0, rtol=limit, atol=0.)
        assert data.shape == (40, )

    def test_contrast_curve(self):

        contrast = ContrastCurveModule(name_in="contrast",
                                       image_in_tag="read",
                                       psf_in_tag="read",
                                       pca_out_tag="pca",
                                       contrast_out_tag="limits",
                                       separation=(0.5, 0.6, 0.1),
                                       angle=(0., 360., 180.),
                                       magnitude=(7.5, 1.),
                                       threshold=("sigma", 5.),
                                       accuracy=1e-1,
                                       psf_scaling=1.,
                                       aperture=0.1,
                                       ignore=True,
                                       pca_number=15,
                                       norm=False,
                                       cent_size=None,
                                       edge_size=None,
                                       extra_rot=0.)

        self.pipeline.add_module(contrast)
        self.pipeline.run_module("contrast")

        data = self.pipeline.get_data("pca")
        assert np.allclose(data[9, 68, 49], 5.707647718560735e-05, rtol=limit, atol=0.)
        assert np.allclose(data[21, 31, 50], 5.4392925807364694e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), -3.668908785383954e-08, rtol=limit, atol=0.)
        assert data.shape == (22, 100, 100)

        data = self.pipeline.get_data("limits")
        assert np.allclose(data[0, 0], 5.00000000e-01, rtol=limit, atol=0.)
        assert np.allclose(data[0, 1], 6.557218774128667, rtol=limit, atol=0.)
        assert np.allclose(data[0, 2], 0.12716145540952048, rtol=limit, atol=0.)
        assert np.allclose(data[0, 3], 0.0002012649090622487, rtol=limit, atol=0.)
        assert data.shape == (1, 4)
