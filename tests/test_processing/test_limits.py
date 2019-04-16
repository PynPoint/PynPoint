import os
import warnings

import h5py
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.fitsreading import FitsReadingModule
from pynpoint.processing.limits import ContrastCurveModule
from pynpoint.processing.psfpreparation import AngleInterpolationModule
from pynpoint.util.tests import create_config, create_star_data, remove_test_data

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

        data = self.pipeline.get_attribute("read", "PARANG", static=False)
        assert data[5] == 2.7777777777777777
        assert np.allclose(np.mean(data), 10.0, rtol=limit, atol=0.)
        assert data.shape == (40, )

    def test_contrast_curve(self):

        proc = ["single", "multi"]

        for item in proc:

            if item == "multi":
                database = h5py.File(self.test_dir+'PynPoint_database.hdf5', 'a')
                database['config'].attrs['CPU'] = 4

            contrast = ContrastCurveModule(name_in="contrast_"+item,
                                           image_in_tag="read",
                                           psf_in_tag="read",
                                           contrast_out_tag="limits_"+item,
                                           separation=(0.5, 0.6, 0.1),
                                           angle=(0., 360., 180.),
                                           threshold=("sigma", 5.),
                                           psf_scaling=1.,
                                           aperture=0.1,
                                           pca_number=15,
                                           cent_size=None,
                                           edge_size=None,
                                           extra_rot=0.)

            self.pipeline.add_module(contrast)
            self.pipeline.run_module("contrast_"+item)

            data = self.pipeline.get_data("limits_"+item)
            assert np.allclose(data[0, 0], 5.00000000e-01, rtol=limit, atol=0.)
            assert np.allclose(data[0, 1], 2.3624384190310397, rtol=limit, atol=0.)
            assert np.allclose(data[0, 2], 0.05234065236317515, rtol=limit, atol=0.)
            assert np.allclose(data[0, 3], 0.00012147700290954244, rtol=limit, atol=0.)
            assert data.shape == (1, 4)
