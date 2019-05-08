import os
import warnings

from urllib.request import urlretrieve

import h5py
import numpy as np

from pynpoint.core.pypeline import Pypeline
from pynpoint.readwrite.fitsreading import FitsReadingModule
from pynpoint.processing.limits import ContrastCurveModule, MassLimitsModule
from pynpoint.processing.psfpreparation import AngleInterpolationModule
from pynpoint.util.tests import create_config, create_star_data, remove_test_data

warnings.simplefilter("always")

limit = 1e-10

class TestDetectionLimits:

    def setup_class(self):

        self.test_dir = os.path.dirname(__file__) + "/"

        create_star_data(path=self.test_dir+"limits")
        create_config(self.test_dir+"PynPoint_config.ini")

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self):

        remove_test_data(path=self.test_dir,
                         folders=["limits"],
                         files=["model.AMES-Cond-2000.M-0.0.NaCo.Vega"])

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
            assert np.allclose(data[0, 1], 1.2034060712266164, rtol=limit, atol=0.)
            assert np.allclose(data[0, 2], 0.2594381985736874, rtol=limit, atol=0.)
            assert np.allclose(data[0, 3], 0.00012147700290954244, rtol=limit, atol=0.)
            assert data.shape == (1, 4)

    def test_mass_limits(self):

        database = h5py.File(self.test_dir+'PynPoint_database.hdf5', 'a')
        database['config'].attrs['CPU'] = 1

        separation = np.linspace(0.1, 1.0, 10)
        contrast = -2.5*np.log10(1e-4/separation)
        variance = 0.1*contrast

        limits = np.zeros((10, 4))
        limits[:, 0] = separation
        limits[:, 1] = contrast
        limits[:, 2] = variance

        database['contrast_limits'] = limits

        url = "https://phoenix.ens-lyon.fr/Grids/AMES-Cond/ISOCHRONES/" \
              "model.AMES-Cond-2000.M-0.0.NaCo.Vega"

        filename = self.test_dir + "model.AMES-Cond-2000.M-0.0.NaCo.Vega"

        urlretrieve(url, filename)

        module = MassLimitsModule(model_file=filename,
                                  star_prop={'magnitude':10., 'distance':100., 'age':20.},
                                  name_in="mass",
                                  contrast_in_tag="contrast_limits",
                                  mass_out_tag="mass_limits",
                                  instr_filter="L\'")

        self.pipeline.add_module(module)
        self.pipeline.run_module("mass")

        data = self.pipeline.get_data("mass_limits")
        assert np.allclose(np.mean(data[:, 0]), 0.55, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data[:, 1]), 0.001891690765603738, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data[:, 2]), 0.000964309686441908, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data[:, 3]), -0.000696402843279597, rtol=limit, atol=0.)
        assert data.shape == (10, 4)
