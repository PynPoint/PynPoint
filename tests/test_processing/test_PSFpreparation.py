import os
import warnings

import numpy as np

from PynPoint.Core.Pypeline import Pypeline
from PynPoint.IOmodules.FitsReading import FitsReadingModule
from PynPoint.ProcessingModules.PSFpreparation import PSFpreparationModule, \
                                                      AngleInterpolationModule, \
                                                      SDIpreparationModule
                                                      
from PynPoint.Util.TestTools import create_config, create_star_data

warnings.simplefilter("always")

limit = 1e-10

class TestPSFpreparation(object):

    def setup_class(self):

        self.test_dir = os.path.dirname(__file__) + "/"

        create_star_data(path=os.path.dirname(__file__)+"/",
                         npix_x=100,
                         npix_y=100,
                         x0=[50, 50, 50, 50],
                         y0=[50, 50, 50, 50],
                         parang_start=[0., 5., 10., 15.],
                         parang_end=[5., 10., 15., 20.],
                         exp_no=[1, 2, 3, 4])

        create_config(self.test_dir+"PynPoint_config.ini")

        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def teardown_class(self):

        for i in range(4):
            os.remove(self.test_dir+'image'+str(i+1).zfill(2)+'.fits')

        os.remove(self.test_dir+'PynPoint_database.hdf5')
        os.remove(self.test_dir+'PynPoint_config.ini')

    def test_read_data(self):

        read = FitsReadingModule(name_in="read",
                                 image_tag="read")

        self.pipeline.add_module(read)

        self.pipeline.run_module("read")

        data = self.pipeline.get_data("read")
        assert np.allclose(data[0, 25, 25], 2.0926464668090656e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010029494781738066, rtol=limit, atol=0.)
        assert data.shape == (40, 100, 100)

    def test_angle_interpolation(self):

        angle = AngleInterpolationModule(name_in="angle",
                                         data_tag="read")

        self.pipeline.add_module(angle)

        self.pipeline.run_module("angle")

        data = self.pipeline.get_data("header_read/PARANG")
        assert np.allclose(data[0], 0., rtol=limit, atol=0.)
        assert np.allclose(data[15], 7.777777777777778, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 10.0, rtol=limit, atol=0.)
        assert data.shape == (40, )

    def test_psf_preparation(self):

        prep = PSFpreparationModule(name_in="prep",
                                    image_in_tag="read",
                                    image_out_tag="prep",
                                    mask_out_tag="mask",
                                    norm=True,
                                    resize=2.,
                                    cent_size=0.1,
                                    edge_size=1.0,
                                    verbose=True)

        self.pipeline.add_module(prep)

        self.pipeline.run_module("prep")

        data = self.pipeline.get_data("prep")
        assert np.allclose(data[0, 25, 25], 0., rtol=limit, atol=0.)
        assert np.allclose(data[0, 99, 99], 0., rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.0001818623671899089, rtol=limit, atol=0.)
        assert data.shape == (40, 200, 200)

    def test_sdi_preparation(self):

        sdi = SDIpreparationModule(name_in="sdi",
                                   wavelength=(0.65, 0.6),
                                   width=(0.1, 0.5),
                                   image_in_tag="read",
                                   image_out_tag="sdi")

        self.pipeline.add_module(sdi)

        self.pipeline.run_module("sdi")

        data = self.pipeline.get_data("sdi")
        assert np.allclose(data[0, 25, 25], -2.6648118007008814e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 2.0042892634995876e-05, rtol=limit, atol=0.)
        assert data.shape == (40, 100, 100)
