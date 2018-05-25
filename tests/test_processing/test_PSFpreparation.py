import os
import math
import warnings

import numpy as np

from astropy.io import fits

from PynPoint.Core.Pypeline import Pypeline
from PynPoint.Core.DataIO import DataStorage
from PynPoint.IOmodules.FitsReading import FitsReadingModule
from PynPoint.ProcessingModules.PSFpreparation import PSFpreparationModule, AngleInterpolationModule, \
                                                      SDIpreparationModule
from PynPoint.Util.TestTools import create_config, create_star_data

warnings.simplefilter("always")

limit = 1e-10

def setup_module():
    create_star_data(path=os.path.dirname(__file__)+"/",
                     npix_x=100,
                     npix_y=100,    
                     x0=[50, 50, 50, 50],
                     y0=[50, 50, 50, 50],
                     parang_start=[0., 5., 10., 15.],
                     parang_end=[5., 10., 15., 20.])

    create_config(os.path.dirname(__file__)+"/PynPoint_config.ini")

def teardown_module():
    test_dir = os.path.dirname(__file__) + "/"

    for i in range(4):
        os.remove(test_dir + 'image'+str(i+1).zfill(2)+'.fits')

    os.remove(test_dir + 'PynPoint_database.hdf5')
    os.remove(test_dir + 'PynPoint_config.ini')

class TestPSFpreparation(object):

    def setup(self):
        self.test_dir = os.path.dirname(__file__) + "/"
        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def test_psf_preparation(self):

        read = FitsReadingModule(name_in="read",
                                 image_tag="read")

        self.pipeline.add_module(read)

        angle = AngleInterpolationModule(name_in="angle",
                                         data_tag="read")

        self.pipeline.add_module(angle)

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

        sdi = SDIpreparationModule(name_in="sdi",
                                   wavelength=(0.65, 0.6),
                                   width=(0.1, 0.5),
                                   image_in_tag="read",
                                   image_out_tag="sdi")

        self.pipeline.add_module(sdi)

        self.pipeline.run()

        data = self.pipeline.get_data("read")
        assert np.allclose(data[0, 25, 25], 2.0926464668090656e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010029494781738066, rtol=limit, atol=0.)

        data = self.pipeline.get_data("prep")
        assert np.allclose(data[0, 25, 25], 0., rtol=limit, atol=0.)
        assert np.allclose(data[0, 99, 99], 0., rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.0001818623671899089, rtol=limit, atol=0.)
        assert data.shape == (40, 200, 200)

        data = self.pipeline.get_data("sdi")
        assert np.allclose(data[0, 25, 25], 2.8586863287949902e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 1.719735044879641e-05, rtol=limit, atol=0.)
        assert data.shape == (40, 108, 108)
