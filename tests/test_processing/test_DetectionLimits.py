import os
import math
import warnings

import numpy as np

from astropy.io import fits

from PynPoint.Core.Pypeline import Pypeline
from PynPoint.Core.DataIO import DataStorage
from PynPoint.IOmodules.FitsReading import FitsReadingModule
from PynPoint.ProcessingModules.DetectionLimits import ContrastCurveModule
from PynPoint.ProcessingModules.PSFpreparation import AngleInterpolationModule
from PynPoint.Util.TestTools import create_config, create_star_data

warnings.simplefilter("always")

limit = 1e-10

def setup_module():
    create_star_data(path=os.path.dirname(__file__),
                     npix_x=100,
                     npix_y=100,    
                     x0=[50, 50, 50, 50],
                     y0=[50, 50, 50, 50],
                     parang_start=[0., 5., 10., 15.],
                     parang_end=[5., 10., 15., 20.])

    create_config(os.path.join(os.path.dirname(__file__), "/PynPoint_config.ini"))

def teardown_module():
    test_dir = os.path.dirname(__file__) + "/"

    for i in range(4):
        os.remove(test_dir + 'image'+str(i+1).zfill(2)+'.fits')

    os.remove(test_dir + 'PynPoint_database.hdf5')
    os.remove(test_dir + 'PynPoint_config.ini')

class TestDetectionLimits(object):

    def setup(self):
        self.test_dir = os.path.dirname(__file__) + "/"
        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def test_contrast_curve(self):

        read = FitsReadingModule(name_in="read",
                                 image_tag="read")

        self.pipeline.add_module(read)

        angle = AngleInterpolationModule(name_in="angle",
                                         data_tag="read")

        self.pipeline.add_module(angle)

        contrast = ContrastCurveModule(name_in="contrast",
                                       image_in_tag="read",
                                       psf_in_tag="read",
                                       pca_out_tag="pca",
                                       contrast_out_tag="limits",
                                       separation=(0.5, 0.6, 0.1),
                                       angle=(0., 360., 180.),
                                       magnitude=(7.5, 1.),
                                       sigma=5.,
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

        self.pipeline.run()

        storage = DataStorage(self.test_dir+"/PynPoint_database.hdf5")
        storage.open_connection()

        data = storage.m_data_bank["read"]
        assert np.allclose(data[0, 10, 10], 0.00012958496246258364, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010029494781738066, rtol=limit, atol=0.)

        data = storage.m_data_bank["header_read/PARANG"]
        assert data[5] == 2.7777777777777777

        data = storage.m_data_bank["pca"]
        assert np.allclose(data[9, 68, 49], 5.707647718560735e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), -3.66890878538392e-08, rtol=limit, atol=0.)

        data = storage.m_data_bank["pca"]
        assert np.allclose(data[21, 31, 50], 5.4392925807364694e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), -3.668908785383954e-08, rtol=limit, atol=0.)

        storage.close_connection()
