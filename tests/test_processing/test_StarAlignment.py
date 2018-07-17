import os
import math
import warnings

import numpy as np

from astropy.io import fits

from PynPoint.Core.Pypeline import Pypeline
from PynPoint.Core.DataIO import DataStorage
from PynPoint.IOmodules.FitsReading import FitsReadingModule
from PynPoint.ProcessingModules.StarAlignment import StarExtractionModule, StarAlignmentModule, \
                                                     ShiftImagesModule, StarCenteringModule
from PynPoint.Util.TestTools import create_config, create_star_data

warnings.simplefilter("always")

limit = 1e-10

def setup_module():
    test_dir = os.path.dirname(__file__) + "/"

    create_star_data(path=test_dir,
                     npix_x=100,
                     npix_y=102,
                     x0=[25, 75, 75, 25],
                     y0=[75, 75, 25, 25],
                     parang_start=[0., 25., 50., 75.],
                     parang_end=[25., 50., 75., 100.])

    filename = os.path.dirname(__file__) + "/PynPoint_config.ini"
    create_config(filename)

def teardown_module():
    test_dir = os.path.dirname(__file__) + "/"

    for i in range(4):
        os.remove(test_dir + 'image'+str(i+1).zfill(2)+'.fits')

    os.remove(test_dir + 'PynPoint_database.hdf5')
    os.remove(test_dir + 'PynPoint_config.ini')

class TestStarAlignment(object):

    def setup(self):
        self.test_dir = os.path.dirname(__file__) + "/"
        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def test_star_alignment(self):

        read = FitsReadingModule(name_in="read",
                                 image_tag="read")

        self.pipeline.add_module(read)

        extraction = StarExtractionModule(name_in="extract",
                                          image_in_tag="read",
                                          image_out_tag="extract",
                                          image_size=0.6,
                                          fwhm_star=0.1,
                                          position=None)

        self.pipeline.add_module(extraction)

        align = StarAlignmentModule(name_in="align",
                                    image_in_tag="extract",
                                    ref_image_in_tag=None,
                                    image_out_tag="align",
                                    accuracy=10,
                                    resize=2)

        self.pipeline.add_module(align)

        shift = ShiftImagesModule((6., 4.),
                                  name_in="shift",
                                  image_in_tag="align",
                                  image_out_tag="shift")

        self.pipeline.add_module(shift)

        center = StarCenteringModule(name_in="center",
                                     image_in_tag="shift",
                                     image_out_tag="center",
                                     mask_out_tag=None,
                                     fit_out_tag="center_fit",
                                     method="full",
                                     interpolation="spline",
                                     radius=0.1,
                                     sign="positive",
                                     guess=(6., 4., 1., 1., 1., 0.))

        self.pipeline.add_module(center)

        self.pipeline.run()

        storage = DataStorage(self.test_dir+"/PynPoint_database.hdf5")
        storage.open_connection()

        data = storage.m_data_bank["read"]
        assert np.allclose(data[0, 10, 10], 0.00012958496246258364, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 9.832838021311831e-05, rtol=limit, atol=0.)

        data = storage.m_data_bank["extract"]
        assert np.allclose(data[0, 10, 10], 0.05304008435511765, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.0020655767159466613, rtol=limit, atol=0.)

        data = storage.m_data_bank["header_extract/STAR_POSITION"]
        assert data[10, 0] ==  data[10, 1] == 75

        data = storage.m_data_bank["shift"]
        assert np.allclose(data[0, 10, 10], -4.341611534220891e-05, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.0005164420068450968, rtol=limit, atol=0.)

        data = storage.m_data_bank["center"]
        assert np.allclose(data[0, 10, 10], 4.128859892625027e-05, rtol=1e-4, atol=0.)
        assert np.allclose(np.mean(data), 0.0005163806188663894, rtol=1e-4, atol=0.)

        storage.close_connection()
