import os
import math
import warnings

import numpy as np

from astropy.io import fits

from PynPoint import Pypeline
from PynPoint.Core.DataIO import DataStorage
from PynPoint.IOmodules.FitsReading import FitsReadingModule
from PynPoint.ProcessingModules.StarAlignment import StarExtractionModule, StarAlignmentModule, \
                                                     LocateStarModule, ShiftForCenteringModule, \
                                                     StarCenteringModule

warnings.simplefilter("always")

limit = 1e-10

def setup_module():
    test_dir = os.path.dirname(__file__) + "/"

    fwhm = 3
    npix = 100
    ndit = 10
    naxis3 = ndit
    exp_no = [1, 2, 3, 4]
    x0 = [25, 75, 75, 25]
    y0 = [75, 75, 25, 25]
    parang_start = [0., 25., 50., 75.]
    parang_end = [25., 50., 75., 100.]

    np.random.seed(1)

    for j, item in enumerate(exp_no):
        sigma = fwhm / (2. * math.sqrt(2.*math.log(2.)))

        x = y = np.arange(0., npix, 1.)
        xx, yy = np.meshgrid(x, y)

        image = np.zeros((naxis3, npix+2, npix))

        for i in range(ndit):
            star = (1./(2.*np.pi*sigma**2)) * np.exp(-((xx-x0[j])**2 + (yy-y0[j])**2) / (2.*sigma**2))
            noise = np.random.normal(loc=0, scale=2e-4, size=(npix, npix))
            image[i, 0:npix, 0:npix] = star+noise

        hdu = fits.PrimaryHDU()
        header = hdu.header
        header['INSTRUME'] = 'IMAGER'
        header['HIERARCH ESO DET EXP NO'] = item
        header['HIERARCH ESO DET NDIT'] = ndit
        header['HIERARCH ESO ADA POSANG'] = parang_start[j]
        header['HIERARCH ESO ADA POSANG END'] = parang_end[j]
        header['HIERARCH ESO SEQ CUMOFFSETX'] = "None"
        header['HIERARCH ESO SEQ CUMOFFSETY'] = "None"
        hdu.data = image
        hdu.writeto(test_dir+'image'+str(j+1).zfill(2)+'.fits')

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
                                          fwhm_star=0.1)

        self.pipeline.add_module(extraction)

        align = StarAlignmentModule(name_in="align",
                                    image_in_tag="extract",
                                    ref_image_in_tag=None,
                                    image_out_tag="align",
                                    accuracy=10,
                                    resize=2)

        self.pipeline.add_module(align)

        locate = LocateStarModule(name_in="locate",
                                  data_tag="align",
                                  gaussian_fwhm=0.05)

        self.pipeline.add_module(locate)

        shift = ShiftForCenteringModule((4., 6.),
                                        name_in="shift",
                                        image_in_tag="align",
                                        image_out_tag="shift")

        self.pipeline.add_module(shift)

        center = StarCenteringModule(name_in="center",
                                     image_in_tag="shift",
                                     image_out_tag="center",
                                     fit_out_tag="center_fit",
                                     method="full",
                                     interpolation="spline",
                                     guess=(6., 4., 1., 1., 1., 0.))

        self.pipeline.add_module(center)

        self.pipeline.run()

        storage = DataStorage(self.test_dir+"/PynPoint_database.hdf5")
        storage.open_connection()

        data = storage.m_data_bank["read"]
        assert np.allclose(data[0, 10, 10], 0.00012958496246258364, rtol=limit)
        assert np.allclose(np.mean(data), 9.832838021311831e-05, rtol=limit)

        data = storage.m_data_bank["extract"]
        assert np.allclose(data[0, 10, 10], 0.05304008435511765, rtol=limit)
        assert np.allclose(np.mean(data), 0.0020655767159466613, rtol=limit)

        data = storage.m_data_bank["header_extract/STAR_POSITION"]
        assert data[10, 0] ==  data[10, 1] == 75

        data = storage.m_data_bank["shift"]
        assert np.allclose(data[0, 10, 10], -4.341611534220891e-05, rtol=limit)
        assert np.allclose(np.mean(data), 0.0005164420068450968, rtol=limit)

        data = storage.m_data_bank["center"]
        assert np.allclose(data[0, 10, 10], 4.128859892625027e-05, rtol=limit)
        assert np.allclose(np.mean(data), 0.0005163769620309259, rtol=limit)

        storage.close_connection()
