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
from PynPoint.Util.TestTools import create_config

warnings.simplefilter("always")

limit = 1e-10

def setup_module():
    test_dir = os.path.dirname(__file__) + "/"

    fwhm = 3
    npix = 100
    ndit = 10
    naxis3 = ndit
    exp_no = [1, 2, 3, 4]
    x0 = [50, 50, 50, 50]
    y0 = [50, 50, 50, 50]
    parang_start = [0., 5., 10., 15.]
    parang_end = [5., 10., 15., 20.]

    np.random.seed(1)

    for j, item in enumerate(exp_no):
        sigma = fwhm / (2. * math.sqrt(2.*math.log(2.)))

        x = y = np.arange(0., npix, 1.)
        xx, yy = np.meshgrid(x, y)

        image = np.zeros((naxis3, npix, npix))

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
