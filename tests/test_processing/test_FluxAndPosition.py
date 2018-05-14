import os
import math
import warnings

import numpy as np

from astropy.io import fits

from PynPoint.Core.Pypeline import Pypeline
from PynPoint.Core.DataIO import DataStorage
from PynPoint.IOmodules.FitsReading import FitsReadingModule
from PynPoint.ProcessingModules.FluxAndPosition import FakePlanetModule, SimplexMinimizationModule, \
                                                       FalsePositiveModule, AperturePhotometryModule
from PynPoint.ProcessingModules.PSFpreparation import AngleInterpolationModule
from PynPoint.ProcessingModules.PSFSubtractionPCA import PcaPsfSubtractionModule
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

class TestFluxAndPosition(object):

    def setup(self):
        self.test_dir = os.path.dirname(__file__) + "/"
        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def test_fake_planet(self):

        read = FitsReadingModule(name_in="read",
                                 image_tag="read")

        self.pipeline.add_module(read)

        angle = AngleInterpolationModule(name_in="angle",
                                         data_tag="read")

        self.pipeline.add_module(angle)

        fake = FakePlanetModule(position=(0.5, 90.),
                                magnitude=5.,
                                psf_scaling=1.,
                                interpolation="spline",
                                name_in="fake",
                                image_in_tag="read",
                                psf_in_tag="read",
                                image_out_tag="fake",
                                verbose=True)

        self.pipeline.add_module(fake)

        simplex = SimplexMinimizationModule(position=(31., 49.),
                                            magnitude=5.,
                                            psf_scaling=-1.,
                                            name_in="simplex",
                                            image_in_tag="fake",
                                            psf_in_tag="read",
                                            res_out_tag="pca_res",
                                            flux_position_tag="flux_position",
                                            merit="sum",
                                            aperture=0.05,
                                            sigma=0.027,
                                            tolerance=0.1,
                                            pca_number=2,
                                            cent_size=None,
                                            edge_size=None,
                                            extra_rot=0.)

        self.pipeline.add_module(simplex)

        pca = PcaPsfSubtractionModule(pca_numbers=(2, ),
                                      name_in="pca",
                                      images_in_tag="fake",
                                      reference_in_tag="fake",
                                      res_mean_tag="res_mean",
                                      res_median_tag=None,
                                      res_arr_out_tag=None,
                                      res_rot_mean_clip_tag=None,
                                      extra_rot=0.)

        self.pipeline.add_module(pca)

        false = FalsePositiveModule(position=(31., 49.),
                                    aperture=0.1,
                                    ignore=True,
                                    name_in="false",
                                    image_in_tag="res_mean",
                                    snr_out_tag="snr_fpf")

        self.pipeline.add_module(false)

        photometry = AperturePhotometryModule(radius=0.1,
                                              position=None,
                                              name_in="photometry",
                                              image_in_tag="read",
                                              phot_out_tag="photometry")

        self.pipeline.add_module(photometry)

        self.pipeline.run()

        storage = DataStorage(self.test_dir+"/PynPoint_database.hdf5")
        storage.open_connection()

        data = storage.m_data_bank["read"]
        assert np.allclose(data[0, 10, 10], 0.00012958496246258364, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.00010029494781738066, rtol=limit, atol=0.)

        data = storage.m_data_bank["header_read/PARANG"]
        assert data[5] == 2.7777777777777777

        data = storage.m_data_bank["fake"]
        assert np.allclose(data[0, 49, 31], 0.00036532633147006946, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.0001012983225928772, rtol=limit, atol=0.)

        data = storage.m_data_bank["pca_res"]
        assert np.allclose(data[46, 49, 31], 1.8936434205673436e-06, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), -2.8850503470725503e-08, rtol=limit, atol=0.)

        data = storage.m_data_bank["flux_position"]
        assert np.allclose(data[46, 0], 31.378322643989137, rtol=limit, atol=0.)
        assert np.allclose(data[46, 1], 49.94461112812883, rtol=limit, atol=0.)
        assert np.allclose(data[46, 2], 0.5027875127365663, rtol=limit, atol=0.)
        assert np.allclose(data[46, 3], 90.17042177085045, rtol=limit, atol=0.)
        assert np.allclose(data[46, 4], 5.072394778584217, rtol=limit, atol=0.)

        data = storage.m_data_bank["pca_res"]
        assert np.allclose(data[0, 49, 31], -0.00024018362015251087, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), -2.8850503470725503e-08, rtol=limit, atol=0.)

        data = storage.m_data_bank["snr_fpf"]
        assert np.allclose(data[0, 2], 0.513710034941892, rtol=limit, atol=0.)
        assert np.allclose(data[0, 3], 93.01278750418334, rtol=limit, atol=0.)
        assert np.allclose(data[0, 4], 11.775360946367874, rtol=limit, atol=0.)
        assert np.allclose(data[0, 5], 2.9838031156970146e-08, rtol=limit, atol=0.)

        data = storage.m_data_bank["photometry"]
        assert np.allclose(data[0][0], 0.983374353660573, rtol=limit, atol=0.)
        assert np.allclose(data[39][0], 0.9841484973083519, rtol=limit, atol=0.)
        assert np.allclose(np.mean(data), 0.9835085649488583, rtol=limit, atol=0.)

        storage.close_connection()
