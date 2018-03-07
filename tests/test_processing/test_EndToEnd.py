import os
import math
import warnings

import numpy as np

from astropy.io import fits
from scipy.ndimage import shift

from PynPoint import Pypeline
from PynPoint.Core.DataIO import DataStorage, InputPort
from PynPoint.IOmodules import FitsReadingModule
from PynPoint.ProcessingModules.FrameSelection import RemoveLastFrameModule, RemoveFramesModule
from PynPoint.ProcessingModules.PSFSubtractionPCA import PSFSubtractionModule
from PynPoint.ProcessingModules.PSFpreparation import AngleCalculationModule
from PynPoint.ProcessingModules.ImageResizing import RemoveLinesModule
from PynPoint.ProcessingModules.BackgroundSubtraction import MeanBackgroundSubtractionModule
from PynPoint.ProcessingModules.BadPixelCleaning import BadPixelSigmaFilterModule
from PynPoint.ProcessingModules.StarAlignment import StarExtractionModule, StarAlignmentModule
from PynPoint.ProcessingModules.StackingAndSubsampling import StackAndSubsetModule

warnings.simplefilter("always")

limit = 1e-10

def setup_module():
    test_dir = os.path.dirname(__file__) + "/"

    fwhm =  [ 3, 3, 3, 3 ]
    exp_no =  [ 1, 2, 3, 4 ]
    npix = [ 100, 100, 100, 100 ]
    ndit = [ 22, 17, 21, 18 ]
    naxis3 = [ 23, 18, 22, 19 ]
    x0 = [ 25, 75, 75, 25 ]
    y0 = [ 75, 75, 25, 25 ]
    parang_start = [ 0., 25., 50., 75. ]
    parang_end = [ 25., 50., 75., 100. ]
    sep = 10
    contrast = 1e-2

    parang = []
    for i in range(len(parang_start)):
        for j in range(ndit[i]):
            parang.append(parang_start[i]+float(j)*(parang_end[i]-parang_start[i])/float(ndit[i]))

    np.random.seed(1)

    p_count = 0
    for j in range(len(fwhm)):

        sigma = fwhm[j] / ( 2. * math.sqrt(2.*math.log(2.)) )

        x = np.arange(0., npix[j], 1.)
        y = np.arange(0., npix[j]+2, 1.)
        xx, yy = np.meshgrid(x,y)
    
        image = np.zeros((naxis3[j], npix[j]+2, npix[j]))

        for i in range(ndit[j]):
            star = (1./(2.*np.pi*sigma**2)) * np.exp( -((xx-x0[j])**2 + (yy-y0[j])**2) / (2.*sigma**2) )
            noise = np.random.normal(loc=0, scale=2e-4, size=(102, 100))

            planet = contrast*(1./(2.*np.pi*sigma**2)) * np.exp( -((xx-x0[j])**2 + (yy-y0[j])**2) / (2.*sigma**2) )
            x_shift = sep*math.cos(parang[p_count]*math.pi/180.)
            y_shift = sep*math.sin(parang[p_count]*math.pi/180.)
            planet = shift(planet, (x_shift, y_shift), order=5)

            image[i, 0:npix[j]+2, 0:npix[j]] = star+noise+planet
        
            p_count += 1

        hdu = fits.PrimaryHDU()
        header = hdu.header
        header['INSTRUME'] = 'IMAGER'
        header['HIERARCH ESO DET EXP NO'] = exp_no[j]
        header['HIERARCH ESO DET NDIT'] = ndit[j]
        header['HIERARCH ESO ADA POSANG'] = parang_start[j]
        header['HIERARCH ESO ADA POSANG END'] = parang_end[j]
        header['HIERARCH ESO SEQ CUMOFFSETX'] = 5.
        header['HIERARCH ESO SEQ CUMOFFSETY'] = 5.
        hdu.data = image
        hdu.writeto(test_dir+'adi'+str(j+1).zfill(2)+'.fits')

    config_file = os.path.dirname(__file__) + "/PynPoint_config.ini"

    f = open(config_file, 'w')
    f.write('[header]\n\n')
    f.write('INSTRUMENT: INSTRUME\n')
    f.write('NFRAMES: NAXIS3\n')
    f.write('EXP_NO: ESO DET EXP NO\n')
    f.write('NDIT: ESO DET NDIT\n')
    f.write('PARANG_START: ESO ADA POSANG\n')
    f.write('PARANG_END: ESO ADA POSANG END\n')
    f.write('DITHER_X: ESO SEQ CUMOFFSETX\n')
    f.write('DITHER_Y: ESO SEQ CUMOFFSETY\n\n')
    f.write('[settings]\n\n')
    f.write('PIXSCALE: 0.027\n')
    f.write('MEMORY: 100\n')
    f.write('CPU: 1')
    f.close()

def teardown_module():
    test_dir = os.path.dirname(__file__) + "/"

    for i in range(4):
        file_in = test_dir + 'adi'+str(i+1).zfill(2)+'.fits'
        os.remove(file_in)

    os.remove(test_dir + 'PynPoint_database.hdf5')
    os.remove(test_dir + 'PynPoint_config.ini')

class TestEndToEnd(object):

    def setup(self):
        self.test_dir = os.path.dirname(__file__) + "/"
        self.pipeline = Pypeline(self.test_dir, self.test_dir, self.test_dir)

    def test_read(self):
        read_fits = FitsReadingModule(name_in="read_fits",
                                      image_tag="im",
                                      overwrite=True)

        self.pipeline.add_module(read_fits)
        self.pipeline.run_module("read_fits")

        storage = DataStorage(self.test_dir+"PynPoint_database.hdf5")
        storage.open_connection()
        data = storage.m_data_bank["im"]

        assert np.allclose(data[0, 0, 0], 0.00032486907273264834, rtol=limit)
        assert np.allclose(np.mean(data), 9.4518306864680034e-05, rtol=limit)
        assert data.shape == (82, 102, 100)

        storage.close_connection()

    def test_remove_last(self):
        remove_last = RemoveLastFrameModule(name_in="remove_last",
                                            image_in_tag="im",
                                            image_out_tag="im_last")

        self.pipeline.add_module(remove_last)
        self.pipeline.run_module("remove_last")

        storage = DataStorage(self.test_dir+"PynPoint_database.hdf5")
        storage.open_connection()
        data = storage.m_data_bank["im_last"]

        assert np.allclose(data[0, 0, 0], 0.00032486907273264834, rtol=limit)
        assert np.allclose(np.mean(data), 9.9365399524407205e-05, rtol=limit)
        assert data.shape == (78, 102, 100)

        storage.close_connection()

    def test_parang(self):
        angle = AngleCalculationModule(name_in="angle",
                                       data_tag="im_last")

        self.pipeline.add_module(angle)
        self.pipeline.run_module("angle")

        storage = DataStorage(self.test_dir+"PynPoint_database.hdf5")
        storage.open_connection()
        port = InputPort("im_last", storage)

        assert port.get_attribute("Used_Files")[0] == self.test_dir+'adi01.fits'
        assert port.get_attribute("NEW_PARA")[1] == 1.1904761904761905

        port.close_database()
        storage.close_connection()

    def test_cut_lines(self):
        cut_lines = RemoveLinesModule(lines=(0, 0, 0, 2),
                                      name_in="cut_lines",
                                      image_in_tag="im_last",
                                      image_out_tag="im_cut")

        self.pipeline.add_module(cut_lines)
        self.pipeline.run_module("cut_lines")

        storage = DataStorage(self.test_dir+"PynPoint_database.hdf5")
        storage.open_connection()
        data = storage.m_data_bank["im_cut"]

        assert np.allclose(data[0, 0, 0], 0.00032486907273264834, rtol=limit)
        assert np.allclose(np.mean(data), 0.00010141595132969683, rtol=limit)
        assert data.shape == (78, 100, 100)

        storage.close_connection()

    def test_background(self):
        background = MeanBackgroundSubtractionModule(shift=None,
                                                     cubes=1,
                                                     name_in="background",
                                                     image_in_tag="im_cut",
                                                     image_out_tag="im_bg")

        self.pipeline.add_module(background)
        self.pipeline.run_module("background")

        storage = DataStorage(self.test_dir+"PynPoint_database.hdf5")
        storage.open_connection()
        data = storage.m_data_bank["im_bg"]

        assert np.allclose(data[0, 0, 0], 0.00037132392435389595, rtol=limit)
        assert np.allclose(np.mean(data), 2.3675404363850964e-07, rtol=limit)
        assert data.shape == (78, 100, 100)

        storage.close_connection()

    def test_bad_pixel(self):
        bad_pixel = BadPixelSigmaFilterModule(name_in="bad_pixel",
                                              image_in_tag="im_bg",
                                              image_out_tag="im_bp",
                                              box=9,
                                              sigma=8,
                                              iterate=3)

        self.pipeline.add_module(bad_pixel)
        self.pipeline.run_module("bad_pixel")

        storage = DataStorage(self.test_dir+"PynPoint_database.hdf5")
        storage.open_connection()
        data = storage.m_data_bank["im_bp"]

        assert np.allclose(data[0, 0, 0], 0.00037132392435389595, rtol=limit)
        assert np.allclose(np.mean(data), 2.3675404363850964e-07, rtol=limit)
        assert data.shape == (78, 100, 100)

        storage.close_connection()

    def test_star(self):
        star = StarExtractionModule(name_in="star",
                                    image_in_tag="im_bp",
                                    image_out_tag="im_star",
                                    image_size=1.08,
                                    fwhm_star=0.0108)

        self.pipeline.add_module(star)
        self.pipeline.run_module("star")

        storage = DataStorage(self.test_dir+"PynPoint_database.hdf5")
        storage.open_connection()
        data = storage.m_data_bank["im_star"]

        assert np.allclose(data[0, 0, 0], 0.00018025424208141221, rtol=limit)
        assert np.allclose(np.mean(data), 0.00063151691905138636, rtol=limit)
        assert data.shape == (78, 40, 40)

        storage.close_connection()

    def test_center(self):
        center = StarAlignmentModule(name_in="center",
                                     image_in_tag="im_star",
                                     ref_image_in_tag=None,
                                     image_out_tag="im_center",
                                     interpolation="spline",
                                     accuracy=10,
                                     resize=5)

        self.pipeline.add_module(center)
        self.pipeline.run_module("center")

        storage = DataStorage(self.test_dir+"PynPoint_database.hdf5")
        storage.open_connection()
        data = storage.m_data_bank["im_center"]

        assert np.allclose(data[1, 0, 0], 1.2113798549047296e-06, rtol=limit)
        assert np.allclose(data[16, 0, 0], 1.0022456564129139e-05, rtol=limit)
        assert np.allclose(data[50, 0, 0], 1.7024977291686637e-06, rtol=limit)
        assert np.allclose(data[67, 0, 0], 7.8143774182171561e-07, rtol=limit)
        assert np.allclose(np.mean(data), 2.5260676762055473e-05, rtol=limit)
        assert data.shape == (78, 200, 200)

        storage.close_connection()

    def test_remove_frames(self):
        remove_frames = RemoveFramesModule((0, 15, 49, 66),
                                           name_in="remove_frames",
                                           image_in_tag="im_center",
                                           image_out_tag="im_remove")

        self.pipeline.add_module(remove_frames)
        self.pipeline.run_module("remove_frames")

        storage = DataStorage(self.test_dir+"PynPoint_database.hdf5")
        storage.open_connection()
        data = storage.m_data_bank["im_remove"]

        assert np.allclose(data[0, 0, 0], 1.2113798549047296e-06, rtol=limit)
        assert np.allclose(data[14, 0, 0], 1.0022456564129139e-05, rtol=limit)
        assert np.allclose(data[47, 0, 0], 1.7024977291686637e-06, rtol=limit)
        assert np.allclose(data[63, 0, 0], 7.8143774182171561e-07, rtol=limit)
        assert np.allclose(np.mean(data), 2.5255308248050269e-05, rtol=limit)
        assert data.shape == (74, 200, 200)

        storage.close_connection()

    def test_subset(self):
        subset = StackAndSubsetModule(name_in="subset",
                                      image_in_tag="im_remove",
                                      image_out_tag="im_subset",
                                      random_subset=37,
                                      stacking=2)

        self.pipeline.add_module(subset)
        self.pipeline.run_module("subset")

        storage = DataStorage(self.test_dir+"PynPoint_database.hdf5")
        storage.open_connection()
        data = storage.m_data_bank["im_subset"]

        assert np.allclose(data[0, 0, 0], -1.9081971570461925e-06, rtol=limit)
        assert np.allclose(np.mean(data), 2.5255308248050275e-05, rtol=limit)
        assert data.shape == (37, 200, 200)

        storage.close_connection()

    def test_pca(self):
        pca = PSFSubtractionModule(name_in="pca",
                                   pca_number=2,
                                   images_in_tag="im_subset",
                                   reference_in_tag="im_subset",
                                   res_arr_out_tag="res_arr",
                                   res_arr_rot_out_tag="res_rot",
                                   res_mean_tag="res_mean",
                                   res_median_tag="res_median",
                                   res_var_tag="res_var",
                                   res_rot_mean_clip_tag="res_rot_mean_clip",
                                   extra_rot=0.0,
                                   cent_size=0.1)

        self.pipeline.add_module(pca)
        self.pipeline.run_module("pca")

        storage = DataStorage(self.test_dir+"PynPoint_database.hdf5")
        storage.open_connection()
        data = storage.m_data_bank["res_mean"]

        assert np.allclose(data[154, 99], 0.00043144351678910169, rtol=limit)
        assert np.allclose(np.mean(data), -1.9270808587607946e-09, rtol=limit)
        assert data.shape == (200, 200)

        storage.close_connection()
