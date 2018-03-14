import os
import warnings

import numpy as np

from astropy.io import fits

import PynPoint as PynPoint

warnings.simplefilter("always")

limit = 1e-10

def setup_module():
    image1 = np.loadtxt(os.path.dirname(__file__) + "/test_data/image1.dat")
    image2 = np.loadtxt(os.path.dirname(__file__) + "/test_data/image2.dat")
    image3 = np.loadtxt(os.path.dirname(__file__) + "/test_data/image3.dat")
    image4 = np.loadtxt(os.path.dirname(__file__) + "/test_data/image4.dat")

    hdu = fits.PrimaryHDU()
    header = hdu.header
    header['INSTRUME'] = "IMAGER"
    header['HIERARCH ESO DET EXP NO'] = 1
    header['HIERARCH ESO DET NDIT'] = 1
    header['HIERARCH ESO ADA POSANG'] = 1.
    header['HIERARCH ESO ADA POSANG END'] = 1.
    header['PARANG'] = -17.3261
    header['PARANG'] = -17.3261
    hdu.data = image1
    hdu.writeto(os.path.dirname(__file__) + "/test_data/image1.fits")

    hdu = fits.PrimaryHDU()
    header = hdu.header
    header['INSTRUME'] = "IMAGER"
    header['HIERARCH ESO DET EXP NO'] = 2
    header['HIERARCH ESO DET NDIT'] = 1
    header['HIERARCH ESO ADA POSANG'] = 1.
    header['HIERARCH ESO ADA POSANG END'] = 1.
    header['PARANG'] = -17.1720
    hdu.data = image2
    hdu.writeto(os.path.dirname(__file__) + "/test_data/image2.fits")

    hdu = fits.PrimaryHDU()
    header = hdu.header
    header['INSTRUME'] = "IMAGER"
    header['HIERARCH ESO DET EXP NO'] = 3
    header['HIERARCH ESO DET NDIT'] = 1
    header['HIERARCH ESO ADA POSANG'] = 1.
    header['HIERARCH ESO ADA POSANG END'] = 1.
    header['PARANG'] = -17.0143
    hdu.data = image3
    hdu.writeto(os.path.dirname(__file__) + "/test_data/image3.fits")

    hdu = fits.PrimaryHDU()
    header = hdu.header
    header['INSTRUME'] = "IMAGER"
    header['HIERARCH ESO DET EXP NO'] = 4
    header['HIERARCH ESO DET NDIT'] = 1
    header['HIERARCH ESO ADA POSANG'] = 1.
    header['HIERARCH ESO ADA POSANG END'] = 1.
    header['PARANG'] = -16.6004
    hdu.data = image4
    hdu.writeto(os.path.dirname(__file__) + "/test_data/image4.fits")

    config_file = os.path.dirname(__file__) + "/test_data/PynPoint_config.ini"

    f = open(config_file, 'w')
    f.write('[header]\n\n')
    f.write('INSTRUMENT: INSTRUME\n')
    f.write('NFRAMES: NAXIS3\n')
    f.write('EXP_NO: ESO DET EXP NO\n')
    f.write('NDIT: ESO DET NDIT\n')
    f.write('PARANG_START: ESO ADA POSANG\n')
    f.write('PARANG_END: ESO ADA POSANG END\n')
    f.write('DITHER_X: None\n')
    f.write('DITHER_Y: None\n\n')
    f.write('[settings]\n\n')
    f.write('PIXSCALE: 0.01\n')
    f.write('MEMORY: 100\n')
    f.write('CPU: 1')
    f.close()

def teardown_module():
    os.remove(os.path.dirname(__file__) + "/test_data/image1.fits")
    os.remove(os.path.dirname(__file__) + "/test_data/image2.fits")
    os.remove(os.path.dirname(__file__) + "/test_data/image3.fits")
    os.remove(os.path.dirname(__file__) + "/test_data/image4.fits")
    os.remove(os.path.dirname(__file__) + "/test_data/PynPoint_database.hdf5")
    os.remove(os.path.dirname(__file__) + "/test_data/PynPoint_config.ini")

class TestResidual(object):

    def setup(self):
        self.test_data_dir = os.path.dirname(__file__) + '/test_data/'
        self.basis = PynPoint.basis.create_wdir(self.test_data_dir, resize=None, ran_sub=False, cent_size=0.2)
        self.images = PynPoint.images.create_wdir(self.test_data_dir, resize=None, ran_sub=False, cent_size=0.2)
        self.res = PynPoint.residuals.create_winstances(self.images, self.basis)
        self.num_files = self.images.im_arr.shape[0]

    def test_res_rot_mean(self):
        assert self.res.res_arr(1).shape == (self.num_files, 146, 146)
        assert np.allclose(self.res.res_arr(1).mean(), 9.196085231064891e-21, rtol=limit, atol=0.)
        assert np.allclose(self.res.res_arr(1).var(), 9.017632410173216e-08, rtol=limit, atol=0.)
        assert self.res.res_rot(1).shape == (self.num_files, 146, 146)
        assert np.allclose(self.res.res_rot(1).mean(), -2.3283496917978495e-08, rtol=limit, atol=0.)
        assert self.res.res_rot_mean(1).shape == (146, 146)
        assert np.allclose(self.res.res_rot_mean(1).mean(), -2.3283496917978114e-08, rtol=limit, atol=0.)
        assert self.res.res_rot_mean_clip(1).shape == (146, 146)
        assert self.res.res_rot_var(1).shape == (146, 146)
        assert np.allclose(self.res.res_rot_var(1).mean(), 3.2090995369569343e-07, rtol=limit, atol=0.)
        assert self.res._psf_im(1).shape == (self.num_files, 146, 146)
        assert np.allclose(self.res._psf_im(1).mean(), 0.0004695346933652781, rtol=limit, atol=0.)

    def test_residuals_save_restore(self):
        temp_file = os.path.join(self.test_data_dir, 'tmp_res.hdf5')
        self.res.save(temp_file)
        temp_res = PynPoint.residuals.create_restore(temp_file)
        assert np.array_equal(self.res.res_rot_mean(1), temp_res.res_rot_mean(1))
        os.remove(temp_file)
