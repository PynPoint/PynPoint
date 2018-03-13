import os
import warnings

import numpy as np

from astropy.io import fits

import PynPoint as PynPoint
import PynPoint.OldVersion

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

class TestImages(object):

    def setup(self):
        self.test_data_dir = os.path.dirname(__file__) + '/test_data/'

        self.files_fits = [self.test_data_dir+'image1.fits', self.test_data_dir+'image2.fits',
                           self.test_data_dir+'image3.fits', self.test_data_dir+'image4.fits']

        self.images1 = PynPoint.images.create_wdir(self.test_data_dir, resize=-1, ran_sub=None, cent_remove=False)
        self.images2 = PynPoint.images.create_wdir(self.test_data_dir, resize=-1, ran_sub=2, cent_remove=False)
        self.images3 = PynPoint.images.create_wdir(self.test_data_dir, resize=-1, ran_sub=None, cent_size=0.2, cent_remove=True)
        self.images4 = PynPoint.images.create_wdir(self.test_data_dir, resize=2., ran_sub=None, cent_remove=False)
        self.images5 = PynPoint.images.create_wdir(self.test_data_dir, resize=2., ran_sub=None, cent_remove=False)

        PynPoint.OldVersion._Util.filename4mdir(self.test_data_dir)

        self.basis = PynPoint.basis.create_wdir(self.test_data_dir, resize=-1, ran_sub=None, cent_size=0.2, cent_remove=True)
        self.imagesfits = PynPoint.images.create_wfitsfiles(self.files_fits, resize=-1, ran_sub=None, cent_size=0.2, cent_remove=True)

    def test_overall_images3(self):
        assert np.array_equal(self.images3.im_size, (146, 146))
        assert self.images3.cent_size == 0.2
        assert self.images3.im_arr.shape == (4, 146, 146)
        assert np.allclose(self.images3.im_arr.min(), -0.0018710878410451591, rtol=limit)
        assert np.allclose(self.images3.im_arr.max(), 0.0053438268740249742, rtol=limit)
        assert np.allclose(self.images3.im_arr.var(), 3.0953862425934607e-07, rtol=limit)
        assert np.allclose(self.images3.im_norm, np.array([79863.82548531, 82103.89026117, 76156.65271824, 66806.05648646]), rtol=limit)
        assert np.array_equal(self.images3.para, np.array([-17.3261, -17.172, -17.0143, -16.6004]))
        assert self.images3.cent_mask.shape == (146, 146)
        assert self.images3.cent_mask.min() == 0.0
        assert self.images3.cent_mask.max() == 1.0
        assert np.allclose(self.images3.cent_mask.var(), 0.22491619287271775, rtol=limit)
        assert np.array_equal(self.images3.im_arr_mask.shape, (4, 146, 146))
        assert np.allclose(self.images3.im_arr_mask.min(), -0.0020284527946860184, rtol=limit)
        assert np.allclose(self.images3.im_arr_mask.max(), 0.13108563152819708, rtol=limit)
        assert np.allclose(self.images3.im_arr_mask.var(), 4.5119275320392779e-05, rtol=limit)

    def test_overall_images1(self):
        images = self.images1
        images_base = self.images3
        assert np.array_equal(images.files, images_base.files)
        assert np.array_equal(images.im_size, images_base.im_size)
        assert images.im_arr.shape == images_base.im_arr.shape
        assert np.array_equal(images.im_norm, images_base.im_norm)
        assert images.im_arr.shape == (4, 146, 146)
        assert np.allclose(images.im_arr.min(), -0.0020284527946860184, rtol=limit)
        assert np.allclose(images.im_arr.max(), 0.13108563152819708, rtol=limit)
        assert np.allclose(images.im_arr.var(), 4.4735294551387646e-05, rtol=limit)
        # assert images.cent_size == -1.
        assert np.array_equal(images.cent_mask, np.ones(shape=(146, 146)))

    def func4test_overall_same(self, images, images_base):
        assert np.array_equal(images.files, images_base.files)
        assert np.array_equal(images.im_size, images_base.im_size)
        assert np.array_equal(images.im_arr.shape, images_base.im_arr.shape)
        assert np.array_equal(images.im_norm, images_base.im_norm)
        assert np.array_equal(images.im_arr, images_base.im_arr)
        assert np.array_equal(images.cent_mask, images_base.cent_mask)

    def test_images_save_restore(self, tmpdir):
        temp_file = os.path.join(self.test_data_dir, 'tmp_res.hdf5')
        self.images3.save(temp_file)
        temp_images = PynPoint.images.create_restore(temp_file)
        self.func4test_overall_same(self.images3, temp_images)
        os.remove(temp_file)

    def test_mk_psfmodel(self):
        basis = self.basis
        self.images3.mk_psfmodel(basis, 3)
        assert self.images3._psf_coeff.shape == (4, 4)
        assert np.allclose(self.images3._psf_coeff.mean(), -9.1344831261194862e-20, rtol=limit)
        assert np.allclose(self.images3._psf_coeff.min(), -0.045193361270144991, rtol=limit)
        assert np.allclose(self.images3._psf_coeff.max(), 0.0348221927176, rtol=limit)
        assert np.allclose(self.images3._psf_coeff.var(), 0.000506169218776, rtol=limit)
        assert self.images3.psf_im_arr.shape == (4, 146, 146)
        assert np.allclose(self.images3.psf_im_arr.mean(), 0.000293242276843, rtol=limit)
        assert np.allclose(self.images3.psf_im_arr.min(), -0.00187108784105, rtol=limit)
        assert np.allclose(self.images3.psf_im_arr.max(), 0.00534382687402, rtol=limit)
        assert np.allclose(self.images3.psf_im_arr.var(), 3.09538624259e-07, rtol=limit)

    def test_mk_psf_realisation(self):
        basis = self.basis
        self.images3.mk_psfmodel(basis, 3)
        im_temp = self.images3.mk_psf_realisation(1, full=False)
        assert im_temp.shape == (146, 146)
        assert np.allclose(im_temp.mean(), 0.000274016013066, rtol=limit)
        assert np.allclose(im_temp.min(), -0.00159553949981, rtol=limit)
        assert np.allclose(im_temp.max(), 0.00425071210256, rtol=limit)
        assert np.allclose(im_temp.var(), 3.06145293065e-07, rtol=limit)

    def test_mk_psf_realisation2(self):
        basis = self.basis
        self.images3.mk_psfmodel(basis, 3)
        im_temp = self.images3.mk_psf_realisation(1, full=True)
        assert im_temp.shape == (146, 146)
        assert np.allclose(im_temp.mean(), 0.00141957914019, rtol=limit)
        assert np.allclose(im_temp.min(), -0.00159553949981, rtol=limit)
        assert np.allclose(im_temp.max(), 0.127411745859, rtol=limit)
        assert np.allclose(im_temp.var(), 4.48979119722e-05, rtol=limit)

    def test_random_subsample(self):
        images2 = self.images2
        PynPoint.images.create_wdir(self.test_data_dir, resize=-1, ran_sub=3, cent_remove=False)
        assert images2.im_arr.shape == (2, 146, 146)
