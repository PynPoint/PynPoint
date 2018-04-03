import os
import warnings

import h5py
import numpy as np

from astropy.io import fits

import PynPoint as PynPoint
import PynPoint.OldVersion

warnings.simplefilter("always")

limit = 1e-10

def setup_module():
    basis1 = np.loadtxt(os.path.dirname(__file__) + "/test_data/basis1.dat", dtype=np.float64)
    basis2 = np.loadtxt(os.path.dirname(__file__) + "/test_data/basis2.dat", dtype=np.float64)
    basis3 = np.loadtxt(os.path.dirname(__file__) + "/test_data/basis3.dat", dtype=np.float64)
    basis4 = np.loadtxt(os.path.dirname(__file__) + "/test_data/basis4.dat", dtype=np.float64)

    basis = np.stack((basis1, basis2, basis3, basis4), axis=0)

    f = h5py.File(os.path.dirname(__file__) + '/test_data/test_data_PynPoint_conv.hdf5', 'w')
    dset = f.create_dataset("basis_arr", data=basis)
    dset.attrs['PIXSCALE'] = 0.01
    f.create_dataset("header_basis_arr/PARANG", data=np.array([1., 1., 1., 1.]))
    f.close()

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
    f.write('DITHER_Y: None\n')
    f.write('DIT: None\n')
    f.write('LATITUDE: None\n')
    f.write('LONGITUDE: None\n')
    f.write('PUPIL: None\n')
    f.write('DATE: None\n')
    f.write('RA: None\n')
    f.write('DEC: None\n\n')
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
    os.remove(os.path.dirname(__file__) + "/test_data/test_data_PynPoint_conv.hdf5")

class TestBasis(object):

    def setup(self):
        self.test_data_dir = os.path.dirname(__file__) + '/test_data/'

        self.files_fits = [self.test_data_dir+'image1.fits', self.test_data_dir+'image2.fits',
                           self.test_data_dir+'image3.fits', self.test_data_dir+'image4.fits']

        self.basis1 = PynPoint.basis.create_wdir(self.test_data_dir, resize=None, ran_sub=None, cent_size=None)
        self.basis3 = PynPoint.basis.create_wdir(self.test_data_dir, resize=None, ran_sub=None, cent_size=0.2)
        self.basis4 = PynPoint.basis.create_wdir(self.test_data_dir, resize=2., ran_sub=None, cent_size=None)
        self.basisfits = PynPoint.basis.create_wfitsfiles(self.files_fits, resize=None, ran_sub=None, cent_size=0.2)

        hdf5file = PynPoint.OldVersion._Util.filename4mdir(self.test_data_dir)
        self.basis5 = PynPoint.basis.create_whdf5input(hdf5file, resize=None, ran_sub=None, cent_size=None)

        self.eg_array1 = np.arange(100.).reshape(4, 5, 5)
        self.ave_eg_array1 = np.array([[37.5, 38.5, 39.5, 40.5, 41.5], [42.5, 43.5, 44.5, 45.5, 46.5],
                                       [47.5, 48.5, 49.5, 50.5, 51.5], [52.5, 53.5, 54.5, 55.5, 56.5],
                                       [57.5, 58.5, 59.5, 60.5, 61.5]])

        self.eg_array2 = np.array([[0.75, -2.25, -0.25, 1.75], [0.5, -2.5, -0.5, 2.5],
                                   [-1.25, 3.75, 0.75, -3.25], [-2.25, -0.25, 0.75, 1.75]])

        self.eg_array2_pca = np.array([[[-0.17410866485907938728, 0.71893395614514299385],
                                        [0.11771735865682392275, -0.66254264994288725177]],
                                       [[0.84800005432046565712, -0.13400452403625343067],
                                        [-0.29357897148679990007, -0.420416558797411688]],
                                       [[-0.0241263485317723507, 0.46387148461541644062],
                                        [-0.80619725314070234123, 0.36645211705705693639]],
                                       [[0.5, 0.5], [0.5, 0.5]]])

        self.eg_array2_coeff = np.array([[-2.937061877035140e+00, 2.751759847981489e-01, -2.189650836484913e-01],
                                         [-3.599604526978027e+00, -1.452405739992628e-01, 1.474870334085662e-01],
                                         [5.155189797925138e+00, -4.163474455600442e-01, -2.594131731843463e-02],
                                         [-8.591506115107919e-01, -2.830412197722555e+00, -2.504032196304351e-02],
                                         [1.00, 0.0, 0.0], [0.0, 1.00, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])

    def test_basis_save_restore(self, tmpdir):
        temp_file = os.path.join(self.test_data_dir, 'tmp_hdf5.hdf5')
        self.basis3.save(temp_file)
        temp_basis = PynPoint.basis.create_restore(temp_file)
        self.func4test_overall_same(self.basis3, temp_basis)
        os.remove(temp_file)

    def test_overall_basis1(self):
        basis = self.basis1
        basis_base = self.basis3
        assert np.array_equal(basis.files, basis_base.files)
        assert np.array_equal(basis.im_size, basis_base.im_size)
        assert np.array_equal(basis.im_arr.shape, basis_base.im_arr.shape)
        assert np.array_equal(basis.im_norm, basis_base.im_norm)
        assert np.array_equal(basis.im_arr.shape, (4, 146, 146))
        assert np.allclose(basis.im_arr.min(), -0.020154546159792064, rtol=limit, atol=0.)
        assert np.allclose(basis.im_arr.max(), 0.02785169033987621, rtol=limit, atol=0.)
        assert np.allclose(basis.im_arr.var(), 7.246775889090766e-07, rtol=limit, atol=0.)
        assert basis.cent_size == -1.
        assert np.array_equal(basis.cent_mask, np.ones(shape=(146, 146)))
        assert np.array_equal(basis.psf_basis.shape, (4, 146, 146))
        assert np.allclose(basis.psf_basis.var(), 4.691179875591796e-05, rtol=1e-6, atol=0.)

    def test_overall_basis3(self):
        assert np.array_equal(self.basis3.im_size, (146, 146))
        assert self.basis3.cent_size == 0.2
        assert self.basis3.im_arr.shape == (4, 146, 146)
        assert np.allclose(self.basis3.im_arr.min(), -0.00251849574318578, rtol=limit, atol=0.)
        assert np.allclose(self.basis3.im_arr.max(), 0.0023918526633428805, rtol=limit, atol=0.)
        assert np.allclose(self.basis3.im_arr.var(), 1.4193786571723436e-07, rtol=limit, atol=0.)
        # assert np.allclose(self.basis3.im_norm, np.array([79863.82548531, 82103.89026117, 76156.65271824, 66806.05648646]), rtol=limit, atol=0.)
        assert np.array_equal(self.basis3.para, np.array([-17.3261, -17.172, -17.0143, -16.6004]))
        assert self.basis3.cent_mask.shape == (146, 146)
        assert self.basis3.cent_mask.min() == 0.0
        assert self.basis3.cent_mask.max() == 1.0
        assert np.allclose(self.basis3.cent_mask.var(), 0.05578190564690256, rtol=limit, atol=0.)
        assert self.basis3.psf_basis.shape == (4, 146, 146)
        assert np.allclose(self.basis3.psf_basis.var(), 4.6562231691378416e-05, rtol=1e-4, atol=0.)
        assert self.basis3.im_ave.shape == (146, 146)
        assert np.allclose(self.basis3.im_ave.min(), -0.0007636431899760693, rtol=limit, atol=0.)
        assert np.allclose(self.basis3.im_ave.max(), 0.008679750758241345, rtol=limit, atol=0.)
        assert np.allclose(self.basis3.im_ave.var(), 5.323616526937163e-07, rtol=limit, atol=0.)

    def func4test_overall_same(self, basis, basis_base):
        assert np.array_equal(basis.im_size, basis_base.im_size)
        assert np.allclose(basis.im_norm, basis_base.im_norm, rtol=limit, atol=0.)
        assert np.allclose(basis.im_arr, basis_base.im_arr, rtol=limit, atol=0.)
        assert np.allclose(basis.psf_basis[0], basis_base.psf_basis[0], rtol=limit, atol=0.)
        assert np.allclose(basis.psf_basis[1], basis_base.psf_basis[1], rtol=limit, atol=0.)
        assert np.allclose(basis.psf_basis[2], basis_base.psf_basis[2], rtol=limit, atol=0.)
        assert np.allclose(basis.psf_basis[3], basis_base.psf_basis[3], rtol=limit, atol=0.)
        assert np.allclose(basis.cent_mask, basis_base.cent_mask, rtol=limit, atol=0.)
        assert np.allclose(basis.im_ave, basis_base.im_ave, rtol=limit, atol=0.)

    def test_overall_basis5(self):
        self.func4test_overall_same(self.basis5, self.basis1)

    def test_mk_psfmodel(self):
        basis = self.basis3
        basis.mk_psfmodel(20)
        assert np.allclose(basis.psf_im_arr.mean(), 0.00043016667279535496, rtol=1e-4, atol=0.)
