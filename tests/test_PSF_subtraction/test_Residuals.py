import os
import warnings

import numpy as np

import PynPoint.OldVersion
from PynPoint.Util.TestTools import prepare_pca_tests, remove_psf_test_data

warnings.simplefilter("always")

limit = 1e-10

def setup_module():
    path = os.path.dirname(__file__)
    prepare_pca_tests(path)

def teardown_module():
    path = os.path.dirname(__file__)
    remove_psf_test_data(path)

class TestResidual(object):

    def setup(self):
        self.test_data_dir = os.path.dirname(__file__) + '/test_data/'
        self.basis = PynPoint.basis.create_wdir(self.test_data_dir, resize=None, ran_sub=None, cent_size=0.2)
        self.images = PynPoint.images.create_wdir(self.test_data_dir, resize=None, ran_sub=None, cent_size=0.2)
        self.res = PynPoint.residuals.create_winstances(self.images, self.basis)
        self.num_files = self.images.im_arr.shape[0]

    def test_res_rot_mean(self):
        assert self.res.res_arr(1).shape == (self.num_files, 146, 146)
        assert np.allclose(self.res.res_arr(1).mean(), 9.19608523106e-21, rtol=0., atol=1e-19)
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
