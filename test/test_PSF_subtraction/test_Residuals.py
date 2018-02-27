import os
import warnings

import numpy as np

import PynPoint as PynPoint

warnings.simplefilter("always")

limit = 1e-10

def setup_module():
    config_file = os.path.dirname(__file__) + "/test_data/PynPoint_config.ini"

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
    f.write('PIXSCALE: 0.01\n')
    f.write('MEMORY: 100\n')
    f.write('CPU: 1')
    f.close()

def teardown_module():
    file_in = os.path.dirname(__file__) + "/test_data/PynPoint_database.hdf5"
    config_file = os.path.dirname(__file__) + "/test_data/PynPoint_config.ini"

    os.remove(file_in)
    os.remove(config_file)

class TestResidual(object):

    def setup(self):
        self.test_data_dir = os.path.dirname(__file__) + '/test_data/'
        self.basis = PynPoint.basis.create_wdir(self.test_data_dir, cent_remove=True, resize=-1, ran_sub=False, cent_size=0.2)
        self.images = PynPoint.images.create_wdir(self.test_data_dir, cent_remove=True, resize=-1, ran_sub=False, cent_size=0.2)
        self.res = PynPoint.residuals.create_winstances(self.images, self.basis)
        self.num_files = self.images.im_arr.shape[0]

    def test_res_rot_mean(self):
        assert self.res.res_arr(1).shape == (self.num_files, 146, 146)
        assert np.allclose(self.res.res_arr(1).mean(), -6.99879052984e-21, rtol=limit)
        assert np.allclose(self.res.res_arr(1).var(), 6.04791954677e-08, rtol=limit)
        assert self.res.res_rot(1).shape == (self.num_files, 146, 146)
        assert np.allclose(self.res.res_rot(1).mean(), -1.98001052939e-09, rtol=limit)
        assert self.res.res_rot_mean(1).shape == (146, 146)
        assert np.allclose(self.res.res_rot_mean(1).mean(), -1.98001052939e-09, rtol=limit)
        assert self.res.res_rot_mean_clip(1).shape == (146, 146)
        assert self.res.res_rot_var(1).shape == (146, 146)
        assert np.allclose(self.res.res_rot_var(1).mean(), 2.39710290476e-07, rtol=limit)
        assert self.res._psf_im(1).shape == (self.num_files, 146, 146)
        assert np.allclose(self.res._psf_im(1).mean(), 0.000293242276843, rtol=limit)

    def test_residuals_save_restore(self):
        temp_file = os.path.join(self.test_data_dir, 'tmp_res_hdf5.h5')
        self.res.save(temp_file)
        temp_res = PynPoint.residuals.create_restore(temp_file)
        assert np.array_equal(self.res.res_rot_mean(1), temp_res.res_rot_mean(1))
        os.remove(temp_file)
