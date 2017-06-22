# Copyright (C) 2014 ETH Zurich, Institute for Astronomy
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/.


"""
Tests for `Residual` module.
"""
# from __future__ import print_function, division, absolute_import, unicode_literals

import os
import numpy as np
import PynPoint2 as PynPoint

limit0 = 1e-20
limit1 = 1e-10
limit2 = 2e-4


class TestResidual(object):

    def setup(self):
        #prepare unit test. Load data etc
        print("setting up " + __name__)
        test_data = str(os.path.dirname(__file__)+'/test_data/')
        self.test_data_dir = test_data

        self.basis = PynPoint.basis.create_wdir(self.test_data_dir,
                                cent_remove=True,resize=False,ran_sub=False,recent=False,cent_size=0.2)

        self.images = PynPoint.images.create_wdir(self.test_data_dir,
                                cent_remove=True,resize=False,ran_sub=False,recent=False,cent_size=0.2)
                                
        self.res = PynPoint.residuals.create_winstances(self.images,self.basis)
        self.num_files = self.images.im_arr.shape[0]

    def test_res_rot_mean(self):

        assert self.res.res_arr(1).shape == (self.num_files,146,146)
        print self.res.res_arr(1).mean()
        print self.res.res_arr(1).var()

        print self.res.res_rot(1).mean()

        print self.res.res_rot_mean(1).mean()

        print self.res.res_rot_var(1).mean()

        print self.res._psf_im(1).mean()

        assert np.allclose(self.res.res_arr(1).mean() , -6.99879052984e-21,rtol=limit1)
        assert np.allclose(self.res.res_arr(1).var() , 6.04791954677e-08,rtol=limit1)

        assert self.res.res_rot(1).shape == (self.num_files,146,146)
        assert np.allclose(self.res.res_rot(1).mean() ,-1.98001052939e-09,rtol=limit1)

        assert self.res.res_rot_mean(1).shape == (146,146)
        assert np.allclose(self.res.res_rot_mean(1).mean() , -1.98001052939e-09,rtol=limit1)

        assert self.res.res_rot_mean_clip(1).shape == (146,146)

        assert self.res.res_rot_var(1).shape == (146,146)
        assert np.allclose(self.res.res_rot_var(1).mean() , 2.39710290476e-07,rtol=limit1)

        assert self.res._psf_im(1).shape == (self.num_files,146,146)
        assert np.allclose(self.res._psf_im(1).mean() , 0.000293242276843,rtol=limit1)

    def test_residuals_save_restore(self,tmpdir):
        temp_file = str(tmpdir.join('tmp_res_hdf5.h5'))
        
        self.res.save(temp_file)
        temp_res = PynPoint.residuals.create_restore(temp_file) 
        
        assert np.array_equal(self.res.res_rot_mean(1),temp_res.res_rot_mean(1))

        os.remove(temp_file)

    def teardown(self):
        #tidy up
        if os.path.isfile(self.test_data_dir + "PynPoint_database.hdf5"):
            os.remove(self.test_data_dir + "PynPoint_database.hdf5")