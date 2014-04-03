
# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

"""
Tests for `PynPoint_v1_5` module.
"""
# from __future__ import print_function, division, absolute_import, unicode_literals

import pytest
# import PynPoint_v1_5 as PynPoint
import PynPoint


import sys
import os
import numpy as np

limit0 = 1e-20
limit1 = 1e-10
limit2 = 2e-4


class TestPynpoint_v1_5(object):

    def setup(self):
        #prepare unit test. Load data etc
        print("setting up " + __name__)
        test_data = str(os.path.dirname(__file__)+'/test_data/')
        self.test_data_dir = test_data        
        file_basis_restore = str(self.test_data_dir+'test_data_basis_v001.hdf5'  )      
        file_images_restore = str(self.test_data_dir+'test_data_images_v001.hdf5')
        # print(type(self.test_data_dir+'testfile_basis.hdf5'))
        self.basis = PynPoint.basis.create_wdir(self.test_data_dir,
                                cent_remove=True,resize=False,ran_sub=False,recent=False)

        self.images = PynPoint.images.create_wdir(self.test_data_dir,
                                cent_remove=True,resize=False,ran_sub=False,recent=False)
                                
        self.res = PynPoint.residuals.create_winstances(self.images,self.basis)
        self.num_files = self.images.im_arr.shape[0]
        
               
        #self.basis_restore = PynPoint.basis.create_restore(file_basis_restore)#,intype='restore')
        # self.images_restore = PynPoint.images.create_restore(file_images_restore)#,intype='restore')
        #self.res = PynPoint.residuals(self.images,self.basis,3)

        pass

    def test_res_rot_mean(self):

        assert self.res.res_arr(1).shape == (self.num_files,146,146)
        assert np.allclose(self.res.res_arr(1).mean() , -1.6263342535536864e-22,rtol=limit1)
        assert np.allclose(self.res.res_arr(1).var() , 2.0608041905384247e-09,rtol=limit1)

        assert self.res.res_rot(1).shape == (self.num_files,146,146)
        assert np.allclose(self.res.res_rot(1).mean() ,-4.4349329318721066e-10,rtol=limit1)

        assert self.res.res_rot_mean(1).shape == (146,146)
        assert np.allclose(self.res.res_rot_mean(1).mean() , -4.4349329318527051e-10,rtol=limit1)

        assert self.res.res_rot_mean_clip(1).shape == (146,146)
        # assert np.allclose(self.res.res_rot_mean_clip().mean() , (146,146),rtol=limit1)

        assert self.res.res_rot_var(1).shape == (146,146)
        assert np.allclose(self.res.res_rot_var(1).mean() , -4.4349329318527051e-10,rtol=limit1)

        assert self.res.psf_im(1).shape == (self.num_files,146,146)
        assert np.allclose(self.res.psf_im(1).mean() , 9.2698583022210567e-06,rtol=limit1)

        assert self.res.res_mean_smooth(1,sigma=(2,2)).shape == (146,146)
        assert np.allclose(self.res.res_mean_smooth(1,sigma=(2,2)).mean() , 0.00021770530872593523,rtol=limit1)

        assert self.res.res_mean_clip_smooth(1,sigma=(2,2)).shape == (146,146)
        assert np.allclose(self.res.res_mean_clip_smooth(1,sigma=(2,2)).mean() , 0.00021770530872593523,rtol=limit1)

        assert self.res.res_median_smooth(1,sigma=(2,2)).shape == (146,146)
        assert np.allclose(self.res.res_median_smooth(1,sigma=(2,2)).mean() , -0.0027173103236604501,rtol=limit1)

        # assert self.res.res_clean_mean_clip.min() == -0.066167782249454216
        # assert self.res.res_clean_mean_clip.max() == 0.066889228235009104
#         assert self.res.res_clean_mean_clip.var() == 0.00017278981280217245 

    def test_res_save_restore(self,tmpdir):
        temp_file = str(tmpdir.join('tmp_res_hdf5.h5'))
        
        self.res.save(temp_file)
        temp_res = PynPoint.residuals.create_restore(temp_file) 
        
        assert np.array_equal(self.res.res_rot_mean(1),temp_res.res_rot_mean(1))       
        #self.func4test_overall_same(self.images3,temp_images)

    def test_plt_res(self):
        res = self.res
        im = res.plt_res(3,imtype='median',smooth=(1,1))
        im = res.plt_res(2,imtype='mean',smooth=(2,2))
        im = res.plt_res(1,imtype='mean_clip',smooth=(3,3))
        
        im = res.plt_res(3,imtype='median')
        im = res.plt_res(2,imtype='mean')
        im = res.plt_res(1,imtype='mean_clip')
        im = res.plt_res(2,imtype='var')
        #need to assert somethings
        
        

    def teardown(self):
        #tidy up
        print("tearing down " + __name__)
        pass