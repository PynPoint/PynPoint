
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
    
#--------------------------        

    # def test_overall_res(self):
    #             
    #     assert self.res.res_rot.shape == (4,146,146)
    #     assert self.res.res_rot.min() == -2.3716922523120405e-19
    #     assert self.res.res_rot.max() == 2.7105054312137616e-19
    #     assert self.res.res_rot.var() == 1.0141351006258307e-39 
    # 
    #     assert self.res.res_rot_var2.shape == (146,146)
    #     assert self.res.res_rot_var2.min() == 0.0
    #     assert self.res.res_rot_var2.max() == 1.3275235419614938e-37
    #     assert self.res.res_rot_var2.var() == 4.8802984473993911e-77 
    #     
    #     assert self.res.res_arr.shape == (4,146,146)
    #     assert self.res.res_arr.min() == -2.3716922523120409e-19
    #     assert self.res.res_arr.max() == 2.7105054312137611e-19
    #     assert self.res.res_arr.var() == 1.0278768366235814e-39 
    #     
    #     assert self.res.res_rot_mean.shape == (146,146)
    #     assert self.res.res_rot_mean.min() == -2.9879531662161895e-20
    #     assert self.res.res_rot_mean.max() == 2.7834474353638114e-20
    #     assert self.res.res_rot_mean.var() == 2.4903348430563392e-41 
    #     
    #     assert self.res.res_rot_var.shape == (146,146)
    #     assert self.res.res_rot_var.min() == 0.0
    #     assert self.res.res_rot_var.max() == 1.0123453782245222e-37
    #     assert self.res.res_rot_var.var() == 3.8865122287878855e-77 
    #     
    #     assert self.res.res_rot_median.shape == (146,146)
    #     assert self.res.res_rot_median.min() == -9.3001809623637242e-20
    #     assert self.res.res_rot_median.max() == 8.9730146834665022e-20
    #     assert self.res.res_rot_median.var() == 1.1850463046706196e-40 
    #     
    #     assert self.res.res_rot_mean_clip.shape == (146,146)
    #     #assert self.res.res_rot_mean_clip.min() == 0000
    #     #assert self.res.res_rot_mean_clip.max() == 000
    #     #assert self.res.res_rot_mean_clip.var() == 0000 
    #     
    #     assert self.res.res_clean_median.shape == (146,146)
    #     assert self.res.res_clean_median.min() == -0.1172708307675544
    #     assert self.res.res_clean_median.max() == 0.15102898151306077
    #     assert self.res.res_clean_median.var() == 0.0010473907380932109 
    #     
    # 
    # def test_peak_find(self):
    #     res_null = PynPoint.residuals('','','',intype='empty')
    #     dim = [400,600]
    #     cents = np.array([[100,200],[50,350],[215,79]])
    #     test_im = np.zeros(shape=dim)
    #     for i in range(0,cents.shape[0]): 
    #         test_im += PynPoint.Util.mk_gauss2D(dim[0],dim[1],30.,xcent=cents[i,0],ycent=cents[i,1]) 
    #     res_null.res_clean_mean_clip = test_im
    # 
    #     assert not hasattr(res_null,'x_peaks')
    #     assert not hasattr(res_null,'y_peaks')
    #     assert not hasattr(res_null,'h_peaks')
    #     assert not hasattr(res_null,'sig')
    #     assert not hasattr(res_null,'p_contour')
    #     assert not hasattr(res_null,'num_peaks')
    # 
    #     res_null.peak_find(limit=0.8)
    #     assert hasattr(res_null,'x_peaks')
    #     assert hasattr(res_null,'y_peaks')
    #     assert hasattr(res_null,'h_peaks')
    #     assert hasattr(res_null,'sig')
    #     assert hasattr(res_null,'p_contour')
    #     assert hasattr(res_null,'num_peaks')
    #     
    #     x = np.array(res_null.x_peaks)
    #     x.sort()
    #     x_cents = cents[:,1].copy()
    #     x_cents.sort()
    # 
    #     y = np.array(res_null.y_peaks)
    #     y.sort()
    #     y_cents = cents[:,0].copy()
    #     y_cents.sort()
    #     
    #     #print(x)
    #     #print(x_cents)
    #     
    #     assert res_null.num_peaks == cents.shape[0]
    #     assert np.allclose(res_null.h_peaks,np.ones(cents.shape[0]),rtol = limit2)
    #     assert np.allclose(x,x_cents,rtol = 1e-3)
    #     assert np.allclose(y,y_cents,rtol = 1e-2)
    #     print('x_peaks:')
    #     print(res_null.x_peaks)
    #     print('y_peaks:')
    #     print(res_null.y_peaks)
    #     print('h_peaks:')
    #     print(res_null.h_peaks)
    #     
    #     #assert 0
    #     
    #     # res_null.x_peaks = x_peaks 
    #     # res_null.y_peaks = y_peaks
    #     # res_null.h_peaks = h_peaks
    #     # res_null.sig = sig
    #     # res_null.p_contour = p
    #     # res_null.num_peaks = num_peaks
    #     
    # 
    #     #self.res.peak_find(limit=0.8)
    #     x = 1
    #     assert x==1
    
    # def test_res_cleaned(self):
    #     #self.res.res_cleaned()
    #     x = 1
    #     assert x==1
    # 
    # def test_resid_nomask(self):
    #     #self.res.resid_nomask()
    #     x = 1
    #     assert x==1
    
    # def test_save_restore(self):
    #     x = 1
    #     assert x==1


    def teardown(self):
        #tidy up
        print("tearing down " + __name__)
        pass