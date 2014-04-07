
# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

"""
Tests for `_Mask` module.
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
from PynPoint import _Mask


limit1 = 1.0e-10
limit2 = 1.0e-7


class TestPynpoint_v1_5(object):

    def setup(self):
        #prepare unit test. Load data etc
        print("setting up " + __name__)
        #file1 = '/Users/amaraa/Work/Active_Projects/PynPoint_svn_sync/PynPoint/Tests/Test_data/Naco_M_im_temp.hdf5'
        #file2 = '/Users/amaraa/Work/Active_Projects/PynPoint_svn_sync/PynPoint/Tests/Test_data/Naco_M_basis.hdf5'
#         mask1 = PynPoint.mask(100,200,pattern='circle',para_ini=0.0,fsize=0.15,fxcent=0.2,fycent=0.5)
#         mask2 = PynPoint.mask(200,200,pattern='circle',para_ini=20.0,fsize=0.15,fxcent=0.2,fycent=0.5)
#         self.mask1 = mask1
#         self.mask2 = mask2
        #self.file2 = file2
        #self.basis = basis

        pass

    def test_mk_circle(self):
        im_temp = _Mask.mk_circle(400,400,100.,200.,20.)
        assert im_temp.shape[0] == 400
        assert im_temp.shape[1] == 400
        assert im_temp.max() == 1.0
        assert im_temp.min() == 0.0
        assert im_temp[200,100] == 0.0
        assert im_temp[200,121] == 1.0
        assert im_temp[200,79] == 1.0

#     def test_initialisation(self):
#         assert self.mask1.xnum == 100
#         assert self.mask1.ynum == 200
#         assert self.mask1.para_ini == 0.0
#         assert self.mask1.fsize == 0.15
#         assert self.mask1.size_pix == 15.
#         assert self.mask1.fxcent == 0.2
#         assert self.mask1.xcent == 20.0
#         assert self.mask1.fycent == 0.5
#         assert self.mask1.ycent== 50.0
#         assert self.mask1.pattern == 'circle'
#         assert len(self.mask1.mask_base.shape) == 2
#         assert self.mask1.mask_base.shape[0] == 100
#         assert self.mask1.mask_base.shape[1] == 200
#         assert self.mask1.mask_base.min() == 0.0
#         assert self.mask1.mask_base.max() == 1.0
#         assert self.mask1.mask_base[50,20] == 0.0
#         assert self.mask1.mask_base[50,28] == 1.0
#         assert self.mask1.mask_base[50,27] == 0.0        


#     def test_mask(self):
#         mask_rot1 = self.mask2.mask(del_para=90)
#         assert self.mask2.mask_base[100,40] == 0.0
#         assert mask_rot1.shape[0] == 200
#         assert mask_rot1.shape[1] == 200
#         assert mask_rot1.max() == 1.0
#         assert mask_rot1.min() == 0.0
#         assert mask_rot1[40,100] == 0.0
#         assert self.mask2.temp_xcent == 100.
#         assert self.mask2.temp_ycent == 40.

#     def test_plt_mask(self):
#         self.mask1.plt_mask(para=None)
#         self.mask2.plt_mask(para=90)
        
    def test_mk_cent_remove(self):
        temp_arr = np.ones([10,400,400])*np.pi
        im_arr_omask,im_arr_imask,cent_mask = _Mask.mk_cent_remove(temp_arr,cent_size=0.2,edge_size=1.0)
        assert temp_arr.shape[0] == 10
        assert im_arr_omask.shape[0] == 10
        assert im_arr_omask.shape[1] == 400
        assert im_arr_omask.shape[2] == 400
        assert np.allclose(im_arr_omask[0,],cent_mask*np.pi,rtol=limit1)
        assert np.allclose(im_arr_imask[0,],(1.0 - cent_mask)*np.pi,rtol=limit1)
        assert np.allclose(im_arr_omask[5,],cent_mask*np.pi,rtol=limit1)
        assert np.allclose(im_arr_imask[5,],(1.0 - cent_mask)*np.pi,rtol=limit1)
        assert im_arr_omask[0,200,200] == 0.0
        assert im_arr_imask[0,200,200] == np.pi
        assert im_arr_omask[0,0,0] == 0.0
        assert im_arr_imask[0,0,0] == np.pi
        assert im_arr_omask[0,200,20] == np.pi
        assert im_arr_imask[0,200,20] == 0.0
        assert cent_mask.max() == 1.0
        assert cent_mask.min() == 0.0
        

    def teardown(self):
        #tidy up
        print("tearing down " + __name__)
        pass

