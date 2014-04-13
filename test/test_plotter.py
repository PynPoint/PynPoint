
# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

"""
Tests for `_Cache` module.
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import os
import numpy as np
import PynPoint

limit0 = 1e-20
limit1 = 1e-10
limit2 = 2e-4



class TestPlotter(object):

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

        pass
        

    def test_plt_res(self):
        res = self.res
        pynplot = PynPoint.plotter
        
        im1 = pynplot.plt_res(res,1,imtype='mean',smooth=None,returnval=True)
        im2 = pynplot.plt_res(res,1,imtype='mean_clip',smooth=None,returnval=True)
        im3 = pynplot.plt_res(res,1,imtype='median',smooth=None,returnval=True)
        im4 = pynplot.plt_res(res,1,imtype='var',smooth=None,returnval=True)

        im1s = pynplot.plt_res(res,1,imtype='mean',smooth=[2,2],returnval=True)
        im2s = pynplot.plt_res(res,1,imtype='mean_clip',smooth=[2,2],returnval=True)
        im3s = pynplot.plt_res(res,1,imtype='median',smooth=[2,2],returnval=True)
        im4s = pynplot.plt_res(res,1,imtype='var',smooth=[2,2],returnval=True)
        
        assert np.allclose(im1.mean() , -4.4349329318527051e-10,rtol=limit1)
        assert np.allclose(im2.mean() , -4.4349329318526586e-10,rtol=limit1)
        assert np.allclose(im3.mean() , -6.937365756153755e-08,rtol=limit1)
        assert np.allclose(im4.mean() , 2.4505879053707469e-10,rtol=limit1)

        # assert im1s.mean() == 0.00021770530872593523
        # assert im2s.mean() == 0.00021770530872593523
        # assert im3s.mean() == -0.0027173103236604501
        
        #!!!! VERY STRANGE THAT VALUES ARE THE SAME!!!! NEED To CHECK!!!!
        # assert im4s.mean() == 1.0


        # assert 1==2
        
    def test_anim_im_arr(self):
        x = 1
        
    def test_plt_psf_bais(self):
        x = 1
        
    def test_plt_psf_fit(self):
        x = 1


    def teardown(self):
        #tidy up
        print("tearing down " + __name__)
        pass