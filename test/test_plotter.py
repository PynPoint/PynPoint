
# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

"""
Tests for `_Cache` module.
"""
# from __future__ import print_function, division, absolute_import, unicode_literals

import os
import numpy as np
import PynPoint
from PynPoint import pynplot

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
        #pynplot = PynPoint.pynplot
        
        im1_temp = self.test_data_dir+'/im1_temp.fits'
        im2_temp = self.test_data_dir+'/im2_temp.fits'
        im3_temp = self.test_data_dir+'/im3_temp.fits'
        if os.path.isfile(im1_temp):
            os.remove(im1_temp)
        if os.path.isfile(im2_temp):
            os.remove(im2_temp)
        if os.path.isfile(im3_temp):
            os.remove(im3_temp)
        
        
        im1 = pynplot.plt_res(res,1,imtype='mean',smooth=None,returnval=True,savefits=im1_temp)
        im2 = pynplot.plt_res(res,1,imtype='mean_clip',smooth=None,returnval=True,savefits=im2_temp)
        im3 = pynplot.plt_res(res,1,imtype='median',smooth=None,returnval=True,savefits=im3_temp)
        im4 = pynplot.plt_res(res,1,imtype='var',smooth=None,returnval=True)
        im5 = pynplot.plt_res(res,1,imtype='sigma',smooth=None,returnval=True)
        im6 = pynplot.plt_res(res,1,imtype='mean_sigmamean',smooth=None,returnval=True)

        im1s = pynplot.plt_res(res,1,imtype='mean',smooth=[2,2],returnval=True)
        im2s = pynplot.plt_res(res,1,imtype='mean_clip',smooth=[2,2],returnval=True)
        im3s = pynplot.plt_res(res,1,imtype='median',smooth=[2,2],returnval=True)
        im4s = pynplot.plt_res(res,1,imtype='var',smooth=[2,2],returnval=True)
        
        assert np.allclose(im1.mean() , -4.4349329318527051e-10,rtol=limit1)
        assert np.allclose(im2.mean() , -4.4349329318526586e-10,rtol=limit1)
        assert np.allclose(im3.mean() , -6.937365756153755e-08,rtol=limit1)
        assert np.allclose(im4.mean() , 2.4505879053707469e-10,rtol=limit1)
        assert np.allclose(im5.mean() , 1.1474961173597471e-05,rtol=limit1)
        assert np.allclose(im6.mean() , 0.00041874127024927644,rtol=limit1)

        assert np.allclose(im1s.mean() , -3.1181363069737826e-10,rtol=limit1)
        assert np.allclose(im2s.mean() , -3.118136306973763e-10,rtol=limit1)
        assert np.allclose(im3s.mean() , -6.8424237888987089e-08,rtol=limit1)
        assert np.allclose(im4s.mean() , -2.369120550016369e-10,rtol=limit1)
        
        if os.path.isfile(im1_temp):
            os.remove(im1_temp)
        if os.path.isfile(im2_temp):
            os.remove(im2_temp)
        if os.path.isfile(im3_temp):
            os.remove(im3_temp)
        
        
        try:
            im_temp = pynplot.plt_res(res,1,imtype='wrong_entry',smooth=None,returnval=True,savefits=im1_temp)
        except AssertionError, e:
            pass
        # assert(e[0][0:20] == 'Error: options for a')
        assert(e[0] == "Error: options for ave keyword are ['mean', 'mean_clip', 'median', 'var', 'sigma', 'mean_sigmamean']")
        
        # assert(1==2)
        
        #!!!! VERY STRANGE THAT VALUES ARE THE SAME!!!! NEED To CHECK!!!!
        # assert im4s.mean() == 1.0


        # assert 1==2
        
    def test_anim_im_arr(self):
        pynplot.anim_im_arr(self.res,im_range=[0,3])
        pynplot.anim_im_arr(self.images,im_range=[0,3],time_gap =0.01)
        pynplot.anim_im_arr(self.basis,im_range=None)
        # x = 1
        
    def test_im_arr(self,tmpdir):
        # im1_temp = tmpdir+'/im1_temp.fits'
        im1_temp = self.test_data_dir+'/im1_temp.fits'
        im2_temp = self.test_data_dir+'/im2_temp.fits'
        im3_temp = self.test_data_dir+'/im3_temp.fits'
        if os.path.isfile(im1_temp):
            os.remove(im1_temp)
        if os.path.isfile(im2_temp):
            os.remove(im2_temp)
        if os.path.isfile(im3_temp):
            os.remove(im3_temp)


        im1 = pynplot.plt_im_arr(self.images,0,returnval=True,savefits=im1_temp)
        # im1 = pynplot.plt_im_arr(self.images,0,returnval=True,savefits='test_data/im1_temp.fits')
        im2 = pynplot.plt_im_arr(self.basis,1,returnval=True,savefits=im2_temp)
        im3 = pynplot.plt_im_arr(self.res,2,returnval=True,savefits=im3_temp)
        
        assert np.allclose(im1.mean(),7.8226590446234423e-06,rtol=limit1)
        assert np.allclose(im2.mean(),-2.1439609406225518e-07,rtol=limit1)
        assert np.allclose(im3.mean(),1.0384630019636352e-05,rtol=limit1)
        
        assert os.path.isfile(im1_temp)
        
        if os.path.isfile(im1_temp):
            os.remove(im1_temp)
        if os.path.isfile(im2_temp):
            os.remove(im2_temp)
        if os.path.isfile(im3_temp):
            os.remove(im3_temp)
        
    
    def test_plt_psf_basis(self):
        im1_temp = self.test_data_dir+'/im1_temp.fits'
        im2_temp = self.test_data_dir+'/im2_temp.fits'
        if os.path.isfile(im1_temp):
            os.remove(im1_temp)
        if os.path.isfile(im2_temp):
            os.remove(im2_temp)

        
        im1 = pynplot.plt_psf_basis(self.res,1,returnval=True,savefits=im1_temp)
        im2 = pynplot.plt_psf_basis(self.basis,2,returnval=True,savefits=im2_temp)

        assert np.allclose(im1.mean(),0.00065586403734947912,rtol=limit1)
        assert np.allclose(im2.mean(),-0.0004242279204660866,rtol=limit1)

        if os.path.isfile(im1_temp):
            os.remove(im1_temp)
        if os.path.isfile(im2_temp):
            os.remove(im2_temp)
        
        
    def test_plt_psf_fit(self):
        im1_temp = self.test_data_dir+'/im1_temp.fits'
        im2_temp = self.test_data_dir+'/im2_temp.fits'
        if os.path.isfile(im1_temp):
            os.remove(im1_temp)
        if os.path.isfile(im2_temp):
            os.remove(im2_temp)

        
        im1 = pynplot.plt_psf_model(self.res,1,2,returnval=True,savefits=im1_temp)
        im2 = pynplot.plt_psf_model(self.res,2,3,returnval=True,savefits=im2_temp)

        assert np.allclose(im1.mean(),9.083088185930825e-06,rtol=limit1)
        assert np.allclose(im2.mean(),1.0384630019636352e-05,rtol=limit1)

        if os.path.isfile(im1_temp):
            os.remove(im1_temp)
        if os.path.isfile(im2_temp):
            os.remove(im2_temp)
        
        

    def teardown(self):
        #tidy up
        print("tearing down " + __name__)
        pass