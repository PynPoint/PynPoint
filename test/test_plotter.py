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
Tests for `_PynPlot` module.
"""
# from __future__ import print_function, division, absolute_import, unicode_literals

import os
import numpy as np
import PynPoint
#from PynPoint import pynplot
import PynPoint.PynPlot as pynplot

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

        self.basis = PynPoint.basis.create_wdir(self.test_data_dir,
                                cent_remove=True,resize=False,ran_sub=False,recent=False)

        self.images = PynPoint.images.create_wdir(self.test_data_dir,
                                cent_remove=True,resize=False,ran_sub=False,recent=False)
                                
        self.res = PynPoint.residuals.create_winstances(self.images,self.basis)
        self.num_files = self.images.im_arr.shape[0]

        pass
        

    def test_plt_res(self):
        res = self.res

        im1_temp = self.test_data_dir+'/outputs/im1_temp.fits'
        im2_temp = self.test_data_dir+'/outputs/im2_temp.fits'
        im3_temp = self.test_data_dir+'/outputs/im3_temp.fits'
        if os.path.isfile(im1_temp):
            os.remove(im1_temp)
        if os.path.isfile(im2_temp):
            os.remove(im2_temp)
        if os.path.isfile(im3_temp):
            os.remove(im3_temp)
        
        
        im1 = pynplot.plt_res(res,1,imtype='mean',smooth=None,returnval=True,savefits=im1_temp,mask_nan=False)
        im2 = pynplot.plt_res(res,1,imtype='mean_clip',smooth=None,returnval=True,savefits=im2_temp,mask_nan=False)
        im3 = pynplot.plt_res(res,1,imtype='median',smooth=None,returnval=True,savefits=im3_temp,mask_nan=False)
        im4 = pynplot.plt_res(res,1,imtype='var',smooth=None,returnval=True,mask_nan=False)
        im5 = pynplot.plt_res(res,1,imtype='sigma',smooth=None,returnval=True,mask_nan=False)
        im6 = pynplot.plt_res(res,1,imtype='mean_sigmamean',smooth=None,returnval=True,mask_nan=False)

        im1s = pynplot.plt_res(res,1,imtype='mean',smooth=[2,2],returnval=True,mask_nan=False)
        im2s = pynplot.plt_res(res,1,imtype='mean_clip',smooth=[2,2],returnval=True,mask_nan=False)
        im3s = pynplot.plt_res(res,1,imtype='median',smooth=[2,2],returnval=True,mask_nan=False)
        im4s = pynplot.plt_res(res,1,imtype='var',smooth=[2,2],returnval=True,mask_nan=False)
        
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

        assert(e[0] == "Error: options for ave keyword are ['mean', 'mean_clip', 'median', 'var', 'sigma', 'mean_sigmamean']")

    def test_anim_im_arr(self):
        pynplot.anim_im_arr(self.res,im_range=[0,3])
        pynplot.anim_im_arr(self.images,im_range=[0,3],time_gap =0.01)
        pynplot.anim_im_arr(self.basis,im_range=None)

    def test_im_arr(self,tmpdir):

        im1_temp = self.test_data_dir+'/outputs/im1_temp.fits'
        im2_temp = self.test_data_dir+'/outputs/im2_temp.fits'
        im3_temp = self.test_data_dir+'/outputs/im3_temp.fits'
        if os.path.isfile(im1_temp):
            os.remove(im1_temp)
        if os.path.isfile(im2_temp):
            os.remove(im2_temp)
        if os.path.isfile(im3_temp):
            os.remove(im3_temp)


        im1 = pynplot.plt_im_arr(self.images,0,returnval=True,savefits=im1_temp,mask_nan=False)
        im2 = pynplot.plt_im_arr(self.basis,1,returnval=True,savefits=im2_temp,mask_nan=False)
        im3 = pynplot.plt_im_arr(self.res,2,returnval=True,savefits=im3_temp,mask_nan=False)
        
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
        im1_temp = self.test_data_dir+'/outputs/im1_temp.fits'
        im2_temp = self.test_data_dir+'/outputs/im2_temp.fits'
        if os.path.isfile(im1_temp):
            os.remove(im1_temp)
        if os.path.isfile(im2_temp):
            os.remove(im2_temp)

        
        im1 = pynplot.plt_psf_basis(self.res,1,returnval=True,savefits=im1_temp,mask_nan=False)
        im2 = pynplot.plt_psf_basis(self.basis,2,returnval=True,savefits=im2_temp,mask_nan=False)

        assert np.allclose(im1.mean(),0.00065586403734947912,rtol=limit1)
        assert np.allclose(im2.mean(),-0.0004242279204660866,rtol=limit1)

        if os.path.isfile(im1_temp):
            os.remove(im1_temp)
        if os.path.isfile(im2_temp):
            os.remove(im2_temp)
        
        
    def test_plt_psf_fit(self):
        im1_temp = self.test_data_dir+'/outputs/im1_temp.fits'
        im2_temp = self.test_data_dir+'/outputs/im2_temp.fits'
        if os.path.isfile(im1_temp):
            os.remove(im1_temp)
        if os.path.isfile(im2_temp):
            os.remove(im2_temp)

        
        im1 = pynplot.plt_psf_model(self.res,1,2,returnval=True,savefits=im1_temp,mask_nan=False)
        im2 = pynplot.plt_psf_model(self.res,2,3,returnval=True,savefits=im2_temp,mask_nan=False)

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