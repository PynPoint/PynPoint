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


# System imports
from __future__ import print_function, division

# External modules
import numpy as np

#import external functions:
from PynPoint._BasePynPoint import base_pynpoint
from PynPoint import _Creators
from PynPoint import _Util

#import extra PynPoint functions:



#Residuals Class
class residuals(base_pynpoint):
    """Object for dealing with the residual data. This includes object detection and flux measurement"""

    def __init__(self):
        """
        Initialise an instance of the residual class.
        """            
        self.obj_type = 'PynPoint_residuals'
        self.num_coeff = np.nan
        self.extra_rot = np.nan



    @classmethod
    def create_restore(cls, filename):
        """
        Restores an instance from saved file.
        See :py:func:`_Creators.restore` for more details.
        """
        
        obj = cls()
        _Creators.restore(obj,filename)
        return obj
        
    @classmethod
    def create_winstances(cls, images,basis):
        #TODO: use normal initializer for this.
        obj = cls()
        #---importing data from images instance:---#
        obj.im_arr = images.im_arr
        obj.para = images.para 
        #---import data from basis instance:---#
        obj.psf_basis = basis.psf_basis
        #---import data from both instances:---#
        assert np.array_equal(basis.cent_mask,images.cent_mask)
        obj.cent_mask = images.cent_mask
        obj.im_ave = basis.im_ave
        assert np.array_equal(basis.psf_basis[0,].shape , images.im_arr[0,].shape)

        obj.im_shape = images.im_arr[0,].shape

        return obj


    def res_arr(self,num_coeff):
        """
        Returns a 3D data cube of the residuals, i.e. a psf model is 
        removed from every image.
        """
        if not (hasattr(self,'_res_arr') and (self.num_coeff == num_coeff)):
            self._mk_res_arr(num_coeff)
        return self._res_arr
        
    def res_rot(self,num_coeff,extra_rot =0.0):
        """
        Returns a 3D data cube of residuals where all the images
        have been rotated to have the same para angle.
        """
        if not (hasattr(self, '_res_rot') and (self.num_coeff == num_coeff) and (self.extra_rot == extra_rot)):
            self._mk_res_rot(num_coeff,extra_rot =extra_rot )
        return self._res_rot
        
    def res_rot_mean(self,num_coeff,extra_rot =0.0):
        """
        Returns a 2D image of residuals after averaging down the stack.
        All the images in the stack are rotated to that they 
        have the same para angle.
        """
        if not (hasattr(self, '_res_rot_mean') and (self.num_coeff == num_coeff) and (self.extra_rot == extra_rot)):
            self._mk_res_rot_mean(num_coeff,extra_rot = extra_rot )
        return self._res_rot_mean

    def res_rot_median(self,num_coeff,extra_rot =0.0):
        """
        Returns a 2D image of residuals after averaging down the stack.
        All the images in the stack are rotated to that they 
        have the same para angle.
        """
        if not (hasattr(self, '_res_rot_median') and (self.num_coeff == num_coeff) and (self.extra_rot == extra_rot)):
            self._mk_res_rot_median(num_coeff,extra_rot =extra_rot )
        return self._res_rot_median

        
    def res_rot_mean_clip(self,num_coeff,extra_rot =0.0):
        """
        Returns a 2D image of residuals after averaging down the stack.
        All the images in the stack are rotated to that they 
        have the same para angle.
        (3 sigma)
        """
        # if not (hasattr(self, '_res_rot_mean_clip') and (self.num_coeff == num_coeff)):
        self._mk_res_rot_mean_clip(num_coeff,extra_rot =extra_rot )
        return self._res_rot_mean_clip
    
        
    def res_rot_var(self,num_coeff,extra_rot = 0.0):
        """
        Returns a 2D image of the variance of the residuals down the stack.
        All the images in the stack are rotated to that they 
        have the same para angle.
        """
        if not (hasattr(self, '_res_rot_var') and (self.num_coeff == num_coeff) and (self.extra_rot == extra_rot)):
            self._mk_res_rot_var(num_coeff,extra_rot =extra_rot )
        return self._res_rot_var
        
                
    def _psf_im(self,num_coeff):
        """
        Returns a data cube with a model for the PSF.
        """
        
        if not (hasattr(self, 'psf_im_arr') and (self.num_coeff == num_coeff)):
            self.mk_psfmodel(num_coeff)
            self.num_coeff = num_coeff
        return self.psf_im_arr

    # ---Internal functions ---#
    
    def _mk_res_arr(self,num_coeff):
        res_arr = self.im_arr.copy()
        psf_im = self._psf_im(num_coeff)
        for i in range(0,len(res_arr[:,0,0])):
            res_arr[i,] -= (psf_im[i,] * self.cent_mask)
        self._res_arr = res_arr
    
    def _mk_res_rot(self,num_coeff,extra_rot = 0.0):

        delta_para = self.para[0] - self.para
        res_rot = np.zeros(shape=self.im_arr.shape)
        res_arr = self.res_arr(num_coeff)
        for i in range(0,len(delta_para)):
            res_temp = res_arr[i,]
            res_rot[i,] = _Util.mk_rotate(res_temp,delta_para[i]+extra_rot)
        self._res_rot = res_rot
         
    def _mk_res_rot_mean(self,num_coeff,extra_rot = 0.0):
        res_rot = self.res_rot(num_coeff,extra_rot = extra_rot)
        self._res_rot_mean = np.mean(res_rot,axis=0)

    def _mk_res_rot_median(self,num_coeff,extra_rot = 0.0):
        self._res_rot_median = np.median(self.res_rot(num_coeff,extra_rot =extra_rot),axis=0)
        
    def _mk_res_rot_var(self,num_coeff,extra_rot = 0.0):
        res_rot_temp = self.res_rot(num_coeff).copy()
        for i in range(0,res_rot_temp.shape[0]):
            res_rot_temp[i,] -= - self.res_rot_mean(num_coeff,extra_rot =extra_rot)
        res_rot_var = (res_rot_temp**2.).sum(axis=0)
        self._res_rot_var = res_rot_var
        
    def _mk_res_rot_mean_clip(self,num_coeff,extra_rot = 0.0):
        res_rot_mean_clip = np.zeros(self.im_shape)
        res_rot = self.res_rot(num_coeff,extra_rot =extra_rot)
        for i in range(0,res_rot_mean_clip.shape[0]):
            for j in range(0,res_rot_mean_clip.shape[1]):
                temp = res_rot[:,i,j]
                if temp.var() > 0.0:
                    a = temp - temp.mean()
                    b1 = a.compress((a < 3.0*np.sqrt(a.var())).flat)
                    b2 = b1.compress((b1 > (-1.0)* 3.0*np.sqrt(a.var())).flat)
                    res_rot_mean_clip[i,j] = temp.mean() + b2.mean()                
        self._res_rot_mean_clip = res_rot_mean_clip #* self.cent_mask
                        
        
    def _mk_res_mean_smooth(self,num_coeff,sigma=(2,2)):
        self._res_mean_smooth = self._mk_arr_smooth(self.res_rot_mean(num_coeff),self.res_rot_var(num_coeff),self.cent_mask,sigma=sigma)

    def _mk_res_mean_clip_smooth(self,num_coeff,sigma=(2,2)):
        self._res_mean_clip_smooth = self._mk_arr_smooth(self.res_rot_mean_clip(num_coeff),self.res_rot_var(num_coeff),self.cent_mask,sigma=sigma)

    def _mk_res_median_smooth(self,num_coeff,sigma=(2,2)):
        self._res_median_smooth = self._mk_arr_smooth(self.res_rot_median(num_coeff),self.res_rot_var(num_coeff),self.cent_mask,sigma=sigma)        

    def mk_psfmodel(self, num):
        super(residuals, self).mk_psfmodel(self, num)

