        
#import external functions:
import pyfits
import scipy
import pylab as pl
# import glob
# import ImageChops
# from PIL import Image
# from scipy.optimize import fmin
import time
from scipy.ndimage.filters import gaussian_filter
# import random
# import h5py
import numpy as np
# from scipy import linalg

#import extra PynPoint functions:
import Util
# from PynPoint_parent import pynpoint_parent
from parent import pynpoint_parent



#Residuals Class
class residuals(pynpoint_parent):
    """Object for dealing with the residual data. This includes object detection and flux measurement"""

    def __init__(self): #self,ims,basis,num_coeff,mask=None,limit=0.8,printit=False,peak_find=False,num_tweak=0,coeff_type=False,intype=None,file_in=None):
    	"""
        Initialise an instance of the residual class. The result is simple and
        almost empty (in terms of attributes)
        """            
        self.obj_type = 'PynPoint_residuals'
        self.num_coeff = np.nan
        
        
        
    @staticmethod
    def create_restore(file_in):

        obj = residuals()
        Util.restore_data(obj,file_in)
        return obj
        
    @staticmethod
    def create_winstances(images,basis):
        obj = residuals()
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


    def plt_res(self,num_coeff,imtype='mean',smooth=None):
        """
        plots the resulting residual images. 
        """
        
        options = ['mean','mean_clip','median','var']#,'psf']
        
        assert ave in option, 'Error: options for ave keyword are %s'%options
        if not smooth is None:
            assert isinstance(smooth, ( int, long,float) ), 'Error: smooth keyword should be set to a number.'
        assert isinstance(num_coeff, ( int, long,float) ), 'Error: num_basis should be set to a number.'
        
        
        if smooth is None:            
            if imtype == 'mean':
                im = self.res_rot_mean()                
            elif imtype == 'mean_clip':
                im = self.res_rot_mean_clip()
            elif imtype == 'median':
                im = self.res_rot_median()
            elif imtype == 'var':
                im = self.res_rot_var()
            else:
                print('Error: something is wrong with ave keyword. Funny its not picked up by assert and options!')
                return
        else:
            if imtype == 'mean':
                im = self.res_mean_smooth()
            elif imtype == 'mean_clip':
                im = self.res_mean_clip_smooth()
            elif imtype == 'median':
                im = self.res_median_smooth()
            elif imtype == 'var':
                print('Error: var image currently does not get plotted with smoothing')
                return                
            else:
                print('Error: something is wrong with ave keyword. Funny its not picked up by assert and options!')
        
        
        pl.figure()
        pl.clf()
        pl.imshow(im,origin='lower',interpolation='nearest')
        pl.title('Residual Image: '+imtype,size='large')
        pl.colorbar()
        return im
            


    def res_arr(self,num_coeff):
        """
        Returns a 3D data cube of the residuals, i.e. a psf model is 
        removed from every image.
        """
        if not (hasattr(self,'_res_arr') and (self.num_coeff == num_coeff)):
            self._mk_res_arr(num_coeff)
        return self._res_arr
        
    def res_rot(self,num_coeff):
        """
        Returns a 3D data cube of residuals where all the images
        have been rotated to have the same para angle.
        """
        if not (hasattr(self, '_res_rot') and (self.num_coeff == num_coeff)):
            self._mk_res_rot(num_coeff)
        return self._res_rot
        
    def res_rot_mean(self,num_coeff):
        """
        Returns a 2D image of residuals after averaging down the stack.
        All the images in the stack are rotated to that they 
        have the same para angle.
        """
        if not (hasattr(self, '_res_rot_mean') and (self.num_coeff == num_coeff)):
            self._mk_res_rot_mean(num_coeff)
        return self._res_rot_mean

    def res_rot_median(self,num_coeff):
        """
        Returns a 2D image of residuals after averaging down the stack.
        All the images in the stack are rotated to that they 
        have the same para angle.
        """
        if not (hasattr(self, '_res_rot_median') and (self.num_coeff == num_coeff)):
            self._mk_res_rot_median(num_coeff)
        return self._res_rot_median

        
    def res_rot_mean_clip(self,num_coeff):
        """
        Returns a 2D image of residuals after averaging down the stack.
        All the images in the stack are rotated to that they 
        have the same para angle.
        """
        if not (hasattr(self, '_res_rot_mean_clip') and (self.num_coeff == num_coeff)):
            self._mk_res_rot_mean_clip(num_coeff)
        return self._res_rot_mean_clip
    
        
    def res_rot_var(self,num_coeff):
        """
        Returns a 2D image of the variance of the residuals down the stack.
        All the images in the stack are rotated to that they 
        have the same para angle.
        """
        if not (hasattr(self, '_res_rot_var') and (self.num_coeff == num_coeff)):
            self._mk_res_rot_var(num_coeff)
        return self._res_rot_var
        
                
    def psf_im(self,num_coeff):
        """
        Returns a data cube with a model for the PSF.
        """
        
        if not (hasattr(self, '_res_rot_var') and (self.num_coeff == num_coeff)):
            self.mk_psfmodel(self,num_coeff)
            self.num_coeff = num_coeff
        return self.psf_im_arr

    def res_mean_smooth(self,num_coeff,sigma=(2,2)):
        """
        """
        if not (hasattr(self, '_res_mean_smooth') and (self.num_coeff == num_coeff)):
            self._mk_res_mean_smooth(num_coeff,sigma=(2,2))
        return self._res_mean_smooth

    def res_mean_clip_smooth(self,num_coeff,sigma=(2,2)):
        """
        """
        if not (hasattr(self, '_res_mean_clip_smooth') and (self.num_coeff == num_coeff)):
            self._mk_res_mean_clip_smooth(num_coeff,sigma=(2,2))
        return self._res_mean_smooth

    def res_median_smooth(self,num_coeff,sigma=(2,2)):
        """
        """
        if not (hasattr(self, '_res_median_smooth') and (self.num_coeff == num_coeff)):
            self._mk_res_median_smooth(num_coeff,sigma=(2,2))
        return self._res_median_smooth

        
    # ---Internal functions ---#
    
    def _mk_res_arr(self,num_coeff):
    	res_arr = self.im_arr.copy()
        psf_im = self.psf_im(num_coeff)
        for i in range(0,len(res_arr[:,0,0])):
            res_arr[i,] -= (psf_im[i,] * self.cent_mask)
            # print('HIHI')
        self._res_arr = res_arr
    
    def _mk_res_rot(self,num_coeff):        
    	delta_para = self.para[0] - self.para
    	res_rot = np.zeros(shape=self.im_arr.shape)
        res_arr = self.res_arr(num_coeff)
        for i in range(0,len(delta_para)):
            res_temp = res_arr[i,]
            res_rot[i,] = Util.mk_rotate(res_temp,delta_para[i])
        self._res_rot = res_rot
         
    def _mk_res_rot_mean(self,num_coeff):
        res_rot = self.res_rot(num_coeff)
        self._res_rot_mean = np.mean(res_rot,axis=0)

    def _mk_res_rot_median(self,num_coeff):
        self._res_rot_median = np.median(self.res_rot(num_coeff),axis=0)
        
    def _mk_res_rot_var(self,num_coeff):
        res_rot_temp = self.res_rot(num_coeff).copy()
        for i in range(0,res_rot_temp.shape[0]):
            res_rot_temp[i,] -= - self.res_rot_mean(num_coeff)
        res_rot_var = (res_rot_temp**2.).sum(axis=0)
        self._res_rot_var = res_rot_var
        
    def _mk_res_rot_mean_clip(self,num_coeff):
        res_rot_mean_clip = np.zeros(self.im_shape)
        res_rot = self.res_rot(num_coeff)
        for i in range(0,res_rot_mean_clip.shape[0]):
            for j in range(0,res_rot_mean_clip.shape[1]):
                temp = res_rot[:,i,j]
                a = temp - temp.mean()
                b1 = a.compress((a < 3.0*np.sqrt(a.var())).flat)
                b2 = b1.compress((b1 > (-1.0)* 3.0*np.sqrt(a.var())).flat)
                res_rot_mean_clip[i,j] = temp.mean() + b2.mean()                
        self._res_rot_mean_clip = res_rot_mean_clip
                        
        
    def _mk_res_mean_smooth(self,num_coeff,sigma=(2,2)):
        self._res_mean_smooth = self._mk_arr_smooth(self.res_rot_mean(num_coeff),self.res_rot_var(num_coeff),self.cent_mask,sigma=sigma)

    def _mk_res_mean_clip_smooth(self,num_coeff,sigma=(2,2)):
        self._res_mean_clip_smooth = self._mk_arr_smooth(self.res_rot_mean_clip(num_coeff),self.res_rot_var(num_coeff),self.cent_mask,sigma=sigma)

    def _mk_res_median_smooth(self,num_coeff,sigma=(2,2)):
        self._res_median_smooth = self._mk_arr_smooth(self.res_rot_median(num_coeff),self.res_rot_var(num_coeff),self.cent_mask,sigma=sigma)        

    def _mk_arr_smooth(self,arr_ave,arr_var,cent_mask,sigma=(2,2)):
        (izero,jzero) = np.where(arr_var==0)
        im1 = arr_ave/np.sqrt(arr_var)
        im1[izero,jzero] = 0.0    
        im1 *= cent_mask
        im2 = gaussian_filter(im1,sigma=sigma)
        return im2 * cent_mask
        
    

        
