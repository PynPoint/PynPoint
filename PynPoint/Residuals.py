        
#import external functions:
import pyfits
import scipy
import pylab as pl
import glob
import ImageChops
from PIL import Image
from scipy.optimize import fmin
import time
from scipy.ndimage.filters import gaussian_filter
import random
import h5py
import numpy as np
from scipy import linalg

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
        for i in range(0,len(res_arr[:,0,0])):
            res_arr[i,] -= (self.psf_im(num_coeff)[i,] * self.cent_mask)
            # print('HIHI')
        self._res_arr = res_arr
    
    def _mk_res_rot(self,num_coeff):        
    	delta_para = self.para[0] - self.para
    	res_rot = np.zeros(shape=self.im_arr.shape)
        for i in range(0,len(delta_para)):
            res_rot[i,] = Util.mk_rotate(self.res_arr(num_coeff)[i,],delta_para[i])
        self._res_rot = res_rot
         
    def _mk_res_rot_mean(self,num_coeff):
        self._res_rot_mean = np.mean(self.res_rot(num_coeff),axis=0)

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
        for i in range(0,res_rot_mean_clip.shape[0]):
            for j in range(0,res_rot_mean_clip.shape[1]):
                temp = self.res_rot(num_coeff)[:,i,j]
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
        
    

        
        
        


    # def mk_res_meanclip(self):
    #     TBD
        
        
    
    
    # def res2d():

        # self.res_arr = res_arr
        # self.res_rot = res_rot
        # self.res_rot_mean = res_rot_mean
        # self.res_rot_mean_clip = res_rot_mean_clip
        # self.res_rot_median = res_rot_median
        # self.res_rot_var = res_rot_var
        # self.res_rot_var2 = res_rot_var2
    
        
        
        # if intype == 'empty':
        #     return
        # elif intype == 'restore':
        #     print 'Restoring data from file:'
        #     print file_in
        #     print 'All other keywords are being ignored.'
        #     self.restore(file_in)
        #     return
        # else:
        #     self.resid_nomask(ims,basis,num_coeff,mask,num_tweak=num_tweak,coeff_type=coeff_type)
        #     self.res_cleaned(basis)
        #     if peak_find is not False:
        #         self.peak_find(limit=limit,printit = printit)
    
    # def resid_nomask(self,ims,basis,num_coeff,mask,num_tweak=0,coeff_type=False):
    #     """ This function is used to correct for the PSF without using a mask"""
    #     print mask
    #     #ims.mk_psfmodel(basis,num_coeff,mask=None)
    #     ims.mk_psfmodel(basis,num_coeff)#,mask=mask)
    #     if num_tweak > 0:
    #         num_it = len(coeff_type) 
    #         for i in range(0,num_it):
    #             ims.mk_tweak(basis,mask,num_tweak,coeff_type=coeff_type[i])
    #             
    #     delta_para = ims.para[0] - ims.para
    #     res_arr = np.zeros(shape=ims.im_arr.shape)
    #     res_rot = np.zeros(shape=ims.im_arr.shape)
    #     res_meansub = np.zeros(shape=ims.im_arr.shape)
    #     res_meansub2 = np.zeros(shape=ims.im_arr.shape)       
    #     for i in range(0,delta_para.size):
    #        res_arr[i,] = ims.im_arr[i,] - (ims.psf_im[i,] * ims.cent_mask)
    #        res_rot[i,] = Util.mk_rotate(res_arr[i,],delta_para[i])
    # 
    #     res_rot_mean = np.mean(res_rot,axis=0)
    #     res_rot_median = np.median(res_rot,axis=0)
    #     for i in range(0,delta_para.size):
    #        res_meansub[i,] = res_arr[i,] - res_rot_mean
    #        res_meansub2[i,] = res_arr[i,] - res_rot_median
    #     res_rot_var =  (res_meansub**2.).sum(axis=0)
    #     res_rot_var2 = (res_meansub2**2.).sum(axis=0)
    #     res_rot_mean_clip = np.zeros(res_rot_mean.shape)
    #     for i in range(0,res_rot_mean.shape[0]):
    #        for j in range(0,res_rot_mean.shape[1]):
    #           temp = temp = res_rot[:,i,j]
    #           a = temp - temp.mean()
    #           b1 = a.compress((a < 3.0*np.sqrt(a.var())).flat)
    #           b2 = b1.compress((b1 > (-1.0)* 3.0*np.sqrt(a.var())).flat)
    #           res_rot_mean_clip[i,j] = temp.mean() + b2.mean()
    # 
    #     self.res_arr = res_arr
    #     self.res_rot = res_rot
    #     self.res_rot_mean = res_rot_mean
    #     self.res_rot_mean_clip = res_rot_mean_clip
    #     self.res_rot_median = res_rot_median
    #     self.res_rot_var = res_rot_var
    #     self.res_rot_var2 = res_rot_var2

    # def res_cleaned(self,basis):
    #    """This function is used to produce a tidied up version of the residual image"""
    # 
    #    (izero,jzero) = np.where(self.res_rot_var==0)
    #    im1 = self.res_rot_mean/np.sqrt(self.res_rot_var)
    #    im1[izero,jzero] = 0.0    
    #    im2 = gaussian_filter(im1*basis.cent_mask,sigma=(2,2))
    #    #    imshow(im2*basis.cent_mask)
    #    self.res_clean_mean = im2*basis.cent_mask
    # 
    #    (izero,jzero) = np.where(self.res_rot_var2==0)
    #    im1 = self.res_rot_median/np.sqrt(self.res_rot_var2)
    #    im1[izero,jzero] = 0.0    
    #    im2 = gaussian_filter(im1*basis.cent_mask,sigma=(2,2))
    #    #    imshow(im2*basis.cent_mask)
    #    self.res_clean_median = im2*basis.cent_mask
    # 
    #    (izero,jzero) = np.where(self.res_rot_var==0)
    #    im1 = self.res_rot_mean_clip/np.sqrt(self.res_rot_var)
    #    im1[izero,jzero] = 0.0    
    #    im2 = gaussian_filter(im1*basis.cent_mask,sigma=(2,2))
    #    #    imshow(im2*basis.cent_mask)
    #    self.res_clean_mean_clip = im2*basis.cent_mask
       
    
    # def peak_find(self,limit=0.8,printit=False):
    #     """Object for detecting the peaks in a residual map and calculating the signal to noise """
    #     #Code for finding and measuring peaks 
    #     #imtemp = self.res_clean_median
    #     imtemp = self.res_clean_mean_clip
    #     c = pl.contour(imtemp,[imtemp.max()*limit])
    #     p = c.collections[0].get_paths()
    #     num_peaks = np.size(p)
    #     x_peaks = np.zeros(num_peaks)
    #     y_peaks = np.zeros(num_peaks)
    #     h_peaks = np.zeros(num_peaks)
    #     for i in range(0,num_peaks):
    #         x = p[i].vertices[:,0]
    #         y = p[i].vertices[:,1]
    #         x_peaks[i] = x.mean()
    #         y_peaks[i] = y.mean()
    #         h_peaks[i] = imtemp[round(y_peaks[i]),round(x_peaks[i])]
    #     sig = np.sqrt((imtemp**2).sum()/np.size(imtemp.nonzero())/2.) #factor of 2 because size gives a 3 column array
    # 
    #     #sig = sqrt((imtemp**2).sum()/size(imtemp.nonzero())) # Don't understand why there's an extra factor of 2
    # 
    #     self.x_peaks = x_peaks 
    #     self.y_peaks = y_peaks
    #     self.h_peaks = h_peaks
    #     self.sig = sig
    #     self.p_contour = p
    #     self.num_peaks = num_peaks
    #     #print 'Managed to get here!'
    #     
    #     if printit is True:
    #         print 'Number of images found:',num_peaks
    #         print 
    #         print 'im num','|','x','|','y'
    #         print '-----------------------'
    #         for i in range(0,num_peaks):
    #             print i,'|',x_peaks[i],'|',y_peaks[i],'|',h_peaks[i],'|',h_peaks[i]/sig
    # 
	
    # def save(self,file = None):
    #     """ Function for saving the attributes of a particular residual instance. 
    #     Currently this uses HDF5 format
    #     (!!Can probably be made more efficient!!)"""
    # 
    #     if file is None:
    #         print 'Error: You have not given a file name where the data should be saved.'
    #         return
    #     num_half = np.size(self.res_rot[:,0,0])/2
    # 
    #     fres = h5py.File(file,'w')
    #     fres.create_dataset('res_clean_mean_clip', data=self.res_clean_mean_clip)
    #     #fres.create_dataset('res_rot', data=self.res_rot)
    #     fres.create_dataset('res_rot_p1', data=self.res_rot[0:num_half,])
    #     fres.create_dataset('res_rot_p2', data=self.res_rot[num_half:,])
    #     fres.create_dataset('res_rot_var2', data=self.res_rot_var2)
    #     fres.create_dataset('res_arr_p1', data=self.res_arr[0:num_half,])
    #     fres.create_dataset('res_arr_p2', data=self.res_arr[num_half:,])
    #     fres.create_dataset('res_clean_mean', data=self.res_clean_mean)
    #     fres.create_dataset('res_rot_mean', data=self.res_rot_mean)
    #     fres.create_dataset('res_rot_var', data=self.res_rot_var )
    #     fres.create_dataset('res_rot_median', data=self.res_rot_median)
    #     fres.create_dataset('res_rot_mean_clip', data=self.res_rot_mean_clip)
    #     fres.create_dataset('res_clean_median', data=self.res_clean_median)
    #     fres.close()
    # 
    # def restore(self,file):
    #     """Function for restoring a previously saved calculation. The format
    #     used is HDF5 format """
    # 
    #     if file is None:
    #         print 'Error: You have not given a file name where the data should be saved.'
    #         return
    # 
    # 
    #     fres = h5py.File(file,'r')
    # 
    #     self.res_clean_mean_clip = fres['res_clean_mean_clip'].value
    #     temp1 = fres['res_rot_p1'].value
    #     temp2 = fres['res_rot_p2'].value
    #     self.res_rot = vstack((temp1,temp2))
    #     del(temp1)
    #     del(temp2) 
    #     self.res_rot_var2 = fres['res_rot_var2'].value
    #     temp1 = fres['res_arr_p1'].value
    #     temp2 = fres['res_arr_p2'].value
    #     self.res_arr = vstack((temp1,temp2))
    #     del(temp1)
    #     del(temp2) 
    #     self.res_clean_mean = fres['res_clean_mean'].value
    #     self.res_rot_mean = fres['res_rot_mean'].value
    #     self.res_rot_var= fres['res_rot_var'].value
    #     #self.im_arr_mask = fim['im_arr_mask'].value
    #     self.res_rot_median = fres['res_rot_median'].value
    #     self.res_rot_mean_clip = fres['res_rot_mean_clip'].value
    #     self.res_clean_median = fres['res_clean_median'].value
    # 
    #     fres.close()
    # 





