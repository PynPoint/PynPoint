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
#from __future__ import print_function, division, absolute_import, unicode_literals


# External modules

import pylab as pl
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import pyfits
import time

#TODO add pyfits to the requirements
pl.ion()

"""
plotting routines for PynPoint classes images, basis and residuals.
"""

def plt_psf_model(res,ind,num_coeff,returnval=False,savefits=False,mask_nan=True):
    """ 
    plotting the PSF model
    
    :param res: an instance of residual class
    :param ind: index of the image being modeled
    :param num_coeff: number of basis sets to use
    :param returnval: If True the 2D array that was plotted is returned
    
    Example ::
        
        
        pynplot.plt_psf_model(res,6,40)
    
    """
    
    assert res.obj_type in ['PynPoint_residuals'], 'Error: This plot function currently only suports the class res'
    psf_model = res._psf_im(num_coeff)
    im = psf_model[ind]
    if mask_nan is True:
        im[np.where(res.cent_mask == 0.)] = np.nan

    pl.clf()            
    pl.imshow(im,origin='lower',interpolation='nearest')
    pl.title('PSF',size='large')
    pl.colorbar()   

    if not savefits is False:
        hdu = pyfits.PrimaryHDU(im)
        hdu.writeto(savefits,clobber=True)
    
    if returnval is True:
        return im
    
    

def plt_psf_basis(obj,ind,returnval=False,savefits=False,mask_nan=True):
    """
    Plots the basis images used to model the PSF.
    
    :param obj: an instance that has psf_basis attribute (basis or residuals)
    :param ind: index of the basis image to be plotted
    :param returnval: set to True if you want the function to return the 2D array
    
    :return: 2D array of what was plotted (optional) 

    """
    
    im = obj.psf_basis[ind,]
    if mask_nan is True:
        im[np.where(obj.cent_mask == 0.)] = np.nan


    
    pl.clf()
    pl.imshow(im, origin='lower',interpolation='nearest')
    pl.title('PCA',size='large')
    pl.colorbar()   
    if not savefits is False:
        hdu = pyfits.PrimaryHDU(im)
        hdu.writeto(savefits,clobber=True)

    if returnval is True:
        return im
    
#TODO: undefined variable 'im'
def plt_im_arr(obj,ind,returnval=False,savefits=False,mask_nan=True):
    """
    Used to plot the im_arr entry, which the image used by the instance

    :param obj: an instance of images, basis or residual
    :param ind: index of the image to be plotted
    :param returnval: set to True if you want the function to return the 2D array
    
    :return: 2D array of what was plotted (optional) 
        
    """
    #To Do:
    #       Renormalise keyword
    im = obj.im_arr[ind,]

    if mask_nan is True:
        im[np.where(obj.cent_mask == 0.)] = np.nan

    
    pl.clf()
    pl.imshow(im,origin='lower',interpolation='nearest')
    pl.title('image_arr:'+str(ind),size='large')
    pl.colorbar()
    pl.show()
    
    if not savefits is False:
        hdu = pyfits.PrimaryHDU(im)
        hdu.writeto(savefits,clobber=True)
    
    if returnval is True:
        return im



def anim_im_arr(obj,time_gap=0.04,im_range = [0,50]):
    """
    Produces an animation of the im_arr entries, which are the images used in the instance.

    :param obj: An instance of images, basis or residual
    :param time_gap: Pause time between images
    :param im_range: If None (default) then all the images will be used (this could take a long time). Otherwise im_range should be set to the range of indecies (e.g. [100,150])
    
    Example::
    
        pynplot.anim_im_arr(res)
        
    """
    
    pl.clf()
    if not im_range is None:
        assert len(im_range) == 2,'Error: im_range needs to be a two element list, e.g. im_range = [100,200]' 
        im_arr = obj.im_arr[im_range[0]:im_range[1],]
    else:
        im_arr = obj.im_arr

    im_max = im_arr[0,].max()
    im_min = im_arr[0,].min()
    
    num_frames = len(im_arr[:,0,0])

    for i in range(0,num_frames):
        pl.clf()
        pl.imshow(im_arr[i,],animated=True,interpolation='nearest',origin='lower',clim=[im_min,im_max])
        pl.title('Im_arr Images')
        pl.draw()
        time.sleep(time_gap)
    

# from residuals:
 
def plt_res(res,num_coeff,imtype='mean',smooth=None,returnval=False,savefits=False, mask_nan=True,extra_rot=0.0):
    """
    Plots the residual results (either an average or the variance) 
    and gives the image as a return value. 
    
    :param res: An instance of residual class
    :param num_coeff: Number of coefficients used in the fit
    :param imtype: Type of image to plot. Options are: 'mean', 'mean_clip', 'median', 'var', 'sigma' and 'mean_sigma' 
    :param smooth: If None (default) then no smoothing is done, otherwise supply a 2 elements list (e.g. [2,2]). The image will be smoothed with a 2D Gaussian with this sigma_x and sigma_y (in pixel units).
    :param returnval: set to True if you want the function to return the 2D array
    :param savefits: Should be either False (nothing happens) or the name of a fits file where the data should be written
    :param mask_nan: If set to True (default) masked region will be set to np.nan else set to zero

    :return: 2D array of what was plotted (optional) 
    
    Example ::
    
    
       pynplot.plt_res(res,20,imtype='mean',returnval=True)
    
    
    """
    # TODO: renormalise to be close to ADU units
    # TODO: include pixel scale
    # TODO: include north-east arrows
     
    assert res.obj_type =='PynPoint_residuals','Error: This method is for an instance of the residual class'
    
    options = ['mean','mean_clip','median','var','sigma','mean_sigmamean']#,'psf']
    
    assert imtype in options, 'Error: options for ave keyword are %s'%options
    if not smooth is None:
        assert len(smooth) == 2, 'Error: smooth option should be a two element list'
        assert isinstance(smooth[0], ( int, long,float) ), 'Error: smooth keyword should be set to a number.'
        assert isinstance(smooth[1], ( int, long,float) ), 'Error: smooth keyword should be set to a number.'
    assert isinstance(num_coeff, ( int, long,float) ), 'Error: num_basis should be set to a number.'
    
    
    if imtype == 'mean':
        im = res.res_rot_mean(num_coeff,extra_rot = extra_rot)
    elif imtype == 'mean_clip':
        im = res.res_rot_mean_clip(num_coeff,extra_rot = extra_rot)
    elif imtype == 'median':
        im = res.res_rot_median(num_coeff,extra_rot = extra_rot)
    elif imtype == 'var':
        im = res.res_rot_var(num_coeff,extra_rot = extra_rot)
    elif imtype == 'sigma':
        im = np.sqrt(res.res_rot_var(num_coeff,extra_rot = extra_rot))
    elif imtype == 'mean_sigmamean':
        im_sigma = np.sqrt(res.res_rot_var(num_coeff,extra_rot = extra_rot))/np.sqrt(len(res.im_arr[:,0.,0.])) #error on mean
        ind = np.where(im_sigma == 0.0)
        im = (res.res_rot_mean(num_coeff,extra_rot = extra_rot)/im_sigma)* res.cent_mask
        im[ind] = 0.0    
    # else:
    #     print('Error: something is wrong with ave keyword. Funny its not picked up by assert and options!')
    #     return
    if not smooth is None:
        im = gaussian_filter(im,sigma=smooth) * res.cent_mask

    if mask_nan is True:
        im[np.where(res.cent_mask == 0.)] = np.nan
        
           
    pl.figure()
    pl.clf()
    pl.imshow(im,origin='lower',interpolation='nearest')
    pl.title('Residual Image: '+imtype,size='large')
    pl.colorbar()
    
    if not savefits is False:
        hdu = pyfits.PrimaryHDU(im)
        hdu.writeto(savefits,clobber=True)
    
    if returnval is True:
        return im
         
 
    


    



