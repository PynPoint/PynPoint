#! /usr/bin/env python

# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

# System imports
#from __future__ import print_function, division, absolute_import, unicode_literals


# External modules

import pylab as pl
from scipy.ndimage.filters import gaussian_filter

#import extra PynPoint functions:
# from _BasePynPoint import base_pynpoint
# import _Util
# import _Creators


#Basis Class
#class basis(base_pynpoint):
"""
plotting routines for PynPoint classes images, basis and residualts.
"""

# def __init__(self):
#     """
#     Initialise an instance of the images class. The result is simple and
#     almost empty (in terms of attributes)        
#     """
#     
#     self.obj_type = 'PynPoint_plotter'
#     

def plt_psf_fit(self,obj,ind,full=False):
    """function for plotting the PSF model"""
    pl.clf()            
    pl.imshow(obj.mk_psf_realisation(ind,full=full),origin='lower',interpolation='nearest')
    pl.title('PSF',size='large')
    pl.colorbar()   



def plt_psf_basis(obj,ind,returnval=False):
    """
    Plots the basis images used to model the PSF.
    
    :param obj: an instance that has psf_basis instance
    :param ind: index of the basis image to be plotted
    :param returnval: set to True if you want the function to return the 2D array
    
    :return: 2D array of what was plotted (optional) 

    """
    
    pl.clf()
    pl.imshow(obj.psf_basis[ind,], origin='lower',interpolation='nearest')
    pl.title('PCA',size='large')
    pl.colorbar()   
    
def plt_im_arr(obj,ind,returnval=False):
    """
    Used to plot the im_arr entry, which the image used by the instance

    :param obj: an instance of images, basis or residual
    :param ind: index of the image to be plotted
    :param returnval: set to True if you want the function to return the 2D array
    
    :return: 2D array of what was plotted (optional) 
        
    """
    pl.clf()
    pl.imshow(obj.im_arr[ind,],origin='lower',interpolation='nearest')
    pl.title('image_arr:'+str(ind),size='large')
    pl.colorbar()
    pl.show()
    
    if returnval is True:
         return im



def anim_im_arr(obj,time_gap=0.04,im_range = None):
    """
    Produces an animation of the im_arr entries, which are the images used in the instance.

    :param obj: An instance of images, basis or residual
    :param time_gap: Pause time between images
    :param im_range: If None (default) then all the images will be used (this could take a long time). Otherwise im_range should be set to the range of indecies (e.g. [100,150])
        
    """
    
    pl.clf()
    if not im_range is None:
        assert len(im_range) == 2,'Error: im_range needs to be a two element list, e.g. im_range = [100,200]' 
        im_arr = obj.im_arr[im_range[0]:im_range[1],]
    else:
        im_arr = obj.im_arr

    im_max = im_arr[0,].max()
    im_min = im_arr[0,].min()

    for i in range(0,num_frames):
        pl.clf()
        pl.imshow(im_arr[i,],animated=True,interpolation='nearest',origin='lower',clim=[im_min,im_max])
        pl.title('Im_arr Images')
        pl.draw()
        time.sleep(time_gap)
    

 # from residuals:
 
def plt_res(res,num_coeff,imtype='mean',smooth=None,returnval=False):
     """
     Plots the residual results (either an average or the variance) 
     and gives the image as a return value. 
     
    :param res: An instance of residual class
    :param num_coeff: Number of coefficients used in the fit
    :param imtype: Type of image to plot. Options are: 'mean', 'mean_clip', 'median' and 'var' 
    :param smooth: If None (default) then no smoothing is done, otherwise supply a 2 elements list (e.g. [2,2]). The image will be smoothed with a 2D Gaussian with this sigma_x and sigma_y (in pixel units).
    :param returnval: set to True if you want the function to return the 2D array
    
    :return: 2D array of what was plotted (optional) 
     
     
     """
     
     assert res.obj_type =='PynPoint_residuals','Error: This method is for an instance of the residual class'
     
     options = ['mean','mean_clip','median','var']#,'psf']
     
     assert imtype in options, 'Error: options for ave keyword are %s'%options
     if not smooth is None:
         assert len(smooth) == 2, 'Error: smooth option should be a two element list'
         assert isinstance(smooth[0], ( int, long,float) ), 'Error: smooth keyword should be set to a number.'
         assert isinstance(smooth[1], ( int, long,float) ), 'Error: smooth keyword should be set to a number.'
     assert isinstance(num_coeff, ( int, long,float) ), 'Error: num_basis should be set to a number.'
     
     
     if imtype == 'mean':
         im = res.res_rot_mean(num_coeff)                
     elif imtype == 'mean_clip':
         im = res.res_rot_mean_clip(num_coeff)
     elif imtype == 'median':
         im = res.res_rot_median(num_coeff)
     elif imtype == 'var':
         im = res.res_rot_var(num_coeff)
     else:
         print('Error: something is wrong with ave keyword. Funny its not picked up by assert and options!')
         return
     if not smooth is None:
        im = gaussian_filter(im,sigma=smooth) * res.cent_mask
         
            
     # print('ADAM: ')
     # print(im)
     # print(imtype)
     # print(smooth)
     # if im is None:
     #     return
     pl.figure()
     pl.clf()
     pl.imshow(im,origin='lower',interpolation='nearest')
     pl.title('Residual Image: '+imtype,size='large')
     pl.colorbar()
     if returnval is True:
         return im
         
 
    


    



