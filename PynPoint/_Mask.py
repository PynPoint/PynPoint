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

import numpy as np

# TODO: remove circular dependency with _Util

def mk_circle(xnum,ynum,xcent,ycent,rad_lim):
#def mk_circle(self,xnum,ynum,xcent,ycent,rad_lim):
    """function for making a circular aperture"""
    from PynPoint import _Util
    Y,X = np.indices([xnum,ynum]) #seems strange and backwards, check!
    rad = _Util.mk_circle(xcent,ycent)(X,Y)
    i,j = np.where(rad <= rad_lim)
    mask_base = np.ones((xnum,ynum),float) #something strange about the order of x and y!
    mask_base[i,j] = 0.0
    return mask_base

def mk_cent_remove(im_arr,cent_size=0.2,edge_size=1.0):
    """This function has been written to mask out the central region (and the corners)"""
    # WOULD BE NICE TO INCLUDE AN OPTION FOR EITHER TOP-HAT CIRCLE OR GAUSSIAN
    im_size = im_arr[0,].shape
#         print(im_size)
    mask_c = mk_circle(im_size[0],im_size[1],im_size[0]/2.,im_size[1]/2.,cent_size*im_size[0])
    mask_outside = mk_circle(im_size[0],im_size[1],im_size[0]/2.,im_size[1]/2.,0.5*im_size[0])
    #mask_c = self.mask(im_size[0],im_size[1],fsize=cent_size,fxcent=0.5,fycent=0.5)
    # NEED TO DECIDE IF I WANT TO KEEP THE CORNERS:
    #mask_outside = self.mask(im_size[0],im_size[1],fsize=edge_size,fxcent=0.5,fycent=0.5)
    
    cent_mask = mask_c * (1.0 - mask_outside)
    res_cent_mask = (1.0 - cent_mask)
    im_arr_imask = im_arr * res_cent_mask
    im_arr_omask = im_arr * cent_mask

    return im_arr_omask,im_arr_imask,cent_mask

