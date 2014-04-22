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

#import external functions:

#import extra PynPoint functions:
from PynPoint._BasePynPoint import base_pynpoint
from PynPoint import _Creators


#Images class:
class images(base_pynpoint):
    """
    
    Deals with the postage stamp images that will be analysed. An instance of images can be 
    created in a number of ways. Inputs can be either fits files or hdf5 files. Once read in
    the data will be processed according to the user specified keyword options.
    
    Once created the instance of images can be saved using its save method. This can later be 
    restored using the restore method.
    
    """
    
    def __init__(self):
        """
        Initialise an instance of the images class.  
        """
        self.obj_type = 'PynPoint_images'
        
        
    @classmethod
    def create_wdir(cls, dir_in,**kwargs): #dir_in,ran_sub=False,force_reload=False,prep_data=True,**kwargs):
        """
        Creates an instance from directory. 
        See :py:func:`_Creators.pynpoint_create_wdir` for more details.
        """

        obj = cls()
        _Creators.pynpoint_create_wdir(obj,dir_in,**kwargs)
        return obj

    @classmethod
    def create_whdf5input(cls, file_in,**kwargs):#file_in,ran_sub=False,prep_data=True,**kwargs)
        """
        Creates an instance from hdf5 file. 
        See :py:func:`_Creators.pynpoint_create_whdf5input` for more details.
        """

        obj = cls()
        _Creators.pynpoint_create_whdf5input(obj,file_in,**kwargs)
        return obj

        
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
    def create_wfitsfiles(cls, *args,**kwargs):
        """
        Creates an instance from fits files. 
        See :py:func:`_Creators.pynpoint_create_wfitsfiles` for more details.
        """
        
        obj = cls()
        _Creators.pynpoint_create_wfitsfiles(obj,*args,**kwargs)
        return obj
        

    def mk_psf_realisation(self,ind,full=False):    
        """Function for making a realisation of the PSF using the data stored in the object"""
        im_temp = self.psf_im_arr[ind,] 
        if self.cent_remove is True:
            if full is True:
                im_temp = im_temp + self.im_arr_mask[ind,]
            elif full is False:
                im_temp = im_temp *self.cent_mask
        return im_temp

