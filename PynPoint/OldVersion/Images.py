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

from PynPoint.OldVersion._BasePynPoint import base_pynpoint

from PynPoint.OldVersion import _Creators


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
        Creates an instance of the images class.

        :param dir_in: name of the directory with fits files
        :param kwargs: accepts the same keyword options as :py:func:`PynPoint.Basis.basis.create_wdir`
        :return: instance of the images class


        """

        obj = cls()
        _Creators.pynpoint_create_wdir(obj, dir_in, **kwargs)
        return obj

    @classmethod
    def create_whdf5input(cls, file_in,**kwargs):#file_in,ran_sub=False,prep_data=True,**kwargs)
        """
        Creates an instance from hdf5 file.

        :param file_in: path to the hdf5 file containing the images
        :param kwargs: accepts the same keyword options as :py:func:`PynPoint.Basis.basis.create_wdir`


        """

        obj = cls()
        _Creators.pynpoint_create_whdf5input(obj, file_in, **kwargs)
        return obj

        
    @classmethod
    def create_restore(cls, filename):
        """
        Restores data from a hdf5 file previously created using the save method of a images instance.

        :param filename: name of the inputfile
        :return: Instance of the images class

        """

        obj = cls()
        _Creators.restore(obj, filename)
        return obj

    @classmethod
    def create_wfitsfiles(cls, *args,**kwargs):
        """
        Creates an instance of images class from a list of fits files.

        :param files: list of fits files
        :param kwargs: accepts the same keyword options as :py:func:`PynPoint.Basis.basis.create_wdir`
        :return: instance of the images class



        """

        obj = cls()
        _Creators.pynpoint_create_wfitsfiles(obj, *args, **kwargs)
        return obj
        

    def mk_psf_realisation(self,ind,full=False):
        """
        Function for making a realisation of the PSF using the data stored in the object

        :param ind: index of the image to be modelled
        :param full: if set to True then the masked region will be included
        :return: an image of the PSF model
        """
        im_temp = self.psf_im_arr[ind,] 
        if self.cent_remove is True:
            if full is True:
                im_temp = im_temp + self.im_arr_mask[ind,]
            elif full is False:
                im_temp = im_temp *self.cent_mask
        return im_temp

