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
import os

#import extra PynPoint functions:
from PynPoint._BasePynPoint import base_pynpoint
from PynPoint import _Creators
from PynPoint import _Util


#Basis Class
class basis(base_pynpoint):
    """This class has been prepared to contain everything
    that is needed to create a basis set from a given set of images
	"""
    
    def __init__(self):
        """
        Initialise an instance of the bais class.
        """
        self.obj_type = 'PynPoint_basis'
        
    @classmethod
    def create_wdir(cls, dir_in,**kwargs):#dir_in,ran_sub=False,force_reload=False,prep_data=True,**kwargs):
        """
        Creates an instance from directory. 
        See :py:func:`_Creators.pynpoint_create_wdir` for more details.
        """

        obj = cls()
        _Creators.pynpoint_create_wdir(obj,dir_in,**kwargs)
        basis_save = obj.mk_basis_set()
        return obj

    @classmethod
    def create_whdf5input(cls, file_in,**kwargs):#file_in,ran_sub=False,prep_data=True,**kwargs)
        """
        Creates an instance from hdf5 file. 
        See :py:func:`_Creators.pynpoint_create_whdf5input` for more details.
        """

        obj = cls()
        _Creators.pynpoint_create_whdf5input(obj,file_in,**kwargs)
        obj.mk_basis_set()
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
    def create_wfitsfiles(cls, files,**kwargs):
        """
        Creates an instance from fits files. 
        See :py:func:`_Creators.pynpoint_create_wfitsfiles` for more details.
        """
        
        obj = cls()
        _Creators.pynpoint_create_wfitsfiles(obj,files,**kwargs)
        obj.mk_basis_set()
        return obj
        



    def mk_basis_set(self,fileout = None):
        """
        creates basis set using the input images stored in im_arr
        """
        if fileout is None:
            dir_in = os.path.dirname(self.files[0])
            filename = _Util.filename4mdir(dir_in,filetype='basis')
        
        basis_info_full = _Util.mk_basis_pca(self.im_arr)#,ave_sub=True)
        self.im_ave = basis_info_full['im_ave']
        self.im_arr = basis_info_full['im_arr']
        self.psf_basis = basis_info_full['basis']
        self.psf_basis_type = basis_info_full['basis_type']


    def mk_orig(self,ind):
        """Function for reproducing an original input image"""
        if self.cent_remove is True:
            imtemp = (self.im_arr[ind,] + self.im_ave + self.im_arr_mask[ind,]) * self.im_norm[ind]
        else:
            imtemp = (self.im_arr[ind,] + self.im_ave) * self.im_norm[ind]

        return imtemp

 
    def mk_psfmodel(self, num):
        super(basis, self).mk_psfmodel(self, num)