#! /usr/bin/env python

# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

# System imports
#from __future__ import print_function, division, absolute_import, unicode_literals


# External modules

import os

#import extra PynPoint functions:
from _BasePynPoint import base_pynpoint
import _Util
import _Creators


#Basis Class
class basis(base_pynpoint):
    """This class has been prepared to contain everything
    that is needed to create a basis set from a given set of images
	"""
    #from PynPoint.External_routines import external_routines as ext
    #ext = PynPoint.ext
    
    def __init__(self):
        """
        Initialise an instance of the bais class. The result is simple and
        almost empty (in terms of attributes)
        
        """
        self.obj_type = 'PynPoint_basis'
        
    @staticmethod
    def create_wdir(dir_in,**kwargs):#dir_in,ran_sub=False,force_reload=False,prep_data=True,**kwargs):
        """
        Creates an instance from directory. 
        See pynpoint_create_wdir (in PynPoint_creator_funcs.py) for more details.        
        """

        obj = basis()
        _Creators.pynpoint_create_wdir(obj,dir_in,**kwargs)
        basis_save = obj.mk_basis_set()
        return obj

    @staticmethod
    def create_whdf5input(file_in,**kwargs):#file_in,ran_sub=False,prep_data=True,**kwargs)
        """
        Creates an instance from hdf5 file. 
        See pynpoint_create_whdf5input (in PynPoint_creator_funcs.py) for more details.        
        """

        obj = basis()
        _Creators.pynpoint_create_whdf5input(obj,file_in,**kwargs)
        obj.mk_basis_set()
        return obj

        
    @staticmethod
    def create_restore(filename):
        """
        Creates an instance from saved file.
        See pynpoint_create_restore (in PynPoint_creator_funcs.py) for more details.
        """
        
        obj = basis()
        _Creators.pynpoint_create_restore(obj,filename)
        #_Util.restore_data(obj,filename)
        return obj

    @staticmethod
    def create_wfitsfiles(files,**kwargs):
        """
        
        """
        obj = basis()
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
        #del(basis_info_full)
        
        # self.save(filename)        
        # return filename


    def mk_orig(self,ind):
        """Function for reproducing an original input image"""
        if self.cent_remove is True:
            imtemp = (self.im_arr[ind,] + self.im_ave + self.im_arr_mask[ind,]) * self.im_norm[ind]
        else:
            imtemp = (self.im_arr[ind,] + self.im_ave) * self.im_norm[ind]

        return imtemp

 
    def mk_psfmodel(self, num):
        super(basis, self).mk_psfmodel(self, num)