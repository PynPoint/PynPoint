#! /usr/bin/env python

# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

# System imports
#from __future__ import print_function, division, absolute_import, unicode_literals


# External modules

import pyfits
import pylab as pl
import glob
import time
import h5py
import numpy as np
from scipy import linalg

#import extra PynPoint functions:
import Util
#from PynPoint_creator_funcs import *
from creator_funcs import *
from Mask import mask
#from PynPoint_parent import pynpoint_parent
from parent import pynpoint_parent


#Basis Class
class basis(pynpoint_parent):
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
        pynpoint_create_wdir(obj,dir_in,**kwargs)
        basis_save = obj.mk_basis_set()
        return obj

    @staticmethod
    def create_whdf5input(file_in,**kwargs):#file_in,ran_sub=False,prep_data=True,**kwargs)
        """
        Creates an instance from hdf5 file. 
        See pynpoint_create_whdf5input (in PynPoint_creator_funcs.py) for more details.        
        """

        obj = basis()
        pynpoint_create_whdf5input(obj,file_in,**kwargs)
        obj.mk_basis_set()
        return obj

        
    @staticmethod
    def create_restore(filename):
        """
        Creates an instance from saved file.
        See pynpoint_create_restore (in PynPoint_creator_funcs.py) for more details.
        """
        
        obj = basis()
        pynpoint_create_restore(obj,filename)
        #Util.restore_data(obj,filename)
        return obj

    @staticmethod
    def create_wfitsfiles(files,**kwargs):
        """
        
        """
        obj = basis()
        pynpoint_create_wfitsfiles(obj,files,**kwargs)
        obj.mk_basis_set()
        return obj
        



    def mk_basis_set(self,fileout = None):
        """
        creates basis set using the input images stored in im_arr
        """
        if fileout is None:
            dir_in = os.path.dirname(self.files[0])
            filename = Util.filename4mdir(dir_in,filetype='basis')
        
        basis_info_full = Util.mk_basis_pca(self.im_arr)#,ave_sub=True)
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

        
    def plt_orig(self,ind):
        """Function for making a plot of the original image"""
        pl.clf()
        pl.imshow(self.mk_orig(ind), origin='lower',interpolation='nearest')
        pl.title('Original Image',size='large')
        pl.colorbar()   


    def plt_active(self,ind):
        """Function for making a plot of the working image"""
        pl.clf()
        pl.imshow(self.im_arr[ind,], origin='lower',interpolation='nearest')
        pl.title('Active Image',size='large')
        pl.colorbar()   


    def plt_pca(self,ind):
        """Function for making a plot of the PCA basis set"""
        pl.clf()
        pl.imshow(self.psf_basis[ind,], origin='lower',interpolation='nearest')
        pl.title('PCA',size='large')
        pl.colorbar()   


    # def plt_pcarecon(self,coeff):
    #     """Function for plotting reconstruction"""
    #     pl.clf()
    #     pl.imshow(self.mk_pcarecon(coeff), origin='lower',interpolation='nearest')
    #     pl.title('Reconstruction',size='large')
    #     pl.colorbar()   


    def anim_orig(self,time_gap=0.04,num_frames = False):
        """function for animating the input images"""
        pl.clf()
        if num_frames is False:
            num_frames = self.num_files
        for i in range(0,num_frames):
            pl.clf()
            pl.imshow(self.mk_orig(i),animated=True,interpolation='nearest',origin='lower')
            pl.title('Original Images')
            pl.draw()
            time.sleep(time_gap)


    def anim_active(self,time_gap=0.04,num_frames = False):
        """function for animating the input images"""
        pl.clf()
        if num_frames is False:
            num_frames = self.num_files
        im_max = self.im_arr[0,].max()
        im_min = self.im_arr[0,].min()
        for i in range(0,num_frames):
            pl.clf()
            pl.imshow(self.im_arr[i,],animated=True,interpolation='nearest',origin='lower',clim=[im_min,im_max])
            pl.title('Active Images')
            pl.draw()
            time.sleep(time_gap)
        
 
 
        
#     @staticmethod
#     def create_restore(filename):
#         """
#         Creates an instance of the basis class using a previously
#         saved file. The file should have been stored as an HDF5 file
#         
#         :param filename: name of the inputfile 
#         
#         """
#         obj = basis()
#         Util.restore_data(obj,filename)
#         return obj
#                 
#         
#     @staticmethod
#     def create_wdir(dir_in,ran_sub=False,force_reload=False,prep_data=True,**kwargs):
#         """
#         Creates an instance of the basis class using dir_in, which is the
#         name of a directory containing the input fits files. 
#         As well as its input parameters, this function can pass the keyword
#         options used by prep_data method. 
#         
#         :param dir_in: name of the directory with fits files
#         :param ran_sub: if a number (N) is passed then a random subset of filessize is selected 
#         
#         """
#         
#         ##convert to hdf5
#         file_hdf5 = Util.filename4mdir(dir_in)
#         if (force_reload is True) or (not os.path.isfile(file_hdf5)):
#             Util.conv_dirfits2hdf5(dir_in,outputfile = file_hdf5)
#         obj = basis.create_whdf5input(file_hdf5,ran_sub=False,**kwargs)
#         return obj
#         
#               
#     @staticmethod
#     def create_whdf5input(file_in,ran_sub=False,prep_data=True,**kwargs):
#         """
#         Creates an instance of the basis class using dir_in, which is the
#         name of a directory containing the input fits files. 
#         As well as its input parameters, this function can pass the keyword
#         options used by prep_data method. 
#         
#         :param dir_in: name of the directory with fits files
#         :param ran_sub: if a number (N) is passed then a random subset of filessize is selected 
#         
#         """
#         obj = basis()
#         #obj_type = obj.obj_type
#         Util.restore_data(obj,file_in,checktype='raw_data')
#         obj.obj_type = 'PynPoint_basis' # restate since this is replaced in the restore_data
#         #use the ran_subset keyword
#         if prep_data is True:
#             Util.prep_data(**kwargs)
#                     
#         return obj
#                       
#     
#     @staticmethod
#     def create_wfitsfiles(files,ran_sub=False,prep_data=True,**kwargs):
#         """
#         Creates an instance of the basis class using a list of fits file names. 
#         As well as its input parameters, this function can pass the keyword
#         options used by prep_data method. 
#         
#         :param files: list with with fits file names
#         :param ran_sub: if a number (N) is passed then a random subset of filessize is selected 
#         
#         """
#         obj = PynPoint.basis()
#         
#         # if ran_sub is not False:
#         #     np.random.shuffle(files)#,random.random)
#         #     files = files[0:ran_sub] 
#         num_files = np.size(files)            # number of files
#         # im0 = pyfits.getdata(files[0],0)   # rd first image to initialise
#         # im_size = im0.shape
#         #dir_in = dir_in
#         obj.files = files
#         obj.num_files = num_files
#         # self.im_size = im_size
#         # self.dir_in = dir_in
# #        self.cent_remove = cent_remove
#         Util.rd_fits(obj,avesub=False,para_sort=para_sort,inner_pix=inner_pix)
#         if prep_data is True:
#             Util.prep_data(**kwargs)
#         
#         
#         
#         # dir_in,recent=False,resize=True,cent_remove=True,F_int=10,F_final=2,ran_sub=False,para_sort=True,inner_pix=False,cent_size=0.2,edge_size=1.0,intype='dir'):
#               
# 
#     
#     def __init__(self,dir_in,recent=False,resize=True,cent_remove=True,F_int=10,F_final=2,ran_sub=False,para_sort=True,inner_pix=False,cent_size=0.2,edge_size=1.0,intype='dir'):
#         """Initialise some of the global properties of the data"""
#         self.obj_type = 'PynPoint_basis'
#         print 
#         print 'Update: Reading data ...'
#         if intype == 'empty':
#             return
#                     
#         elif intype == 'restore':
#             print 'Restoring data from file:'
#             print dir_in
#             print 'All other keywords are being ignored.'
#             self.restore(dir_in)
#             return
# 
#         elif intype == 'dir':
#             files = glob.glob(dir_in+'*.fits') # list of file name
#             self.input_from_fits(files,recent=recent,resize=resize,cent_remove=cent_remove,F_int=F_int,F_final=F_final,ran_sub=ran_sub,para_sort=para_sort,inner_pix=inner_pix,cent_size=cent_size,edge_size=edge_size)
#         elif intype == 'files':
#             files = dir_in
#             self.input_from_fits(files,recent=recent,resize=resize,cent_remove=cent_remove,F_int=F_int,F_final=F_final,ran_sub=ran_sub,para_sort=para_sort,inner_pix=inner_pix,cent_size=cent_size,edge_size=edge_size)
#         elif intype == 'HDF5':
#             if ran_sub is not False:
#                 print 'Warning: ran_sub (and many other) keyword not used'
#             self.input_from_HDF5(dir_in)
#         else:
#             print 'Error: Input format is not recognised'
#         print 'Update: Subtracting Average Image ...'
#         #self.mk_avesub()
#         print 'Update: Calculating PCAs ...'
#         self.mk_pca()
# 
#     def input_from_fits(self,files,recent=True,resize=True,cent_remove=True,F_int=10,F_final=2,ran_sub=False,para_sort=True,inner_pix=False,cent_size=0.2,edge_size=1.0):
#         """Function for inputing data from fits files 
#         (!!! might want to make this s function that is called by images AND basis!!!) """
#         if ran_sub is not False:
#             np.random.shuffle(files)#,np.random.random)
#             files = files[0:ran_sub] 
# 
#         num_files = np.size(files)            # number of files
#         im0 = pyfits.getdata(files[0],0)   # rd first image to initialise
#         im_size = im0.shape
#         #dir_in = dir_in
#         self.files = files
#         self.num_files = num_files
#         self.im_size = im_size
#         #self.dir_in = dir_in
#         self.cent_remove = cent_remove
#         Util.rd_fits(self,para_sort=para_sort,inner_pix=inner_pix)
# 
#         if resize is True and recent is True:
#             self.im_arr = Util.mk_resizerecent(self.im_arr,F_int,F_final)
#             self.im_size = self.im_arr[0,].shape # need to rework into a more elegent solution
#         elif resize is True:
#             self.im_arr = Util.mk_resizeonly(self.im_arr,F_final)
#             self.im_size = self.im_arr[0,].shape # need to rework into a more elegent solution
#         if cent_remove is True:
#             mask1 = mask(self.im_arr[0,].shape[0],self.im_arr[0,].shape[1])
#             im_arr_omask,im_arr_imask,cent_mask = mask1.mk_cent_remove(self.im_arr,cent_size=cent_size,edge_size=edge_size)
#             self.im_arr = im_arr_omask
#             self.im_arr_mask = im_arr_imask
#             self.cent_mask = cent_mask
#         else:
#             self.cent_mask = None
#         self.mk_avesub()
# 
#  
#     def input_from_HDF5(self,file):
#         """Function for inputing data from HDF5 file. For example the one produced by the save function of images class.
#         (!!! might want to make this s function that is called by images AND basis!!!) """
#         fhdf5 = h5py.File(file,'r')
#         self.files = fhdf5['files'].value.tolist()
#         self.num_files = fhdf5['num_files'].value
#         self.im_size = tuple(fhdf5['im_size'].value.tolist())
#         #self.dir_in = fhdf5['dir_in'].value
#         self.cent_remove = fhdf5['cent_remove'].value
#         #temp1 = fhdf5['im_arr_p1'].value
#         #temp2 = fhdf5['im_arr_p2'].value
#         self.im_arr = fhdf5['im_arr'].value#np.vstack((temp1,temp2))
#         #del(temp1)
#         #del(temp2) 
#         self.im_norm = fhdf5['im_norm'].value
#         self.para = fhdf5['para'].value
#         self.cent_mask = fhdf5['cent_mask'].value
#         self.im_ave = fhdf5['im_ave'].value
#         fhdf5.close()
# 
#     def mk_avesub(self):
#         """Function for subtracting the mean image from a Stack"""
#         im_ave = self.im_arr.sum(axis = 0)/self.num_files
#         for i in range(0,self.num_files):
#             self.im_arr[i,] = self.im_arr[i,] - im_ave
#         self.im_ave = im_ave
# 
       
        
        
    # def mk_pca(self) :
    #     """Function for creating the set of PCA's for a stack
    #     of images"""
    #     
    #     U,s,V = linalg.svd(self.im_arr.reshape(self.num_files,self.im_size[0]*self.im_size[1]),full_matrices=False)        
    #     self.basis_pca = V.reshape(V.shape[0],self.im_size[0],self.im_size[1])
    # 

    # def mk_pcafit_one(self,im,num=10,meansub=True):
    #     """Function for fitting the PCA coefficients """
    # 
    #     if im.ndim != 2:
    #         print 'There is a problem with the image dimensions'
    #         return None
    #     if im.shape[0] != self.im_size[0] :
    #         print 'image dimension does not match that of basis'            
    #     if im.shape[1] != self.im_size[1] :
    #         print 'image dimension does not match that of basis'    
    # 
    #     coeff = np.zeros(num)
    #     if meansub is False:
    #         im_temp = im
    #     elif meansub is True:
    #         im_temp = im - self.im_ave
    #     else:
    #         print 'Error: meansub keyword should be True or False only'
    #         return None
    # 
    #     if self.cent_remove is True:
    #         im_temp = im_temp*self.cent_mask
    #         
    #     for i in range(0,num):
    #         coeff[i] = (im_temp * self.basis_pca[i,]).sum()
    #     return coeff
    # 
    #     
    # def mk_pcarecon(self,coeff,meanadd=True):
    #     """Function for creating a reconstruction given an input set of coefficients"""
    #     im = np.zeros(self.im_size)
    #     for i in range(0,coeff.size):
    #         im = im + (self.basis_pca[i,]*coeff[i])
    # 
    #     if meanadd is False:
    #         im_temp = im
    #     elif meanadd is True:
    #         im_temp = im + self.im_ave
    #     else:
    #         print 'Error: meansub keyword should be True or False only'
    #         return None
    #     if self.cent_remove is True:
    #         im_temp = im_temp *self.cent_mask
    #     return im_temp
    # 
    #     

    # def save(self,file = None):
    #     """ Object for saving the attributes of a particular image instance. 
    #     Currently this uses HDF5 format
    #     (!!Can probably be made more efficient!!)"""
    # 
    #     if file is None:
    #         print 'Error: You have not given a file name where the data should be saved.'
    #         return
    # 
    #     fbasis = h5py.File(file,'w')
    #     #num_half = self.num_files/2
    # 
    #     fbasis.create_dataset('files',data=self.files)
    #     fbasis.create_dataset('im_arr',data=self.im_arr,maxshape=(None,None,None))
    #     fbasis.create_dataset('para',data=self.para)
    #     fbasis.create_dataset('im_norm',data=self.im_norm)
    #     fbasis.create_dataset('cent_mask',data=self.cent_mask)
    #     #fim.create_dataset('im_arr_mask',data=self.im_arr_mask) # may not want to save this!!
    #     #fbasis.create_dataset('dir_in',data=self.dir_in)
    #     fbasis.create_dataset('num_files',data=self.num_files)
    #     fbasis.create_dataset('im_size',data=self.im_size)
    #     fbasis.create_dataset('cent_remove',data=self.cent_remove)
    #     fbasis.create_dataset('basis_pca',data=self.basis_pca,maxshape=(None,None,None))
    #     #fbasis.create_dataset('basis_pca_p1',data=self.basis_pca[0:num_half,])
    #     #fbasis.create_dataset('basis_pca_p2',data=self.basis_pca[num_half:,])
    # 
    #     fbasis.create_dataset('im_ave',data=self.im_ave)
    #     fbasis.close()

    # def restore(self,file):
    #     """ Restore function for reading images data. The format is HDF5 format."""
    #     if file is None:
    #         print 'Error: You have not given a file name where the data should be saved.'
    #         return
    #     fbasis = h5py.File(file,'r')
    # 
    #     self.files = fbasis['files'].value
    #     #self.im_arr = fim['im_arr'].value
    #     self.para = fbasis['para'].value
    #     self.im_norm = fbasis['im_norm'].value
    #     self.num_files = fbasis['num_files'].value
    #     self.cent_mask = fbasis['cent_mask'].value
    #     #self.im_arr_mask = fim['im_arr_mask'].value
    #     #self.dir_in = fbasis['dir_in'].value
    #     self.im_size = fbasis['im_size'].value
    #     self.cent_remove = fbasis['cent_remove'].value
    #     self.basis_pca = fbasis['basis_pca'].value
    #     #self.im_arr = fim['im_arr'].value
    #     #temp1 = fbasis['basis_pca_p1'].value
    #     #temp2 = fbasis['basis_pca_p2'].value
    #     #self.basis_pca = np.vstack((temp1,temp2))
    #     #del(temp1)
    #     #del(temp2) 
    #     self.im_ave = fbasis['im_ave'].value
    # 
    #     fbasis.close()

