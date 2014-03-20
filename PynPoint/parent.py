        
#import external functions:
import pyfits
import pylab as pl
import glob
import time
import h5py
import numpy as np
import os

#import extra PynPoint functions:
import Util
from Mask import mask
#import PynPoint_v1_5 as PynPoint


#Images class:
class pynpoint_parent:
    """Object for dealing with the images that need to be analysed"""
    
    def __init__(self):
        """
        Initialise an instance of the images class. The result is simple and
        almost empty (in terms of attributes)
        
        """
        self.obj_type = 'PynPoint_parent'
            
    def save(self,filename):
        """ Object for saving the attributes of a particular image instance. 
        Currently this uses HDF5 format
        (!!Can probably be made more efficient!!)"""
        # print('!!!!!!!!!!!HI!!!!!!!!!!')
        # print(filename)

        Util.save_data(self,filename)

    def plt_im(self,ind):
        """Function for plotting the input images"""
        pl.clf()
        pl.imshow(self.im_arr[ind,],origin='lower',interpolation='nearest')
        pl.title('Image',size='large')
        pl.colorbar()


    def anim_im(self,time_gap=0.04,num_frames = False):
        """function for animating the input images"""
        pl.clf()
        if num_frames is False:
            num_frames = self.num_files
        for i in range(0,num_frames):
            pl.clf()
            pl.imshow(self.im_arr[i,],animated=True,interpolation='nearest',origin='lower')
            pl.title('Original Images')
            pl.draw()
            time.sleep(time_gap)
            
    def mk_psfmodel(self,basis,num):#,mask=None):
        """
        Improved function for making models of the PSF given an input basis object. 
        The idea is to rely more heavily on matrix manipulations. Initially only 
        valid for mask=None case
        """

        temp_im_arr = np.zeros([self.im_arr.shape[0],self.im_arr.shape[1]*self.im_arr.shape[2]])

        if not hasattr(self,'_have_psf_coeffs'):# To speed things up, I plan to calculate the coeffients only once
            for i in range(0,self.im_arr.shape[0]): # Remove the mean used to build the basis. Might be able to speed this up
                temp_im_arr[i,] = self.im_arr[i,].reshape(-1) - basis.im_ave.reshape(-1)

            # use matrix multiplication
            coeff_temp = np.array((np.mat(temp_im_arr) * np.mat(basis.psf_basis.reshape(basis.psf_basis.shape[0],-1)).T)) 
            self._psf_coeff = coeff_temp # attach the full list of coeffients to input object
            self._have_psf_coeffs = True

        psf_im = (np.mat(self._psf_coeff[:,0:num]) * np.mat(basis.psf_basis.reshape(basis.psf_basis.shape[0],-1)[0:num,]))
        for i in range(0,self.im_arr.shape[0]): # Add the mean back to the image
            psf_im[i,] += basis.im_ave.reshape(-1)

        self.psf_im_arr = np.array(psf_im).reshape(self.im_arr.shape[0],self.im_arr.shape[1],self.im_arr.shape[2])


        
        
    # @staticmethod
    # def create_restore(filename):
    #     """
    #     Creates an instance of the images class using a previously
    #     saved file. The file should have been stored as an HDF5 file
    #     
    #     :param filename: name of the inputfile 
    #     
    #     """
    #     obj = pynpoint_parent()
    #     Util.restore_data(obj,filename)
    #     return obj
    #     
    # @staticmethod
    # def create_wdir(dir_in,ran_sub=False,force_reload=False,prep_data=True,**kwargs):
    #     """
    #     Creates an instance of the images class using dir_in, which is the
    #     name of a directory containing the input fits files. 
    #     As well as its input parameters, this function can pass the keyword
    #     options used by prep_data method. 
    #     
    #     :param dir_in: name of the directory with fits files
    #     :param ran_sub: if a number (N) is passed then a random subset of filessize is selected 
    #     
    #     """
    #     
    #     ##convert to hdf5
    #     file_hdf5 = Util.filename4mdir(dir_in)
    #     if (force_reload is True) or (not os.path.isfile(file_hdf5)):
    #         Util.conv_dirfits2hdf5(dir_in,outputfile = file_hdf5)
    #     obj = pynpoint_parent.create_whdf5input(file_hdf5,ran_sub=False,**kwargs)
    #     return obj
    #           
    # @staticmethod
    # def create_wdir2(obj,dir_in,ran_sub=False,force_reload=False,prep_data=True,**kwargs):
    #     """
    #     Creates an instance of the images class using dir_in, which is the
    #     name of a directory containing the input fits files. 
    #     As well as its input parameters, this function can pass the keyword
    #     options used by prep_data method. 
    #     
    #     :param dir_in: name of the directory with fits files
    #     :param ran_sub: if a number (N) is passed then a random subset of filessize is selected 
    #     
    #     """
    #     
    #     ##convert to hdf5
    #     file_hdf5 = Util.filename4mdir(dir_in)
    #     if (force_reload is True) or (not os.path.isfile(file_hdf5)):
    #         Util.conv_dirfits2hdf5(dir_in,outputfile = file_hdf5)
    #     # obj = pynpoint_parent.create_whdf5input(file_hdf5,ran_sub=False,**kwargs)
    #     obj.pp_parent.create_whdf5input2(obj,file_hdf5,ran_sub=False,**kwargs)
    #     return obj
    # 
    # @staticmethod
    # def create_whdf5input2(obj,file_in,ran_sub=False,prep_data=True,**kwargs):
    #     """
    #     Creates an instance of the images class using dir_in, which is the
    #     name of a directory containing the input fits files. 
    #     As well as its input parameters, this function can pass the keyword
    #     options used by prep_data method. 
    #     
    #     :param dir_in: name of the directory with fits files
    #     :param ran_sub: if a number (N) is passed then a random subset of filessize is selected 
    #     
    #     """
    #     #pynpoint_parent.__init__(obj)
    #     #obj = pynpoint_parent()
    #     #obj_type = obj.obj_type
    #     Util.restore_data(obj,file_in,checktype='raw_data')
    #     obj.obj_type = 'PynPoint_images' # restate since this is replaced in the restore_data
    #     #use the ran_subset keyword
    #     if prep_data is True:
    #         Util.prep_data(obj,**kwargs)                    
    # 
    #     return obj
    #                   
    #              
    # @staticmethod
    # def create_whdf5input(file_in,ran_sub=False,prep_data=True,**kwargs):
    #     """
    #     Creates an instance of the images class using dir_in, which is the
    #     name of a directory containing the input fits files. 
    #     As well as its input parameters, this function can pass the keyword
    #     options used by prep_data method. 
    #     
    #     :param dir_in: name of the directory with fits files
    #     :param ran_sub: if a number (N) is passed then a random subset of filessize is selected 
    #     
    #     """
    #     #pynpoint_parent.__init__(obj)
    #     obj = pynpoint_parent()
    #     #obj_type = obj.obj_type
    #     Util.restore_data(obj,file_in,checktype='raw_data')
    #     obj.obj_type = 'PynPoint_images' # restate since this is replaced in the restore_data
    #     #use the ran_subset keyword
    #     if prep_data is True:
    #         Util.prep_data(obj,**kwargs)
    #                 
    #     return obj
    #                   
    # 
    # @staticmethod
    # def create_wfitsfiles(files,ran_sub=False,prep_data=True,**kwargs):
    #     """
    #     Creates an instance of the images class using a list of fits file names. 
    #     As well as its input parameters, this function can pass the keyword
    #     options used by prep_data method. 
    #     
    #     :param files: list with with fits file names
    #     :param ran_sub: if a number (N) is passed then a random subset of filessize is selected 
    #     
    #     """
    #     obj = pynpoint_parent()
    #     num_files = np.size(files)            # number of files
    #     obj.files = files
    #     obj.num_files = num_files
    #     Util.rd_fits(obj,avesub=False,para_sort=para_sort,inner_pix=inner_pix)
    #     if prep_data is True:
    #         obj.prep_data(**kwargs)

    # def anim_active(self,time_gap=0.04,num_frames = False):
    #     """function for animating the input images"""
    #     pl.clf()
    #     temp_im = self.im_arr[0,] - self.mk_psf_realisation(0,full=False)
    #     if num_frames is False:
    #         num_frames = self.num_files
    #     im_max = temp_im.max()
    #     im_min = temp_im.min()
    #     for i in range(0,num_frames):
    #         pl.clf()
    #         #            plt_resid(i)
    #         temp_im = self.im_arr[i,] - self.mk_psf_realisation(i,full=False)
    #         pl.imshow(temp_im,animated=True,interpolation='nearest',origin='lower',clim=[im_min,im_max])
    #         pl.title('Active Images')
    #         pl.draw()
    #         time.sleep(time_gap)

        
    # 
    # def prep_data(self,recent=False,resize=True,cent_remove=True,F_int=4,
    #               F_final=2,ran_sub=False,para_sort=True,inner_pix=False,
    #               cent_size=0.2,edge_size=1.0):
    #     """
    #     Prepares the data according to set of key word inputs. This is 
    #     usually done when an instance is created from fits files. The 
    #     results of this step are attached to self.
    #     
    #     :param recent:
    #     :param resize:
    #     :param cent_remove:
    #     :param F_int:
    #     :param F_final:
    #     :param ran_sub:
    #     :param para_sort:
    #     :param inner_pix:
    #     :param cent_size:
    #     :param edge_size:
    #     
    #     """        
    #     #Normalise the images so that they have 'unit area'
    #     self.cent_remove = cent_remove
    #     #self.im_norm = np.zeros(self.im_arr.shape[0])
    #     self.im_norm = (self.im_arr.sum(axis = 1)).sum(axis = 1)
    #     #self.im_arr = self.im_arr.T.dot(np.diag(1./self.im_norm)).T
    #     for i in range(0,len(self.im_arr[:,0,0])):
    #         self.im_arr[i,] /= self.im_norm[i]
    #         
    #     
    #     # im_arr[i,] = im_temp/im_norm[i] # rd all the images
    #     
    #     
    #     if resize is True and recent is True:
    #         print 'Update: Resizing and recentring ...'
    #         self.im_arr = Util.mk_resizerecent(self.im_arr,F_int,F_final)
    #         self.im_size = self.im_arr[0,].shape # need to rework into a more elegent solution
    #     elif resize is True:
    #         print 'Update: Resizing ...'
    #         self.im_arr = Util.mk_resizeonly(self.im_arr,F_final)
    #         self.im_size = self.im_arr[0,].shape # need to rework into a more elegent solution
    #     
    #     
    #     if cent_remove is True:
    #         mask1 = mask(self.im_arr[0,].shape[0],self.im_arr[0,].shape[1])
    #         im_arr_omask,im_arr_imask,cent_mask = mask1.mk_cent_remove(self.im_arr,cent_size=cent_size,edge_size=edge_size)
    #         self.im_arr = im_arr_omask
    #         self.im_arr_mask = im_arr_imask
    #         self.cent_mask = cent_mask        
    # 
    #     else:
    #         self.cent_mask = None
    # 

    # def mk_psfmodel2(self,basis,num,mask=None):
    #     """
    #     Improved function for making models of the PSF given an input basis object. 
    #     The idea is to rely more heavily on matrix manipulations. Initially only 
    #     valid for mask=None case
    #     """
    # 
    #     temp_im_arr = np.zeros([self.im_arr.shape[0],self.im_arr.shape[1]*self.im_arr.shape[2]])
    #     if mask is not None: #This version is still in development
    #         print 'Error: This version of mk_psfmodel2 does not yet support a mask'
    #         print 'You need to make sure you use mk_psfmodel for now.'
    #         return None
    # 
    #     if not hasattr(self,'psf_coeff_new'):# To speed things up, I plan to calculate the coeffients only once
    #         for i in range(0,self.im_arr.shape[0]): # Remove the mean used to build the basis. Might be able to speed this up
    #             temp_im_arr[i,] = self.im_arr[i,].reshape(-1) - basis.im_ave.reshape(-1)
    #         #coeff_temp = np.array((np.mat(temp_im_arr) * np.mat(basis.basis_pca.reshape(self.im_arr.shape[0],-1)).T)) # use matrix multiplication
    #         coeff_temp = np.array((np.mat(temp_im_arr) * np.mat(basis.basis_pca.reshape(basis.basis_pca.shape[0],-1)).T)) # use matrix multiplication
    #         
    # 
    #         self.psf_coeff = coeff_temp # attach the full list of coeffients to input object
    #         del coeff_temp
    #         self.psf_coeff_new = True
    # 
    #     #psf_im = (np.mat(self.psf_coeff[:,0:num]) * np.mat(basis.basis_pca.reshape(self.im_arr.shape[0],-1)[0:num,]))
    #     psf_im = (np.mat(self.psf_coeff[:,0:num]) * np.mat(basis.basis_pca.reshape(basis.basis_pca.shape[0],-1)[0:num,]))
    #     for i in range(0,self.im_arr.shape[0]): # Add the mean back to the image
    #         psf_im[i,] = psf_im[i,] + basis.im_ave.reshape(-1)
    # 
    #     self.psf_im = np.array(psf_im).reshape(self.im_arr.shape[0],self.im_arr.shape[1],self.im_arr.shape[2])


    # def mk_psfmodel(self,basis,num,mask=None):
    #     """Function for making models of the PSF given an input basis object"""
    #     psf_coeff = np.zeros([self.im_arr.shape[0],num])
    #     psf_im = np.zeros([self.im_arr.shape[0],self.im_arr.shape[1],self.im_arr.shape[2]])
    #     psf_mask = np.ones([self.im_arr.shape[0],self.im_arr.shape[1],self.im_arr.shape[2]])
    #     mask_xcent = np.zeros(self.im_arr.shape[0])
    #     mask_ycent = np.zeros(self.im_arr.shape[0])
    #     norm = np.ones([self.im_arr.shape[0],num])
    # 
    #     if mask is not None:
    #         delta_para = self.para[0] - self.para
    #         for i in range(0,self.im_arr.shape[0]):
    #             #psf_mask[i,],mask_xcent[i],mask_ycent[i] = mask.mask(delta_para[i],xcent=xcent,ycent=ycent)
    #             psf_mask[i,]= mask.mask(delta_para[i])
    #             mask_xcent[i],mask_ycent[i] = mask.temp_xcent,mask.temp_ycent
    #             # NEED TO THINK ABOUT WHETHER I WANT TO USE THIS NORMALISATION:
    #             for j in range(0,num):
    #                 norm[i,j] = (psf_mask[i,]*basis.basis_pca[j,]**2).sum()
    # 
    #         self.psf_mask = psf_mask
    #         self.mask_xcent = mask_xcent
    #         self.mask_ycent = mask_ycent
    #         
    #     for i in range(0,self.im_arr.shape[0]):
    #         im_temp = (self.im_arr[i,] - basis.im_ave) * psf_mask[i,]
    #         psf_coeff[i,] = basis.mk_pcafit_one(im_temp,meansub=False,num=num)/norm[i,]
    #         psf_im[i,] = basis.mk_pcarecon(psf_coeff[i,],meanadd=False)+basis.im_ave
    # 
    #     if hasattr(self,'psf_coeff'):
    #         print 'Replacing current PSF coefficients'
    #     if hasattr(self,'psf_im'):
    #         print 'Replacing current PSF images'
    #         
    #     self.psf_coeff = psf_coeff
    #     self.psf_im = psf_im
        
    # def mk_psf_realisation(self,ind,full=False):    
    #     """Function for making a realisation of the PSF using the data stored in the object"""
    #     im_temp = self.psf_im[ind,] 
    #     if self.cent_remove is True:
    #         if full is True:
    #             im_temp = im_temp + self.im_arr_mask[ind,]
    #         elif full is False:
    #             im_temp = im_temp *self.cent_mask
    #     return im_temp
    # 
    #     # def mk_tweak(self,basis,mask,num_tweak,coeff_type='maskpower_coeff'):
    #     """This function designed to tweak a set of coefficients, using a minimiser, to improve the modeling of the PSF"""
    #     #choose the coefficients to optimise. options (i) sort by power in the mask, (ii) sort by PCA coefficient, (iii) all - might not be possible.
    #     #if coeff_type is 'maskpower':
    #     #    #create an array for the amount of power that each PCA has in the mask:
    #     #    delta_para = self.para[0] - self.para
    #     #    pmask = np.zeros(self.psf_coeff.shape[1])            
    #     #    for i in range(0,self.psf_coeff.shape[1]):
    #     #        pmask[i] = ((basis.basis_pca[i,]**2)*(1.0-self.psf_mask[i,])).sum()             
    #     #   
    #     #    #sort and keep top 4 PCA that have the most power in the mask:
    #     #    ind = pmask.argsort()
    #     #    ind = ind[pmask.size-num_tweak:]
    #     #elif coeff_type is 'coeff'
    #     #else:
    #     #    print 'Other options for coeff_type are not yet supported'
    # 
    #     c_out = np.zeros(shape = self.psf_coeff.shape)
    #     im_out = np.zeros(shape= self.psf_im.shape)
    #     for i in range(0,self.num_files):
    #         im_in = self.im_arr[i,]
    #         c_temp = self.psf_coeff[i,]
    # 
    #         if coeff_type is 'maskpower':
    #             #create an array for the amount of power that each PCA has in the mask:
    #             #delta_para = self.para[0] - self.para
    #             pmask = np.zeros(self.psf_coeff.shape[1])            
    #             for j in range(0,self.psf_coeff.shape[1]):
    #                 pmask[j] = ((basis.basis_pca[j,]**2)*(1.0-self.psf_mask[i,])).sum()                         
    #             #sort and keep top 4 PCA that have the most power in the mask:
    #             ind = pmask.argsort()
    #             ind = ind[pmask.size-num_tweak:]
    #         elif coeff_type is 'coeff':
    #             ind = c_temp.argsort()
    #             ind = ind[c_temp.size-num_tweak:]
    #         elif coeff_type is 'maskpower_coeff':
    #             pmask = np.zeros(self.psf_coeff.shape[1])            
    #             for j in range(0,self.psf_coeff.shape[1]):
    #                 pmask[j] = ((basis.basis_pca[j,]**2)*(1.0-self.psf_mask[i,])).sum()             
    #             temp_arr = abs(c_temp) *pmask
    #             ind = temp_arr.argsort()
    #             ind = ind[temp_arr.size-num_tweak:]
    #         else:
    #             print 'ERROR: Other options for coeff_type are not yet supported'
    #             print coeff_type
    #             return None
    # 
    #         ###NEED TO BE CAREFUL!! NEED TO ROTATE THE MASK!!!****
    #     #    W = mk_gauss2D(self.im_size[0],self.im_size[1],self.im_size[0]*mask.fsize,self.im_size[0]*mask.fxcent,self.im_size[1]*mask.fycent)           
    #         W = mk_gauss2D(self.im_size[0],self.im_size[1],self.im_size[0]*mask.fsize,self.mask_xcent[i],self.mask_ycent[i])           
    #         cmin = fmin(weighted_residuals,c_temp[ind],args=(im_in,basis,c_temp,ind,self.psf_mask[i,],W),ftol=1.0e-6,maxiter=50)
    #     #    print cmin - c_temp[ind]
    #     #    print (cmin - c_temp[ind])/c_temp[ind]
    #         c_comp2 = c_temp.copy()
    #         c_comp2[ind] = cmin
    #         im0_comp2 = basis.mk_pcarecon(c_comp2,meanadd=True)
    #         c_out[i,] = c_comp2
    #         im_out[i,] =  im0_comp2
    #     #rewritting the coefficients and PSF model
    #     print 'Re-write the coefficients and PSF model'
    #     self.psf_coeff = c_out
    #     self.psf_im = im_out
            

    # def plt_psf(self,ind,full=False):
    #     """function for plotting the PSF model"""
    #     pl.clf()            
    #     pl.imshow(self.mk_psf_realisation(ind,full=full),origin='lower',interpolation='nearest')
    #     pl.title('PSF',size='large')
    #     pl.colorbar()   

    # def plt_resid(self,ind):
    #     """function for plotting the residuals between the image and PSF model"""
    #     pl.clf()
    #     pl.imshow(self.im_arr[ind,] - self.mk_psf_realisation(ind,full=False),origin='lower',interpolation='nearest')
    #     pl.title('Residual',size='large')
    #     pl.colorbar()   

    # def plt_stackave(self):
    #     """function for plotting the residuals between the image and PSF model"""
    #     pl.clf()
    #     res_arr = self.im_arr - self.psf_im
    #     pl.imshow((res_arr.sum(axis = 0)/self.num_files)*self.cent_mask,origin='lower',interpolation='nearest')
    #     pl.title('Average of Stack',size='large')
    #     pl.colorbar()   


        
    # 
    #     if file is None:
    #         print 'Error: You have not given a file name where the data should be saved.'
    #         return
    # 
    #     num_half = self.num_files/2
    # 
    #     fim = h5py.File(file,'w')
    #     fim.create_dataset('files',data=self.files)
    #     fim.create_dataset('im_arr',data=self.im_arr,maxshape=(None,None,None))                
    #     fim.create_dataset('para',data=self.para)
    #     fim.create_dataset('im_norm',data=self.im_norm)
    #     fim.create_dataset('cent_mask',data=self.cent_mask)
    #     #fim.create_dataset('im_arr_mask',data=self.im_arr_mask) # may not want to save this!!
    #     #fim.create_dataset('dir_in',data=self.dir_in)
    #     fim.create_dataset('num_files',data=self.num_files)
    #     fim.create_dataset('im_size',data=self.im_size)
    #     fim.create_dataset('cent_remove',data=self.cent_remove)
    #     fim.close()
    # 
    # def restore(self,file):
    #     """ Restore function for reading images data. The format is HDF5 format."""
    #     if file is None:
    #         print 'Error: You have not given a file name where the data should be saved.'
    #         return
    #     fim = h5py.File(file,'r')
    # 
    #     self.files = fim['files'].value
    #     self.im_arr = fim['im_arr'].value
    #     self.para = fim['para'].value
    #     self.im_norm = fim['im_norm'].value
    #     self.num_files = fim['num_files'].value
    #     self.cent_mask = fim['cent_mask'].value
    #     #self.im_arr_mask = fim['im_arr_mask'].value
    #     #self.dir_in = fim['dir_in'].value
    #     self.im_size = fim['im_size'].value
    #     self.cent_remove = fim['cent_remove'].value
    # 
    #     fim.close()
    # 
        #print 'GOT THROUGHT THIS'


