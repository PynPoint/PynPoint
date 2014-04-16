        
#import external functions:
import pylab as pl
import time
import numpy as np

#import extra PynPoint functions:
import _Util


#base class:
class base_pynpoint:
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

        _Util.save_data(self,filename)

    def plt_im(self,ind):
        """Function for plotting the input images"""
        pl.clf()
        pl.imshow(self.im_arr[ind,],origin='lower',interpolation='nearest')
        pl.title('Image',size='large')
        pl.colorbar()


    # def anim_im(self,time_gap=0.04,num_frames = False):
    #     """function for animating the input images"""
    #     pl.clf()
    #     if num_frames is False:
    #         num_frames = self.num_files
    #     for i in range(0,num_frames):
    #         pl.clf()
    #         pl.imshow(self.im_arr[i,],animated=True,interpolation='nearest',origin='lower')
    #         pl.title('Original Images')
    #         pl.draw()
    #         time.sleep(time_gap)
            
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


