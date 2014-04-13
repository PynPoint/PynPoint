        
#import external functions:
import pylab as pl
import time

#import extra PynPoint functions:
from _BasePynPoint import base_pynpoint
import _Creators


#Images class:
class images(base_pynpoint):
    """Object for dealing with the images that need to be analysed"""
    
    def __init__(self):
        """
        Initialise an instance of the images class. The result is simple and
        almost empty (in terms of attributes)        
        """
        
        self.obj_type = 'PynPoint_images'
        
        
    @staticmethod
    def create_wdir(dir_in,**kwargs): #dir_in,ran_sub=False,force_reload=False,prep_data=True,**kwargs):
        """
        Creates an instance from directory. 
        See pynpoint_create_wdir (in PynPoint_creator_funcs.py) for more details.        
        """

        obj = images()
        _Creators.pynpoint_create_wdir(obj,dir_in,**kwargs)
        return obj

    @staticmethod
    def create_whdf5input(file_in,**kwargs):#file_in,ran_sub=False,prep_data=True,**kwargs)
        """
        Creates an instance from hdf5 file. 
        See pynpoint_create_whdf5input (in PynPoint_creator_funcs.py) for more details.        
        """

        obj = images()
        _Creators.pynpoint_create_whdf5input(obj,file_in,**kwargs)
        return obj

        
    @staticmethod
    def create_restore(filename):
        """
        Creates an instance from saved file.
        See pynpoint_create_restore (in PynPoint_creator_funcs.py) for more details.
        """
        
        obj = images()
        _Creators.pynpoint_create_restore(obj,filename)
        #Util.restore_data(obj,filename)
        return obj

    @staticmethod
    def create_wfitsfiles(*args,**kwargs):
        """
        
        """
        obj = images()
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


    def plt_psf(self,ind,full=False):
        """function for plotting the PSF model"""
        pl.clf()            
        pl.imshow(self.mk_psf_realisation(ind,full=full),origin='lower',interpolation='nearest')
        pl.title('PSF',size='large')
        pl.colorbar()   


    def plt_resid(self,ind):
        """function for plotting the residuals between the image and PSF model"""
        pl.clf()
        pl.imshow(self.im_arr[ind,] - self.mk_psf_realisation(ind,full=False),origin='lower',interpolation='nearest')
        pl.title('Residual',size='large')
        pl.colorbar()   


    def plt_stackave(self):
        """function for plotting the residuals between the image and PSF model"""
        pl.clf()
        res_arr = self.im_arr - self.psf_im_arr
        pl.imshow((res_arr.sum(axis = 0)/self.num_files)*self.cent_mask,origin='lower',interpolation='nearest')
        pl.title('Average of Stack',size='large')
        pl.colorbar()   


    def anim_active(self,time_gap=0.04,num_frames = False):
        """function for animating the input images"""
        pl.clf()
        temp_im = self.im_arr[0,] - self.mk_psf_realisation(0,full=False)
        if num_frames is False:
            num_frames = self.num_files
        im_max = temp_im.max()
        im_min = temp_im.min()
        for i in range(0,num_frames):
            pl.clf()
            #            plt_resid(i)
            temp_im = self.im_arr[i,] - self.mk_psf_realisation(i,full=False)
            pl.imshow(temp_im,animated=True,interpolation='nearest',origin='lower',clim=[im_min,im_max])
            pl.title('Active Images')
            pl.draw()
            time.sleep(time_gap)
    
