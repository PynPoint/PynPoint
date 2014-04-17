        
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

