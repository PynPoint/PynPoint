import numpy as np
import pylab as pl
import Util

class mask:
    """Object for creating and manipulating the masks"""
    def __init__(self,xnum,ynum,pattern='circle',para_ini=0.0,fsize=0.15,fxcent=0.2,fycent=0.5):
        self.xnum = xnum # array size in x-direction
        self.ynum = ynum # array size in y-direction
        self.para_ini = para_ini # para angle of first image
        self.fsize = fsize # size of the mask as a fraction of the image size
        self.size_pix = fsize*xnum # size of the mask in pixels
        self.fxcent = fxcent
        self.xcent = fxcent*xnum
        self.fycent = fycent
        self.ycent = fycent*xnum
        self.pattern = pattern
        self.mask_base = self.mk_circle(xnum,ynum,self.xcent,self.ycent,self.size_pix/2.) # This is the mask for the first image
    @staticmethod
    def mk_circle(xnum,ynum,xcent,ycent,rad_lim):
    #def mk_circle(self,xnum,ynum,xcent,ycent,rad_lim):
        """function for making a circular aperture"""
        Y,X = np.indices([xnum,ynum]) #seems strange and backwards, check!
        rad = Util.mk_circle(xcent,ycent)(X,Y)
        i,j = np.where(rad <= rad_lim)
        mask_base = np.ones((xnum,ynum),float) #something strange about the order of x and y!
        mask_base[i,j] = 0.0
        return mask_base
        
    # @staticmethod
    # def mk_circle2(xnum,ynum,xcent,ycent,rad_lim):
    #     """function for making a circular aperture"""
    #     Y,X = np.indices([xnum,ynum]) #seems strange and backwards, check!
    #     rad = Util.mk_circle(xcent,ycent)(X,Y)
    #     i,j = np.where(rad <= rad_lim)
    #     mask_base = np.ones((xnum,ynum),float) #something strange about the order of x and y!
    #     mask_base[i,j] = 0.0
    #     return mask_base
    #         
        
    def mask(self,del_para=None):
    	"""This function return an array containing the mask with the correct para rotation"""
    	mask_current = self.mask_base
    	if del_para is not None:
    		theta = del_para
    		xp = np.cos(theta/180.*np.pi)*(self.fxcent-0.5) - np.sin(theta/180.*np.pi)*(self.fycent - 0.5)
    		yp = np.sin(theta/180.*np.pi)*(self.fxcent-0.5) + np.cos(theta/180.*np.pi)*(self.fycent - 0.5)
    		xcent = (0.5+xp)*self.xnum
    		ycent = (0.5+yp)*self.ynum
    		mask_current =  self.mk_circle(self.xnum,self.ynum,xcent,ycent,self.size_pix/2.)
    		self.temp_xcent=xcent
    		self.temp_ycent=ycent
    	else:
    		xcent=ycent=None
    	return mask_current#,xcent,ycent
    	
    def plt_mask(self,para=None):
		""""This function plots the mask with the correct para rotation"""
		pl.clf()
		pl.imshow(self.mask(del_para=para),origin='lower',interpolation='nearest')
		pl.title('Mask',size='large')
		pl.colorbar()   
        
    @staticmethod
    def mk_cent_remove(im_arr,cent_size=0.2,edge_size=1.0):
        """This function has been written to mask out the central region (and the corners)"""
        # WOULD BE NICE TO INCLUDE AN OPTION FOR EITHER TOP-HAT CIRCLE OR GAUSSIAN
        im_size = im_arr[0,].shape
        print(im_size)
        mask_c = mask.mk_circle(im_size[0],im_size[1],im_size[0]/2.,im_size[1]/2.,cent_size*im_size[0])
        mask_outside = mask.mk_circle(im_size[0],im_size[1],im_size[0]/2.,im_size[1]/2.,0.5*im_size[0])
        #mask_c = self.mask(im_size[0],im_size[1],fsize=cent_size,fxcent=0.5,fycent=0.5)
        # NEED TO DECIDE IF I WANT TO KEEP THE CORNERS:
        #mask_outside = self.mask(im_size[0],im_size[1],fsize=edge_size,fxcent=0.5,fycent=0.5)
        cent_mask = mask_c * (1.0 - mask_outside)
        im_arr_omask = np.zeros(shape = im_arr.shape)
        im_arr_imask = np.zeros(shape = im_arr.shape)

        #CAN PROBABLY SPEED UP THIS LOOP:!!!
        for i in range(0,im_arr.shape[0]):
            im_arr_imask[i,] = im_arr[i,] *(1.0 - cent_mask)
            im_arr_omask[i,] = im_arr[i,] *cent_mask

        return im_arr_omask,im_arr_imask,cent_mask
 
#    def mk_cent_remove(self,cent_size=0.2,edge_size=1.0):
#        """This function has been written to mask out the central region (and the corners)"""
#        # WOULD BE NICE TO INCLUDE AN OPTION FOR EITHER TOP-HAT CIRCLE OR GAUSSIAN
#        mask_c = mask(self.im_size[0],self.im_size[1],fsize=cent_size,fxcent=0.5,fycent=0.5)
#        # NEED TO DECIDE IF I WANT TO KEEP THE CORNERS:
#        mask_outside = mask(self.im_size[0],self.im_size[1],fsize=edge_size,fxcent=0.5,fycent=0.5)
#        cent_mask = mask_c.mask() * (1.0 - mask_outside.mask())
#        im_arr_mask = np.zeros(shape = shape(self.im_arr))
#        for i in range(0,self.num_files):
#            im_arr_mask[i,] = self.im_arr[i,] *(1.0 - cent_mask)
#            self.im_arr[i,] = self.im_arr[i,] *cent_mask
#        self.cent_mask = cent_mask
#        self.im_arr_mask = im_arr_mask
 
