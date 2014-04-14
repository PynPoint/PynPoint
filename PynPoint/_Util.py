#
# Written by Adam Amara 2012
# <adam.amara@phys.ethz.ch>
#

import numpy as np
from scipy import ndimage
import pyfits
import os
import h5py
import glob
from scipy import linalg
import pylab as pl
from PynPoint import _Mask


class dummyclass():
    def __init__(self):
        return None

def print_attributes(Obj):
    for key in Obj.__dict__:
            print key,': ',np.shape(Obj.__dict__[key])#.shape

def getattr(Obj):
    """
    gets attributes of the input Object and output a list containing their names

    :param Obj: an instance
    
    :return: list of attributes, i.e. the data attached to the instance
    """
    data_types = []
    for key in Obj.__dict__:
        data_types.append(key)
    return data_types

#ndimage.rotate(lena, 45)

#have moved the mk_cent function to the mask class!!
#def mk_cent_remove(obj,cent_size=0.2,edge_size=1.0):
#    """This function has been written to mask out the central region (and the corners)"""
#    # WOULD BE NICE TO INCLUDE AN OPTION FOR EITHER TOP-HAT CIRCLE OR GAUSSIAN
#    mask_c = mask(obj.im_size[0],obj.im_size[1],fsize=cent_size,fxcent=0.5,fycent=0.5)
#    # NEED TO DECIDE IF I WANT TO KEEP THE CORNERS:
#    mask_outside = mask(obj.im_size[0],obj.im_size[1],fsize=edge_size,fxcent=0.5,fycent=0.5)
#    cent_mask = mask_c.mask() * (1.0 - mask_outside.mask())
#    im_arr_mask = np.zeros(shape = shape(obj.im_arr))
#    for i in range(0,obj.num_files):
#        im_arr_mask[i,] = obj.im_arr[i,] *(1.0 - cent_mask)
#        obj.im_arr[i,] = obj.im_arr[i,] *cent_mask
#    obj.cent_mask = cent_mask
#    obj.im_arr_mask = im_arr_mask

#Might want to think about restructing the rd_fits function. If can be called from the basis and images classes  
#can get rid of avesub
def rd_fits(obj):#,avesub=True,para_sort=True,inner_pix=False):
    """
    Reading a set of Fits images and storing them in a 3D datacube stack.
    
    :param obj: object that the data will be appended to
    :param avesub: flag if true average is subtracted
    :param para_sort: flag if true then the files are sorted by their paralaxtic angle
    
    :return var:  text
      
    """
    #inner_pix needs to be a 2 elements vector with dimension in x and y
    #    im_arr = np.zeros([obj.num_files,obj.im_size[0],obj.im_size[1]]) # array to store images
    assert (obj.num_files > 0), 'Error: Number of files is zero - input: %f' %obj.num_files
    assert (os.path.isfile(obj.files[0])), 'Error: The first input file is not a file  - input: %s' %obj.files[0]
    

    im_norm = np.zeros(obj.num_files)    
    para = np.zeros(obj.num_files)
    for i in range(0,obj.num_files):
        file_temp = obj.files[i]

        hdu = pyfits.open(file_temp)
        para[i] = hdu[0].header['NEW_PARA']
        im_temp = hdu[0].data
        hdu.close()

        #im_temp = pyfits.getdata(obj.files[i],0)

        #print im_temp.shape
        # if inner_pix is not False:
        #     midx = im_temp.shape[0]/2.# mid point in the x direction 
        #     midy = im_temp.shape[1]/2.# mid point in the y direction
        #     x_ext_2 = inner_pix[0]/2.# half extent of the new image in x
        #     y_ext_2 = inner_pix[1]/2.# half extent of the new image in y
        #     indx1 = np.floor(midx - x_ext_2) # lower index in x
        #     indx2 = np.ceil(midx + x_ext_2)# upper index in x
        #     indy1 = np.floor(midy - y_ext_2) # lower index in y
        #     indy2 = np.ceil(midy + y_ext_2)# upper index in y
        #     im_temp = im_temp[indx1:indx2,indy1:indy2]

        if (i == 0):
            #print indx1,indx2,indy1,indy2
            # print im_temp.shape
            im_arr = np.zeros([obj.num_files,im_temp.shape[0],im_temp.shape[1]]) # setup array to store images
            obj.im_size = im_temp.shape
        #im_norm[i] = im_temp.sum()
        im_arr[i,] = im_temp#/im_norm[i] # rd all the images
    # if para_sort is True:
    #     inds = np.argsort(para)
    #     im_arr = im_arr[inds,]
    #     im_norm = im_norm[inds,]
    #     para = para[inds,]
    obj.im_arr = im_arr
    # obj.im_norm = im_norm
    obj.para = para


def prep_data(obj,recent=False,resize=True,cent_remove=True,F_int=4,
              F_final=2,ran_sub=False,para_sort=True,inner_pix=False,
              cent_size=0.2,edge_size=1.0,stackave=None):


    """
    Prepares the data according to set of key word inputs. This is 
    usually done when an instance is created from fits files. The 
    results of this step are attached to self.
    
    :param recent:
    :param resize:
    :param cent_remove:
    :param F_int:
    :param F_final:
    :param ran_sub:
    :param para_sort:
    :param inner_pix:
    :param cent_size:
    :param edge_size:
    
    """        

    if para_sort is True:
        inds = np.argsort(obj.para)
#         print(inds)
#         print(type(inds))
        obj.im_arr = obj.im_arr[inds,]
        obj.para = obj.para[inds]
        # print(obj.files)
#         print(obj.files)
#         print(len(obj.files))
        
        obj.files = [ obj.files[i] for i in inds] #obj.files[[0,2,1,3]]
        
    # print_attributes(obj)

    #Normalise the images so that they have 'unit area'
    obj.cent_remove = cent_remove
    # obj.im_norm = np.zeros(obj.im_arr.shape[0])
    obj.im_norm = (obj.im_arr.sum(axis = 1)).sum(axis = 1)
    # for i in range(0,len(obj.im_arr[:,0,0])):
    #     obj.im_norm[i,] = obj.im_arr[i,].sum()

    #self.im_arr = self.im_arr.T.dot(np.diag(1./self.im_norm)).T
    for i in range(0,len(obj.im_arr[:,0,0])):
        obj.im_arr[i,] /= obj.im_norm[i]
        
    
    # im_arr[i,] = im_temp/im_norm[i] # rd all the images
    
    
    if str(resize) == 'True' and str(recent) == 'True':
        print 'Update: Resizing and recentring ...'
        obj.im_arr = mk_resizerecent(obj.im_arr,F_int,F_final)
        obj.im_size = obj.im_arr[0,].shape # need to rework into a more elegent solution
    elif str(resize) == 'True':
        print 'Update: Resizing ...'
        obj.im_arr = mk_resizeonly(obj.im_arr,F_final)
        obj.im_size = obj.im_arr[0,].shape # need to rework into a more elegent solution
    
    # print('!!!')
    # print(type(cent_remove))
    # print('L')
    # print(str(cent_remove) == 'True')

    if str(cent_remove) == 'True':
        # print('!!!!! REACHED CENT_REMOVE !!!!!!')
        im_arr_omask,im_arr_imask,cent_mask = _Mask.mk_cent_remove(obj.im_arr,cent_size=cent_size,edge_size=edge_size)
        obj.im_arr = im_arr_omask
        obj.im_arr_mask = im_arr_imask
        obj.cent_mask = cent_mask        

    else:
        # obj.cent_mask = False
        obj.cent_mask = np.ones(shape = obj.im_arr[0,].shape)
        

#def mk_pca(self) :
def mk_basis_pca(im_arr_in):#,ave_sub=True):
    """
    Function for creating the set of PCA's for a stack
    of images
    """
    num_entries = im_arr_in.shape[0]
    im_size = [im_arr_in.shape[1],im_arr_in.shape[2]]
    
    # if ave_sub is False:
    #     print('Warning: Calculating PCA components without subtracting the mean.')
    #     im_ave = np.zeros(shape=np.im_arr_in[0].shape)
    #     im_arr = im_arr_in
    # else:
    #     im_arr,im_ave = mk_avesub(im_arr_in)        
    im_arr,im_ave = mk_avesub(im_arr_in)        
    
    U,s,V = linalg.svd(im_arr.reshape(num_entries,im_size[0]*im_size[1]),full_matrices=False)        
    #U,s,V = linalg.svd(im_arr.reshape(self.num_files,self.im_size[0]*self.im_size[1]),full_matrices=False)        

    basis_pca_arr = V.reshape(V.shape[0],im_size[0],im_size[1])
    
    basis_pca = {'im_ave':im_ave,'im_arr':im_arr,'basis':basis_pca_arr,'basis_type':'pca'}
    return basis_pca

def mk_avesub(im_arr_in):
    """Function for subtracting the mean image from a Stack"""
    im_arr = im_arr_in.copy()
    im_ave = im_arr.mean(axis = 0)#/self.num_files
    #print(len(im_arr[:,0,0]))
    for i in range(0,len(im_arr[:,0,0])):
        # print(i,im_arr[i,].shape,im_ave.shape)
        im_arr[i,] -= im_ave
    return im_arr,im_ave



def gaussian(amp, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: amp*np.exp(-(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)
    
#     
# def extract_inner_pix():
#     ##NEED TO WRITE A FUNCTION LIKE THIS
#     if inner_pix is not False:
#         midx = im_temp.shape[0]/2.# mid point in the x direction 
#         midy = im_temp.shape[1]/2.# mid point in the y direction
#         x_ext_2 = inner_pix[0]/2.# half extent of the new image in x
#         y_ext_2 = inner_pix[1]/2.# half extent of the new image in y
#         indx1 = np.floor(midx - x_ext_2) # lower index in x
#         indx2 = np.ceil(midx + x_ext_2)# upper index in x
#         indy1 = np.floor(midy - y_ext_2) # lower index in y
#         indy2 = np.ceil(midy + y_ext_2)# upper index in y
#         im_temp = im_temp[indx1:indx2,indy1:indy2]
#     

# def sort_by_para():
#     ####NEED TO WRITE A FUNCTION FOR THIS
#     if inner_pix is not False:
#         midx = im_temp.shape[0]/2.# mid point in the x direction 
#         midy = im_temp.shape[1]/2.# mid point in the y direction
#         x_ext_2 = inner_pix[0]/2.# half extent of the new image in x
#         y_ext_2 = inner_pix[1]/2.# half extent of the new image in y
#         indx1 = np.floor(midx - x_ext_2) # lower index in x
#         indx2 = np.ceil(midx + x_ext_2)# upper index in x
#         indy1 = np.floor(midy - y_ext_2) # lower index in y
#         indy2 = np.ceil(midy + y_ext_2)# upper index in y
#         im_temp = im_temp[indx1:indx2,indy1:indy2]
#     

def mk_circle(center_x,center_y):
    """sets up a function for calculating the radius to x,y (after having been initialised with x_cent and y_cent) """
    return lambda x,y:np.sqrt((center_x-x)**2 +(center_y-y)**2)

def moments(data):
	"""Returns (height, x, y, width_x, width_y) the gaussian parameters of a 2D 	
	distribution by calculating its   	
	moments   	
	"""
	total = data.sum()
	Y,X = np.indices(data.shape) #seems strange and backwards, check!
	x = (X*data).sum()/total
	y = (Y*data).sum()/total
	return x, y


def mk_gauss2D(xnum,ynum,gauss_width,xcent=None,ycent=None):
    """Function for making a 2D Gaussian that is centred in the image"""
    X,Y = np.indices([xnum,ynum]) #seems strange and backwards, check!
    if xcent is None:
        xcent = (xnum -1.0)/2.
    if ycent is None:
        ycent = (ynum -1.0)/2.
    G = gaussian(1.0,xcent,ycent,gauss_width,gauss_width)(X,Y)
    return G

def gausscent(data,gauss_width=20.,itnum=3):
    """Measures the centroid after Gaussian Weight has been used"""
    xcent,ycent = moments(data)
    Y,X = np.indices(data.shape) #seems strange and backwards, check!
    for i in range(0,itnum):
        xcent,ycent
        G = gaussian(1.0,xcent,ycent,gauss_width,gauss_width)(X,Y)
        xcent,ycent = moments(data*G)        
        
    return xcent,ycent

def mk_resize(im,xnum,ynum):
    """Routines for resizing images"""
    zoomx = np.float(xnum)/np.float(im.shape[0])
    zoomy = np.float(ynum)/np.float(im.shape[1])
    im_res = ndimage.interpolation.zoom(im,[zoomx,zoomy],order=3)
    #im_res = ndimage.interpolation.zoom(im,3.0,order=1)

    #im_Image =  Image.fromarray(im)
    #Im_Image_rs = im_Image.resize([xnum,ynum],Image.BILINEAR)
    #im_res = np.array(Im_Image_rs)
    return im_res
   
def mk_recent(im,xoff,yoff):
    """Routines for recentring an image"""
    im_res = ndimage.interpolation.shift(im,[yoff,xoff])
    #im_Image =  Image.fromarray(im)
    #im_Image_off = ImageChops.offset(im_Image,xoff,yoff)
    #im_res = np.array(im_Image_off)
    return im_res

def mk_rotate(im,angle):
    """Routines for rotating an image"""
    im_res = ndimage.rotate(im, angle,reshape=False)
    #im_Image =  Image.fromarray(im)
    #im_Image_rot = im_Image.rotate(angle)
    #im_res = np.array(im_Image_rot)
    return im_res
   
def mk_resizerecent(im_arr,F_int,F_final):
    """Function that takes in an array of images and will resize them,
    by increasing the size of the images by a factor of F_int and will then
    reposition each of the images to the gaussian weighted centroid and will
    then resize each of the images again to produce a final images that has size
    dimensions that are F_final that of the input images
	"""
    xnum_int,ynum_int = int(im_arr.shape[1]*F_int),int(im_arr.shape[2]*F_int)
    xnum_final,ynum_final = int(im_arr.shape[1]*F_final),int(im_arr.shape[2]*F_final)
    # print xnum_int,ynum_int
    x_im_cent_int,y_im_cent_int = (xnum_int - 1.) /2.,(ynum_int - 1.) /2.
    #    im_arr_res = np.zeros([im_arr.shape[0],xnum_int,ynum_int])
    im_arr_res = np.zeros([im_arr.shape[0],xnum_final,ynum_final])
    for i in range(0,im_arr.shape[0]):
        im_temp = im_arr[i]
        im_temp = mk_resize(im_temp,xnum_int,ynum_int)
        xcent,ycent = gausscent(im_temp,gauss_width = 20.*F_int)
        xoff,yoff =  x_im_cent_int -xcent, y_im_cent_int - ycent 
        im_temp = mk_recent(im_temp,int(np.round(xoff)),int(np.round(yoff)))
        im_temp = mk_resize(im_temp,xnum_final,ynum_final)
        im_arr_res[i,] = im_temp
    return im_arr_res

def mk_resizeonly(im_arr,F_final):
    """Function that takes in an array of images and will resize them,
    by increasing the size of the images by a factor of F_int and will then
    reposition each of the images to the gaussian weighted centroid and will
    then resize each of the images again to produce a final images that has size
    dimensions that are F_final that of the input images
	"""
    xnum_final,ynum_final = int(im_arr.shape[1]*F_final),int(im_arr.shape[2]*F_final)
#     print(F_final)
#     print(type(F_final))
#     print(im_arr.shape[1],im_arr.shape[2])
#     print(im_arr.shape[0],xnum_final,ynum_final)
    im_arr_res = np.zeros([im_arr.shape[0],xnum_final,ynum_final])
    for i in range(0,im_arr.shape[0]):
        im_temp = im_arr[i]
        im_temp = mk_resize(im_temp,xnum_final,ynum_final)
        im_arr_res[i,] = im_temp
    return im_arr_res

    
def save_data(Obj,filename):
    """ 
    Saves data from an instance to a hdf5 file. This can then be loaded using
    the restore function.
    
    :param Obj: an instance
    :param filename: name of file (hdf5) where the data will be stored 
    
    """
    filename = str(filename)
    # print(filename)
    # print(type(filename))
    assert (type(filename) == str) , 'You need to provide the name of the file where data should be saved.'
    if os.path.isfile(filename):
        print('Warning: the file %s have been overwritten' %filename)

    data_types = getattr(Obj)
    
    fsave = h5py.File(filename,'w')
    # print(filename)
    for i in range(0,len(data_types)):
        # print(data_types[i],':',Obj.__dict__[data_types[i]])
        fsave.create_dataset(data_types[i],data=Obj.__dict__[data_types[i]],maxshape=None)        
    fsave.close()
    
    
def restore_data(Obj,filename,checktype=None):
    """
    Restores data from a hdf5 file that was created using the save function
    of one of the PynPoint instances. 
    
    :param Obj: an instance where the data will be attached
    :param filename: name of the file (hdf5) where the data has been saved
    
    """
    # assert type(filename) == str , 'You need to provide the name of the file where data was saved.'
    # print(filename)
    assert os.path.isfile(filename), 'This file doesnot exist'
    if checktype == None:
        type_comp = Obj.obj_type
    else:
        type_comp = checktype 

    frestore = h5py.File(filename,'r')
    type_file = frestore['obj_type'].value
    frestore.close()
    assert type_comp == type_file,' Error: You have attempted to restore data from the wrong file type \n'
 # The file you provided as type: %s \n
 #        print('')
 #        print('data types & restore options:')
 #        print('   raw_data: read using one of the .create_w* instance builds')
 #        print('   PynPoint_basis: restored from a basis instance (basis.create_restore)')
 #        print('   PynPoint_images: restored from a images instance (images.create_restore)')
 #        frestore.close()
 #        return np.nan
            
    frestore = h5py.File(filename,'r')
    for keys in frestore:
        setattr(Obj,keys,frestore[keys].value)
    frestore.close()
    
def check_type(filename):
    """
    Checks a hdf5 save file to find out which PynPoint class produced it
    
    :param filename: name of the hdf5 file to be checked
    
    :return: string identifying the source of the file: 'PynPoint_basis', 'PynPoint_images', 'PynPoint_residuals' etc
    
    """
    #assert type(filename) == 'str' , 'You need to provide the name of the file where data was saved.'
    assert os.path.isfile(filename), 'This file doesnot exist'
    frestore = h5py.File(filename,'r')    
    obj_type = frestore['obj_type'].value
    frestore.close()
    return obj_type
    
def filename4mdir(dir_in,filetype='convert'):
    """
    derives a hdf5 file name from a directory name 
    """
    if not dir_in.endswith('/'):
        dir_in = dir_in+'/'

    if filetype == 'convert':
        hdffilename = dir_in+os.path.split(os.path.dirname(dir_in))[1]+'_PynPoint_conv.hdf5'
    elif filetype == 'basis':
        hdffilename = dir_in+os.path.split(os.path.dirname(dir_in))[1]+'_PynPoint_basis.hdf5'
    elif filetype == 'images':
        hdffilename = dir_in+os.path.split(os.path.dirname(dir_in))[1]+'_PynPoint_images.hdf5'        
    else:
        print('Warning: file type not recognised - general temp name created')
        hdffilename = dir_in+os.path.split(os.path.dirname(dir_in))[1]+'_PynPoint_temp.hdf5'

    # if not stackave is None:
    #     hdffilename = hdffilename[:-5]+'_stck_'+str(stackave)+'_'+hdffilename[-5:]
        
    return hdffilename
    
def filenme4stack(hdffilename,stackave):
    hdffilename_stack = hdffilename[:-5]+'_stck_'+str(stackave)+'_'+hdffilename[-5:]
    return hdffilename_stack
    
    
    
def conv_dirfits2hdf5(dir_in,outputfile = None):#,stackave=None):
    """
    Converts fits inputs (stored in directory) into hdf5 file format. These files are faster to load and 
    can be handeled more easily by PynPoint.
    
    :param dir_in: name of directory containing the fits images
    :param outputfile: name of the hdf5 file for storage. If none then a name is derived from the dir_in.
    
    """
    obj = dummyclass()
    obj.files = file_list(dir_in)
    obj.num_files = len(obj.files)#num_files
    rd_fits(obj)#,avesub=False,para_sort=False,inner_pix=False)
    obj.obj_type = 'raw_data'
    if outputfile == None:
        filehdf5 = filename4mdir(dir_in)
    else:
        filehdf5 = outputfile
        
    # if not stackave is None:
    #     obj = stackave_func(obj,stackave)
            
    save_data(obj,filehdf5)

def mkstacked(file_in,file_stck,stackave):
    """
    Averages over adjacent images. This has the effect of reducing the size of the stack. 
    """
    # assert stackave ###
    obj = dummyclass()

    restore_data(obj,file_in,checktype='raw_data')
    
    num_new = int(np.floor(float(obj.num_files)/float(stackave)))
    para_new = np.zeros(num_new)
    im_arr_new = np.zeros([num_new,obj.im_arr.shape[1],obj.im_arr.shape[2]])
    for i in range(0,num_new):
        para_new[i] = obj.para[i*stackave:(i+1)*stackave].mean()
        im_arr_new[i,] = obj.im_arr[i*stackave:(i+1)*stackave,].mean(axis=0)
    obj.im_arr = im_arr_new
    obj.para = para_new
    obj.num_files = num_new
    
    save_data(obj,file_stck)
    ### HACK:
    # filebookkeep = open('temp_HACK.temp','w')
    # filebookkeep.write('#-Time-#: %10.20f' %time.time()) 
    # print('#-Time-#: %10.20f' %time.time()) 
    # filebookkeep.close()
    
    

def file_list(dir_in,ran_sub=False):
    """
    lists all the fits files in a given directory
    """
    
    files = glob.glob(dir_in+'*.fits')
    if ran_sub is not False:
        np.random.shuffle(files)#,random.random)
        files = files[0:ran_sub]     

    return files
    
    
def peak_find(imtemp,limit=0.8,printit=False):
    """
    Detects peaks in an image.
    """
    c = pl.contour(imtemp,[imtemp.max()*limit])
    p = c.collections[0].get_paths()
    num_peaks = np.size(p)
    x_peaks = np.zeros(num_peaks)
    y_peaks = np.zeros(num_peaks)
    h_peaks = np.zeros(num_peaks)
    for i in range(0,num_peaks):
        x = p[i].vertices[:,0]
        y = p[i].vertices[:,1]
        x_peaks[i] = x.mean()
        y_peaks[i] = y.mean()
        h_peaks[i] = imtemp[round(y_peaks[i]),round(x_peaks[i])]
        sig = np.sqrt((imtemp**2).sum()/np.size(imtemp.nonzero())/2.) 
    
    if printit is True:
        print 'Number of images found:',num_peaks
        print 
        print 'im num','|','x','|','y'
        print '-----------------------'
        for i in range(0,num_peaks):
            print i,'|',x_peaks[i],'|',y_peaks[i],'|',h_peaks[i],'|',h_peaks[i]/sig
            
    return x_peaks,y_peaks, h_peaks,sig,num_peaks

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")
