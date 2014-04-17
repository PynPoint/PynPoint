        
#import external functions:
import pyfits
import numpy as np
import os

#import extra PynPoint functions:
from PynPoint import _Util



def pynpoint_create_restore(obj,filename):
    """
    Creates an instance of the images class using a previously
    saved file. The file should have been stored as an HDF5 file
    
    :param filename: name of the inputfile 
    
    """
    #obj = pynpoint_parent()
    _Util.restore_data(obj,filename)
    #return obj

def pynpoint_create_wdir(obj,dir_in,ran_sub=False,force_reload=False,prep_data=True,**kwargs):
    """
    Creates an instance of the images class using dir_in, which is the
    name of a directory containing the input fits files. 
    As well as its input parameters, this function can pass the keyword
    options used by prep_data method. 
    
    :param dir_in: name of the directory with fits files
    :param ran_sub: if a number (N) is passed then a random subset of filessize is selected 
    
    """
    
    ##convert to hdf5
    assert (os.path.isdir(dir_in)), 'Error: Input directory does not exist - input requested: %s'%dir_in
    file_hdf5 = _Util.filename4mdir(dir_in)    
#     print(file_hdf5)
    # assert os.path.isfile(file_hdf5), 'Error: No fits files found in the directory: %s, %s' %dir_in %file_hdf5
    if (force_reload is True) or (not os.path.isfile(file_hdf5)):
        _Util.conv_dirfits2hdf5(dir_in,outputfile = file_hdf5)
    # obj = pynpoint_parent.create_whdf5input(file_hdf5,ran_sub=False,**kwargs)
    pynpoint_create_whdf5input(obj,file_hdf5,ran_sub=False,**kwargs)
    #return obj

def pynpoint_create_whdf5input(obj,file_in,ran_sub=False,prep_data=True,**kwargs):
    """
    Creates an instance of the images class using dir_in, which is the
    name of a directory containing the input fits files. 
    As well as its input parameters, this function can pass the keyword
    options used by prep_data method. 
    
    :param dir_in: name of the directory with fits files
    :param ran_sub: if a number (N) is passed then a random subset of filessize is selected 
    
    """
    #pynpoint_parent.__init__(obj)
    #obj = pynpoint_parent()
    assert (os.path.isfile(file_in)), 'Error: Input file does not exist - input requested: %s'%file_in  
    if 'stackave' in kwargs:
        stackave = kwargs['stackave']
    else:
        stackave = None
    
    obj_type = obj.obj_type
    if not stackave is None:
        filename_stck = _Util.filenme4stack(file_in,stackave)
        if not os.path.isfile(filename_stck):
            _Util.mkstacked(file_in,filename_stck,stackave)
        file_in = filename_stck
    
    # if not os.isfile()
    _Util.restore_data(obj,file_in,checktype='raw_data')
    obj.obj_type = obj_type #'PynPoint_images' # restate since this is replaced in the restore_data
    #use the ran_subset keyword
    if prep_data is True:
        _Util.prep_data(obj,**kwargs)                    

    #return obj

def pynpoint_create_wfitsfiles(obj,files,ran_sub=False,prep_data=True,**kwargs):
    """
    Creates an instance of the images class using a list of fits file names. 
    As well as its input parameters, this function can pass the keyword
    options used by prep_data method. 
    
    :param files: list with with fits file names
    :param ran_sub: if a number (N) is passed then a random subset of filessize is selected 
    
    """
    #obj = pynpoint_parent()
    num_files = np.size(files)            # number of files
    assert (num_files >0),'Error: No files inputs, e.g.: %s' %files[0]
    assert os.path.isfile(files[0]),'Error: No files inputs, e.g.: %s' %files[0]
    obj.files = files
    obj.num_files = num_files
    _Util.rd_fits(obj)#,avesub=False,para_sort=para_sort,inner_pix=inner_pix)
    if prep_data is True:
        _Util.prep_data(obj,**kwargs)
        
