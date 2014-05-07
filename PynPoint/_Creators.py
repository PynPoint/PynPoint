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
   
import numpy as np
import os

from PynPoint import _Util



def restore(obj,filename):
    """
    Restores data from a hdf5 file previously saved
    
    :param filename: name of the inputfile 
    
    """
    _Util.restore_data(obj,filename)

def pynpoint_create_wdir(obj,dir_in,force_reload=False,prep_data=True,**kwargs):
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
    random_sample =  None
    if ('ran_sub' in kwargs.keys()):
        if not kwargs['ran_sub'] in (None,False):
            ran_sub = int(kwargs['ran_sub'])
            file_hdf5 = file_hdf5[:-5] +'_random_sample_'+str(ran_sub)+file_hdf5[-5:]
            random_sample = ran_sub


    print('random_sample: %s' %random_sample)
    print(kwargs.keys())
    print(kwargs['ran_sub'])


    if (force_reload is True) or (not os.path.isfile(file_hdf5)):
        _Util.conv_dirfits2hdf5(dir_in,outputfile = file_hdf5,random_sample_size=random_sample)

    pynpoint_create_whdf5input(obj,file_hdf5,**kwargs)


def pynpoint_create_whdf5input(obj,file_in,prep_data=True,**kwargs):
    """
    Creates an instance of the images class using dir_in, which is the
    name of a directory containing the input fits files. 
    As well as its input parameters, this function can pass the keyword
    options used by prep_data method. 
    
    :param dir_in: name of the directory with fits files
    :param ran_sub: if a number (N) is passed then a random subset of filessize is selected 
    
    """


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
    

    _Util.restore_data(obj,file_in,checktype='raw_data')
    obj.obj_type = obj_type #'PynPoint_images' # restate since this is replaced in the restore_data

    if prep_data is True:
        _Util.prep_data(obj,**kwargs)                    


def pynpoint_create_wfitsfiles(obj,files,ran_sub=None,prep_data=True,**kwargs):
    """
    Creates an instance of the images class using a list of fits file names. 
    As well as its input parameters, this function can pass the keyword
    options used by prep_data method. 
    
    :param files: list with with fits file names
    :param ran_sub: if a number (N) is passed then a random subset of filessize is selected 
    
    """
    num_files = np.size(files)            # number of files
    assert (num_files >0),'Error: No files inputs, e.g.: %s' %files[0]
    assert os.path.isfile(files[0]),'Error: No files inputs, e.g.: %s' %files[0]
    obj.files = files
    obj.num_files = num_files
    _Util.rd_fits(obj)#,avesub=False,para_sort=para_sort,inner_pix=inner_pix)
    if prep_data is True:
        _Util.prep_data(obj,**kwargs)
        
