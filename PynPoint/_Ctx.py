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

from __future__ import print_function

import os
import h5py
from PynPoint.Basis import basis
from PynPoint.Images import images
from PynPoint.Residuals import residuals


class Ctx():
    """
    Returns the current global namespace context.
    
    :return: reference to the context module
    
    """
    def __init__(self):
        self._data = {}        

    def add(self,name,obj):
        self._data[name] = obj
        
    def get(self,name):
        return self._data[name]
        
    def entries(self):
        return self._data.keys()
        
    def save(self,dirout):
        ctx_dir = dirout+'/ctx_info/'
        os.mkdir(ctx_dir)
        entries = self.entries()
        print(entries)
        fsave = h5py.File(ctx_dir+'ctx_basic.hdf5','w')
        fsave.create_dataset('entries',data=entries)
        fsave.close()
        
        for i in range(0,len(entries)):
            file_temp = ctx_dir+entries[i]+'.hdf5'
            temp = self.get(entries[i])
            temp.save(file_temp) 
        return ctx_dir     
        
    @staticmethod
    def restore(dirin):
        obj = Ctx()
        ctx_dir = dirin

        fhdf = h5py.File(ctx_dir+'ctx_basic.hdf5','r')
        entries = fhdf.get('entries')[:]
        fhdf.close()
        

        for i in range(0,len(entries)):
            filetemp = ctx_dir+entries[i]+'.hdf5'
            data_type = str.split(entries[i],'_')[0]
            if data_type == 'basis':
                run_temp = basis.create_restore(filetemp)
            elif data_type == 'images':
                run_temp = images.create_restore(filetemp)
            elif data_type == 'residuals':
                run_temp = residuals.create_restore(filetemp)
            else:
                print('ERROR: filetype not recognised!')
            obj.add(entries[i],run_temp)

        return obj
        
                
        
        
        
        
        
        
        
        

