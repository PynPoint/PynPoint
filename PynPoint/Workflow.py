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

# External modules
import ConfigParser
import types
import os
from time import gmtime, strftime
import shutil
import h5py
import time
import re
from operator import itemgetter

from PynPoint.Basis import basis
from PynPoint.Images import images
from PynPoint.Residuals import residuals
from PynPoint._Ctx import Ctx
from PynPoint import _Util


class workflow():
    """
    A simple workflow engine for managing PynPoint runs. This engine takes in a configuration 
    file where the user can specify all the operations that should be run along with keyword options
    
    """
    
    def __init__(self):
        """
        Initialise an instance of the images class.  
        """
        self.obj_type = 'PynPoint_workflow'
        # section title to be used in config to identify modules to be run
        self.module_string = 'module'
     
    @staticmethod
    def run(config,force_replace=False):
        """
        run the workflow using config. Need to pass either a
        config instance or name of file containing config information.
        
        :param config: name of the config file with details of the run to be executed
        :param force_replace: If True then the workspace directory will be overwritten if it already exists
        
        """
        obj = workflow()
        obj._init_config(config)
        obj._setup_workspace(force_replace=force_replace)
        obj._runmods()
        obj._save()
        # obj._report()
        # obj._helpfiles()
        obj._tidyup()
        
        return obj
        
    @staticmethod
    def restore(dirin):
        """
        Restores a previously a workspace that has previously been 
        calculated by the workflow.
        
        :param dirin: Work directory created by by an earlier calculation (using run method). 
        
        
        """
        
        obj = workflow()
        obj._ctx = Ctx.restore(dirin+'/ctx_info/')
        return obj
        
        
    def _save(self):#,dirout):
        """
        save
        """
        
        dirout = self.dirname
        self._ctx.save(dirout)
        fsave = h5py.File(dirout+'/ws_basic.hdf5','w')
        fsave.create_dataset('dirname',data=self.dirname)
        fsave.create_dataset('modules',data=self.modules)        
        fsave.close()
        fileconfig = open(dirout+'/ws.config','w')
        self.config.write(fileconfig)
        fileconfig.close()
        return dirout
        
        
    def get(self,name):
        """
        Used to extract instances of images, basis or residuals from the workflow instance
        
        :param name: name of the option to be restored - see get_options for available options
        """
        return self._ctx.get(name)   
        
    def get_available(self):
        """
        Returns the available module names
        
        :return: List of modules
        """
        
        return self._ctx.entries()
           
    
    def _init_config(self,config_in):
        #TODO: change to isinstance!
        if (type(config_in) == types.InstanceType):
            assert config_in.__module__ == 'ConfigParser', 'Error: This instance is not from ConfigParser'
            self.config = config_in
        else:
            assert(type(config_in) == str)
            self.config = ConfigParser.ConfigParser() #dict_type=collections.OrderedDict)
            self.config.optionxform = str

            # self.config.optionxform(str())
            self.config.read(config_in)
            
        #sort the list of modules according to the digit at the end of the section name
        moduleList = [s for s in self.config.sections() if self.module_string in s]
        self.modules = map(itemgetter(1), sorted([(int(re.search('\d+', e).group(0)),e) for e in moduleList ], key=itemgetter(0)))
        

    def _setup_workspace(self,force_replace=False):
        dirname = self.config.get('workspace','workdir')
        if force_replace==True and os.path.exists(dirname):
            print('Warning: The directory %s already existed. It has been deleted and replaced!' %dirname)
            shutil.rmtree(dirname)
            
            
        # print(dirname)
        #assert 2==1
        os.mkdir(dirname)
        fileconfig = open(dirname+'/wf.config','w')
        self.config.write(fileconfig)
        fileconfig.close()
        self.dirname = dirname
        filebookkeep = open(dirname+'/book_keeping.txt','w')
        filebookkeep.write('Start time:\n')
        filebookkeep.write(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        filebookkeep.write('\n')
        filebookkeep.close()
        
    def _tidyup(self):
        dirname = self.config.get('workspace','workdir')
        # if force_replace=True and os.path.exists(dirname):
        #     shutil.rmtree(dirname)
            
        filebookkeep = open(dirname+'/book_keeping.txt','a')
        filebookkeep.write('\n')
        filebookkeep.write('Calculations completed successfully.\n')
        filebookkeep.write('\n')
        filebookkeep.write('End time:\n')
        filebookkeep.write(strftime("%Y-%m-%d %H:%M:%S", gmtime())+'\n')
        filebookkeep.write('#-Time-#: %10.20f' %time.time())        
        filebookkeep.write('\n')
        filebookkeep.close()


    def _timestamp(self,extra_text=None):
        dirname = self.config.get('workspace','workdir')
        filebookkeep = open(dirname+'/book_keeping.txt','a')
        filebookkeep.write('\n')
        if not extra_text is None:
            filebookkeep.write(extra_text+'\n')
        filebookkeep.write('Time stamp:\n')
        filebookkeep.write(strftime("%Y-%m-%d %H:%M:%S", gmtime())+'\n')
        filebookkeep.write('#-Time-#: %10.20f' %time.time())        
        filebookkeep.write('\n')
        filebookkeep.close()
        

    def _runmods(self):
        self._ctx = Ctx()
        result_names = []
        
        for mod in self.modules:
            mod_type = self.config.get(mod,'mod_type')
            if mod_type == 'images':
                run_temp = self._run_images_mod(mod)
            elif mod_type == 'basis':
                run_temp = self._run_basis_mod(mod)
            elif mod_type == 'residuals':
                run_temp = self._run_residuals_mod(mod,self._ctx)
            else:
                raise TypeError('Error: mod_type option can be: images, basis or residuals')
            
            name = mod_type+'_'+mod
            result_names.append(name)
            self._ctx.add(name,run_temp)  
            self._timestamp('Finished '+name)
            
    
    def _run_images_mod(self,section_id):
        input_data = self.config.get('workspace','datadir')+self.config.get(section_id,'input')
        kwargs = self._get_keyword_options(section_id)
        
        
        
        if self.config.get(section_id,'intype') == 'dir':
            images_run = images.create_wdir(input_data,**kwargs)

        elif self.config.get(section_id,'intype') == 'hdffile':
            images_run = images.create_whdf5input(input_data,**kwargs)

        elif self.config.get(section_id,'intype') == 'restore':
            if not kwargs == None:
                print('Warning: Keyword options are being ignored since input type is restore') 
            images_run = images.create_restore(input_data)

        else:
            assert 1==2,'Error: workflow supported input types are dir, hdffile, restore'
        return images_run                    
            
        
        
    def _run_basis_mod(self,section_id):
        input_data = self.config.get('workspace','datadir')+self.config.get(section_id,'input')
        kwargs = self._get_keyword_options(section_id)
        
        if self.config.get(section_id,'intype') == 'dir':
            basis_run = basis.create_wdir(input_data,**kwargs)

        elif self.config.get(section_id,'intype') == 'hdffile':
            basis_run = basis.create_whdf5input(input_data,**kwargs)

        elif self.config.get(section_id,'intype') == 'restore':
            if not kwargs == None:
                print('Warning: Keyword options are being ignored since input type is restore') 
            basis_run = basis.create_restore(input_data)

        else:
            assert 1==2,'Error: workflow supported input types are dir, hdffile, restore'
        return basis_run                    
        

    def _run_residuals_mod(self,section_id,ctx):
        # input_data = self.config.get('workspace','datadir')+self.config.get(section_id,'input')
        images_in = self.config.get(section_id,'images_input')
        basis_in = self.config.get(section_id,'basis_input')
        
        images = ctx.get('images_'+images_in)
        basis = ctx.get('basis_'+basis_in)
        if self.config.get(section_id,'intype') == 'instances':
            res_run = residuals.create_winstances(images,basis)
            
        return res_run
        
        
    def _get_keyword_options(self,section_id):
        if self.config.get(section_id,'options') == 'None':
            kwargs = None
        else:
            options_section = self.config.get(section_id,'options')
            kwargs = self.config._sections[options_section]
        if '__name__' in kwargs:
            del kwargs['__name__']
        
        if not kwargs is None:
            kwargs = self._check_kwargs(**kwargs)

        return kwargs

        
    def _check_kwargs(self,**kwargs):

        if 'recent' in kwargs.keys():
            kwargs['recent'] = _Util.str2bool(kwargs['recent'])
            
            
        if 'resize' in kwargs.keys():
        # if hasattr(kwargs, 'resize'):
            kwargs['resize'] = _Util.str2bool(kwargs['resize'])
        
        if 'cent_remove' in kwargs.keys():
        # if hasattr(kwargs, 'cent_remove'):
            kwargs['cent_remove'] = _Util.str2bool(kwargs['cent_remove'])
        
        if 'ran_sub' in kwargs.keys():
        # if hasattr(kwargs, 'ran_sub'):
            kwargs['ran_sub'] = _Util.str2bool(kwargs['ran_sub'])
    
        if 'para_sort' in kwargs.keys():
        # if hasattr(kwargs, 'para_sort'):
            kwargs['para_sort'] = _Util.str2bool(kwargs['para_sort'])
    
        if 'inner_pix' in kwargs.keys():
        # if hasattr(kwargs, 'inner_pix'):
            kwargs['inner_pix'] = _Util.str2bool(kwargs['inner_pix'])

        if 'F_int' in kwargs.keys():
        # if hasattr(kwargs, 'F_int'):
            kwargs['F_int'] = float(kwargs['F_int'])
    
        if 'F_final' in kwargs.keys():
        # if hasattr(kwargs, 'F_final'):
            kwargs['F_final'] = float(kwargs['F_final'])
    
        if 'cent_size' in kwargs.keys():
        # if hasattr(kwargs, 'cent_size'):
            kwargs['cent_size'] = float(kwargs['cent_size'])
    
        if 'edge_size' in kwargs.keys():
        # if hasattr(kwargs, 'edge_size'):
            kwargs['edge_size'] = float(kwargs['edge_size'])

        if 'stackave' in kwargs.keys():
        # if hasattr(kwargs, 'edge_size'):
            kwargs['stackave'] = int(kwargs['stackave'])

        return kwargs
