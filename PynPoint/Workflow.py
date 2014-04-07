
import ConfigParser 
import types
import os
from time import gmtime, strftime
from Basis import basis
from Images import images
from Residuals import residuals
from _Ctx import Ctx
import shutil
import h5py
import time


class workflow():
    
    obj_type = 'PynPoint_workflow'
    # section title to be used in config to identify modules to be run
    module_string = 'module'
     
    def __init__(self):
        """
        Initialise an instance of the images class. The result is simple and
        almost empty (in terms of attributes)        
        """
        pass
        
        
    @staticmethod
    def run(config,force_replace=False):
        """
        run the workflow using config. Need to pass either a
        config instance or name of file containing config information.
        """
        obj = workflow()
        obj._init_config(config)
        obj._setup_workspace(force_replace=force_replace)
        obj._runmods()
        obj.save()
        # obj._report()
        # obj._helpfiles()
        obj._tidyup()
        
        return obj
        
    @staticmethod
    def restore(dirin):
        """
        Restores a previously a workspace that has previously been 
        calculated by the workflow.
        """
        
        obj = workflow()
        obj._ctx = Ctx.restore(dirin+'/ctx_info/')
        return obj
        
        
    def save(self):#,dirout):
        dirout = data=self.dirname
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
        return self._ctx.get(name)   
        
    def get_options(self):
        return self._ctx.entries()
           
    
    def _init_config(self,config_in):
        # print(type(config_in))        
        if (type(config_in) == types.InstanceType):
            assert config_in.__module__ == 'ConfigParser', 'Error: This instance is not from ConfigParser'
            self.config = config_in
        else:
            assert(type(config_in) == str)
            self.config = ConfigParser.ConfigParser()
            self.config.optionxform = str

            # self.config.optionxform(str())
            self.config.read(config_in)
        self.modules = [s for s in self.config.sections() if self.module_string in s]
            
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
        
        #need to manage the data in some way!
        for i in range(0,len(self.modules)):
            mod_type = self.config.get(self.modules[i],'mod_type')
            if mod_type == 'images':
                run_temp = self._run_images_mod(self.modules[i])
            elif mod_type == 'basis':
                run_temp = self._run_basis_mod(self.modules[i])
            elif mod_type == 'residuals':
                run_temp = self._run_residuals_mod(self.modules[i],self._ctx)
            else:
                assert 1==2,'Error: mod_type option can be: images, basis or residuals'
            name = mod_type+'_'+self.modules[i]
            result_names.append(name)
            self._ctx.add(name,run_temp)  
            self._timestamp('Finished '+name)
            
#         import numpy
#         import itertools
#         
#         fields = ["cent_mask", "im_arr", "im_arr_mask", "im_norm", "im_ave", "para", "psf_basis", "files"]
#         for field in fields:
#             for (n1,n2) in itertools.combinations(result_names, 2):
#                 o1 = self._ctx.get(n1)
#                 o2 = self._ctx.get(n2)
#                 try:
#                     arr1 = getattr(o1, field)
#                     arr2 = getattr(o2, field)
#                     state = (numpy.all(arr1==arr2))
#                     try:
#                         print("%s, %s %s %s %s: %s" %(field, n1, arr1.shape, n2, arr2.shape, state))
#                     except Exception:
#                         print("%s, %s %s %s %s: %s" %(field, n1, numpy.array(arr1).shape, n2, numpy.array(arr2).shape, state))
#                 except Exception:
#                     pass
            
    
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
        # print(section_id)
        # print('hihi')
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
            kwargs = self.check_kwargs(**kwargs)

        return kwargs

        ##
        
    def check_kwargs(self,**kwargs):

        if 'recent' in kwargs.keys():
            kwargs['recent'] = self.str2bool(kwargs['recent'])
            
            
        if 'resize' in kwargs.keys():
        # if hasattr(kwargs, 'resize'):
            kwargs['resize'] = self.str2bool(kwargs['resize'])
        
        if 'cent_remove' in kwargs.keys():
        # if hasattr(kwargs, 'cent_remove'):
            kwargs['cent_remove'] = self.str2bool(kwargs['cent_remove'])
        
        if 'ran_sub' in kwargs.keys():
        # if hasattr(kwargs, 'ran_sub'):
            kwargs['ran_sub'] = self.str2bool(kwargs['ran_sub'])
    
        if 'para_sort' in kwargs.keys():
        # if hasattr(kwargs, 'para_sort'):
            kwargs['para_sort'] = self.str2bool(kwargs['para_sort'])
    
        if 'inner_pix' in kwargs.keys():
        # if hasattr(kwargs, 'inner_pix'):
            kwargs['inner_pix'] = self.str2bool(kwargs['inner_pix'])

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
    
        
    def str2bool(self,v):
        return v.lower() in ("yes", "true", "t", "1")
  
def run(config,force_replace=False):
    """
    Delegates the execution to `workflow.run`
    
    :param config:
    :param force_replace:
    """
    ws = workflow.run(config, force_replace)
    return ws
    
def restore(dirin):
    """
    Delegates the execution to `workflow.restore`
    
    :param dirin:
    """
    ws = workflow.restore(dirin)
    return ws