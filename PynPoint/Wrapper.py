import numpy as np
import warnings
import os

# own modules
from PynPoint.Pypeline import Pypeline
from FitsReading import ReadFitsCubesDirectory
from PSFsubPreparation import PSFdataPreparation
from Hdf5Writing import Hdf5WritingModule
from DataIO import InputPort

class BasePynpointWrapper(object):

    def __init__(self):
        pass

    def save(self, filename):
        pass


    def plt_im(self,ind):
        pass


class ImageWrapper(object):

    class_counter = 1

    def __init__(self):
        self._m_center_remove = True
        self._m_resize = False
        self._m_ran_sub = None # TODO implement in some module
        self._m_cent_size = 0.05
        self._m_edge_size = 1.0
        self._m_f_final = 2.0
        self._pypeline = None

        # In the old PynPoint it was possible to create multiple image instances working on
        # separated data (in memory). Hence, every time a new ImageWrapper is created a new database
        # entry is required
        self._m_image_data_tag = "im_arr" + str(ImageWrapper.class_counter).zfill(2)
        self._m_image_data_port = InputPort(self._m_image_data_tag)
        self._m_image_data_masked_tag = "im_mask_arr" + str(ImageWrapper.class_counter).zfill(2)
        self._m_image_data_masked_port = InputPort(self._m_image_data_masked_tag)
        self._m_mask_tag = "mask_arr" + str(ImageWrapper.class_counter).zfill(2)
        self._m_mask_port = InputPort(self._m_mask_tag)
        ImageWrapper.class_counter += 1

    def __getattr__(self, item):

        # All static and non static attributes and their names in the database
        # {#Name_seen_from_outside: #database_name}
        simple_attributes = {"num_files" : "Num_Files",
                             "files" : "Used_Files",
                             "im_norm" : "im_norm",
                             "para" : "NEW_PARA",
                             "cent_remove" : "cent_remove",
                             "resize" : "resize",
                             "para_sort" : "para_sort",
                             "F_final" : "F_final",
                             "cent_size" : "cent_size",
                             "edge_size" : "edge_size"}

        data_bases = {"im_arr": self._m_image_data_port,
                      "cent_mask": self._m_mask_port,
                      "im_arr_mask": self._m_image_data_masked_port}

        if item in simple_attributes:
            print self._m_image_data_port.get_attribute(simple_attributes[item])
            print type(self._m_image_data_port.get_attribute(simple_attributes[item]))
            print type(True)
            return self._m_image_data_port.get_attribute(simple_attributes[item])

        elif item in data_bases:
            return data_bases[item].get_all()

        elif item == "im_size":
            return (self._pypeline.get_data(self._m_image_data_tag).shape[1],
                    self._pypeline.get_data(self._m_image_data_tag).shape[2])

    @classmethod
    def create_wdir(cls, dir_in, **kwargs):

        obj = cls()

        if "cent_remove" in kwargs:
            obj._m_center_remove = kwargs["cent_remove"]

        if "resize" in kwargs:
            obj._m_resize = kwargs["resize"]

        if "recent" in kwargs:
            warnings.warn('Recentering is not longer supported in PynPoint preparation')

        if "cent_size" in kwargs:
            obj._m_cent_size = kwargs["cent_size"]

        if "F_final" in kwargs:
            obj._m_f_final = kwargs["F_final"]

        if "edge_size" in kwargs:
            obj._m_edge_size = kwargs["edge_size"]

        obj._pypeline = Pypeline(dir_in,
                                 dir_in,
                                 dir_in)

        # connect Ports
        obj._m_image_data_port.set_database_connection(obj._pypeline.m_data_storage)
        obj._m_mask_port.set_database_connection(obj._pypeline.m_data_storage)
        obj._m_image_data_masked_port.set_database_connection(obj._pypeline.m_data_storage)

        reading = ReadFitsCubesDirectory(name_in="reading_mod",
                                         input_dir=dir_in,
                                         image_tag=obj._m_image_data_tag)

        preparation = PSFdataPreparation(name_in="prep",
                                         image_in_tag=obj._m_image_data_tag,
                                         image_mask_out_tag=obj._m_image_data_masked_tag,
                                         mask_out_tag=obj._m_mask_tag,
                                         image_out_tag=obj._m_image_data_tag,
                                         resize=obj._m_resize,
                                         cent_remove=obj._m_center_remove,
                                         F_final=obj._m_f_final,
                                         cent_size=obj._m_cent_size)

        obj._pypeline.add_module(reading)
        obj._pypeline.add_module(preparation)
        obj._pypeline.run()
        return obj

    def save(self, filename):

        filename = str(filename)

        if os.path.isfile(filename):
            warnings.warn('The file %s have been overwritten' % filename)

        head, tail = os.path.split(filename)

        dictionary = {self._m_image_data_tag: "im_arr",
                      self._m_image_data_masked_tag: "im_mask_arr",
                      self._m_mask_tag: "mask_arr"}

        writing = Hdf5WritingModule("hdf5_writing",
                                    tail,
                                    output_dir=head,
                                    tag_dictionary=dictionary,
                                    keep_attributes=True)

        self._pypeline.add_module(writing)

        self._pypeline.run_module("hdf5_writing")

    def plt_im(self,ind):
        pass

    @classmethod
    def create_whdf5input(cls, file_in,**kwargs):
        pass

    @classmethod
    def create_restore(cls, filename):
        pass

    @classmethod
    def create_wfitsfiles(cls, *args,**kwargs):
        pass

    def mk_psf_realisation(self,ind,full=False):
        pass


class ResidualsWrapper(BasePynpointWrapper):

    def __init__(self):
        pass

    @classmethod
    def create_restore(cls, filename):
        pass

    @classmethod
    def create_winstances(cls, images,basis):
        pass

    def res_arr(self,num_coeff):
        pass

    def res_rot(self,num_coeff,extra_rot =0.0):
        pass

    def res_rot_mean(self,num_coeff,extra_rot =0.0):
        pass

    def res_rot_median(self,num_coeff,extra_rot =0.0):
        pass

    def res_rot_mean_clip(self,num_coeff,extra_rot =0.0):
        pass

    def res_rot_var(self,num_coeff,extra_rot = 0.0):
        pass

    def _psf_im(self,num_coeff):
        pass

    def mk_psfmodel(self, num):
        pass


class BasisWrapper(BasePynpointWrapper):

    def __init__(self):
        pass

    @classmethod
    def create_wdir(cls, dir_in,**kwargs):
        pass

    @classmethod
    def create_whdf5input(cls, file_in,**kwargs):
        pass

    @classmethod
    def create_restore(cls, filename):
        pass

    @classmethod
    def create_wfitsfiles(cls, files,**kwargs):
        pass

    def mk_basis_set(self):
        pass

    def mk_orig(self,ind):
        pass

    def mk_psfmodel(self, num):
        pass