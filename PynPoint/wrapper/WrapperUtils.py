# the only way to do this is to access private members

import os
import warnings

from PynPoint.io_modules.Hdf5Reading import Hdf5ReadingModule
from PynPoint.io_modules.Hdf5Writing import Hdf5WritingModule
from PynPoint.processing_modules.PSFsubPreparation import PSFdataPreparation
from PynPoint.processing_modules.StackingAndSubsampling import StackAndSubsetModule

from PynPoint.core.Pypeline import Pypeline
from PynPoint.io_modules.FitsReading import FitsReadingModule
from PynPoint.processing_modules.PSFSubtractionPCA import MakePSFModelModule

warnings.simplefilter("always")


class BasePynpointWrapper(object):
    class_counter = 1

    def __init__(self,
                 working_pypeline):
        self._pypeline = working_pypeline
        self._m_center_remove = True
        self._m_resize = False
        self._m_ran_sub = None
        self._m_stacking = None
        self._m_cent_size = 0.05
        self._m_edge_size = 1.0
        self._m_f_final = 2.0

        # Attributes / Ports set individually by Image and Basis
        self._m_image_data_tag = None
        self._m_image_data_masked_tag = None
        self._m_mask_tag = None

        self._m_image_data_port = None
        self._m_image_data_masked_port = None
        self._m_mask_port = None

        self._m_tag_root_image = None
        self._m_tag_root_mask_image = None
        self._m_tag_root_mask = None

        self._m_psf_image_arr_tag = None

        # containing all entry to be saved and restored (set by Basis and Image individually)
        self._m_restore_tag_dict = {}
        self._m_save_tag_dict = {}

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
                      "im_arr_mask": self._m_image_data_masked_port,
                      "psf_im_arr": self._m_psf_image_arr_port}

        if item in simple_attributes:
            return self._m_image_data_port.get_attribute(simple_attributes[item])

        elif item in data_bases:
            return data_bases[item].get_all()

        if item == "_psf_coeff":
            return self._m_psf_image_arr_port.get_attribute("psf_coeff")

        elif item == "im_size":
            return (self._m_image_data_port.get_all().shape[1],
                    self._m_image_data_port.get_all().shape[2])

    @classmethod
    def create_wdir(cls,
                    dir_in,
                    **kwargs):

        tmp_pypeline = Pypeline(dir_in,
                                dir_in,
                                dir_in)

        obj = cls(tmp_pypeline)

        obj._save_kwargs(**kwargs)

        reading = FitsReadingModule(name_in="reading_mod",
                                    input_dir=dir_in,
                                    image_tag=obj._m_image_data_tag)

        obj._pypeline.add_module(reading)
        obj._pypeline.run_module("reading_mod")
        obj._prepare_data()

        return obj

    def _save_kwargs(self,
                     **kwargs):
        if "cent_remove" in kwargs:
            self._m_center_remove = kwargs["cent_remove"]

        if "resize" in kwargs:
            self._m_resize = kwargs["resize"]

        if "recent" in kwargs:
            warnings.warn('Recentering is not longer supported in PynPoint preparation')

        if "cent_size" in kwargs:
            self._m_cent_size = kwargs["cent_size"]

        if "F_final" in kwargs:
            self._m_f_final = kwargs["F_final"]

        if "edge_size" in kwargs:
            self._m_edge_size = kwargs["edge_size"]

        if "ran_sub" in kwargs:
            self._m_ran_sub = kwargs["ran_sub"]

        if "stackave" in kwargs:
            self._m_stacking = kwargs["stackave"]

    def _prepare_data(self):
        preparation = PSFdataPreparation(name_in="prep",
                                         image_in_tag=self._m_image_data_tag,
                                         image_mask_out_tag=self._m_image_data_masked_tag,
                                         mask_out_tag=self._m_mask_tag,
                                         image_out_tag=self._m_image_data_tag,
                                         resize=self._m_resize,
                                         cent_remove=self._m_center_remove,
                                         F_final=self._m_f_final,
                                         cent_size=self._m_cent_size)

        subsample_module = StackAndSubsetModule(name_in="stacking",
                                                image_in_tag=self._m_image_data_tag,
                                                image_out_tag=self._m_image_data_tag,
                                                random_subset=self._m_ran_sub,
                                                stacking=self._m_stacking)

        self._pypeline.add_module(preparation)
        self._pypeline.add_module(subsample_module)

        self._pypeline.run_module("prep")
        self._pypeline.run_module("stacking")

    def save(self,
             filename):

        filename = str(filename)

        if os.path.isfile(filename):
            warnings.warn('The file %s have been overwritten' % filename)

        head, tail = os.path.split(filename)

        writing = Hdf5WritingModule(tail,
                                    name_in="hdf5_writing",
                                    output_dir=head,
                                    tag_dictionary=self._m_save_tag_dict,
                                    keep_attributes=True)

        self._pypeline.add_module(writing)

        self._pypeline.run_module("hdf5_writing")

    @classmethod
    def create_restore(cls,
                       filename,
                       pypline_working_place=None):

        head, tail = os.path.split(filename)

        if pypline_working_place is None:
            working_place = head
        else:
            working_place = pypline_working_place

        tmp_pypeline = Pypeline(working_place,
                                head,
                                head)

        obj = cls(tmp_pypeline)

        reading = Hdf5ReadingModule("reading",
                                    input_filename=tail,
                                    input_dir=head,
                                    tag_dictionary=obj._m_restore_tag_dict)

        obj._pypeline.add_module(reading)
        obj._pypeline.run_module("reading")

        return obj

    @classmethod
    def create_wfitsfiles(cls, files,**kwargs):
        pass

    @classmethod
    def create_whdf5input(cls,
                          file_in,
                          pypline_working_place=None,
                          **kwargs):

        obj = cls.create_restore(file_in,
                                 pypline_working_place)

        obj._save_kwargs(**kwargs)

        obj._prepare_data()

        return obj

    def mk_psfmodel(self,
                     basis,
                     num):

        tmp_im_arr_in_tag = self._m_image_data_tag
        tmp_basis_tag = basis._m_basis_tag
        tmp_basis_average_in_tag = basis._m_im_average_tag
        tmp_psf_basis_out_tag = self._m_psf_image_arr_tag

        psf_model_module = MakePSFModelModule(num,
                                              name_in="psf_model_module",
                                              im_arr_in_tag=tmp_im_arr_in_tag,
                                              basis_in_tag=tmp_basis_tag,
                                              basis_average_in_tag=tmp_basis_average_in_tag,
                                              psf_basis_out_tag=tmp_psf_basis_out_tag)

        self._pypeline.add_module(psf_model_module)
        self._pypeline.run_module("psf_model_module")
