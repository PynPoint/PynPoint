import warnings
import os

from PynPoint.Pypeline import Pypeline
from PynPoint.FitsReading import ReadFitsCubesDirectory
from PynPoint.PSFsubPreparation import PSFdataPreparation
from PynPoint.Hdf5Writing import Hdf5WritingModule
from PynPoint.Hdf5Reading import Hdf5ReadingModule


class BasePynpointWrapper(object):
    class_counter = 1

    def __init__(self,
                 working_pypeline):
        self._pypeline = working_pypeline
        self._m_center_remove = True
        self._m_resize = False
        self._m_ran_sub = None # TODO implement in some module
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
            return self._m_image_data_port.get_attribute(simple_attributes[item])

        elif item in data_bases:
            return data_bases[item].get_all()

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

        dictionary = {self._m_image_data_tag: self._m_tag_root_image,
                      self._m_image_data_masked_tag: self._m_tag_root_mask_image,
                      self._m_mask_tag: self._m_tag_root_mask}

        writing = Hdf5WritingModule("hdf5_writing",
                                    tail,
                                    output_dir=head,
                                    tag_dictionary=dictionary,
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

        tag_dict = {obj._m_tag_root_image: obj._m_image_data_tag,
                    obj._m_tag_root_mask_image: obj._m_image_data_masked_tag,
                    obj._m_tag_root_mask: obj._m_mask_tag}

        reading = Hdf5ReadingModule("reading",
                                    input_filename=tail,
                                    input_dir=head,
                                    tag_dictionary=tag_dict)

        obj._pypeline.add_module(reading)
        obj._pypeline.run_module("reading")

        return obj

    def plt_im(self,ind):
        pass