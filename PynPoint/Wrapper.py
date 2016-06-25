import numpy as np
import warnings

# own modules
from PynPoint.Pypeline import Pypeline
from FitsReading import ReadFitsCubesDirectory
from PSFsubPrepatation import PSFdataPreparation

class BasePynpointWrapper(object):

    def __init__(self):
        pass

    def save(self,filename):
        pass

    def plt_im(self,ind):
        pass



class ImageWrapper(BasePynpointWrapper):

    class_counter = 1

    def __init__(self):
        self._m_center_remove = None
        self._m_resize = None
        self._m_ran_sub = None
        self._m_recent = None
        self._m_cent_size = None
        self._pypeline = None

        # In the old PynPoint it was possible to create multiple image instances working on
        # separated data (in memory). Hence, every time a new ImageWrapper is created a new database
        # entry is required
        self._m_image_data_tag = "im_arr" + str(ImageWrapper.class_counter)
        self._m_image_data_masked_tag = "im_mask_arr" + str(ImageWrapper.class_counter)
        self._m_mask_tag = "mask_arr" + str(ImageWrapper.class_counter)
        ImageWrapper.class_counter += 1

    def __getattr__(self, item):
        if item == "num_files":
            return self._pypeline.get_attribute(self._m_image_data_tag, "Num_Files")

        elif item == "files":
            return self._pypeline.get_data("header_" + str(self._m_image_data_tag) + "/Used_Files")

        elif item == "im_size":
            return (self._pypeline.get_data(self._m_image_data_tag).shape[1],
                    self._pypeline.get_data(self._m_image_data_tag).shape[2])

        elif item == "cent_remove":
            return self._m_center_remove

        elif item == "im_arr":
            return self._pypeline.get_data(self._m_image_data_tag)

        elif item == "im_norm":
            return self._pypeline.get_data("header_" + str(self._m_image_data_tag) + "/im_norm")

        elif item == "para":
            return self._pypeline.get_data("header_" + str(self._m_image_data_tag) + "/NEW_PARA")

        elif item == "cent_mask":
            return self._pypeline.get_data(self._m_mask_tag)

        elif item == "im_arr_mask":
            return self._pypeline.get_data(self._m_image_data_masked_tag)


    @classmethod
    def create_wdir(cls, dir_in, **kwargs):

        obj = cls()

        # take all attributes
        if "ran_sub" in kwargs:
            obj._m_ran_sub = kwargs["ran_sub"]

        if "cent_remove" in kwargs:
            obj._m_center_remove = kwargs["cent_remove"]

        if "resize" in kwargs:
            obj._m_resize = kwargs["resize"]

        if "recent" in kwargs:
            warnings.warn('Recentering is not longer supported in PynPoint preparation')
            obj._m_recent = kwargs["recent"]

        if "cent_size" in kwargs:
            obj._m_cent_size = kwargs["cent_size"]

        obj._pypeline = Pypeline(dir_in,
                                 dir_in,
                                 dir_in)

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
                                         cent_size=obj._m_cent_size,
                                         ran_sub=obj._m_ran_sub)

        # TODO check all parameters

        obj._pypeline.add_module(reading)
        obj._pypeline.add_module(preparation)
        obj._pypeline.run()
        return obj

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