import numpy as np
import warnings
import os

# own modules
from PynPoint.WrapperUtils import BasePynpointWrapper

from DataIO import InputPort


class ImageWrapper(BasePynpointWrapper):

    def __init__(self,
                 working_pypeline):

        super(ImageWrapper, self).__init__(working_pypeline)

        # needed for data export (we want to get rid of the identification numbers used for the
        # image instances
        self._m_tag_root_image = "im_arr"
        self._m_tag_root_mask_image = "im_mask_arr"
        self._m_tag_root_mask = "im_cent_mask"

        # In the old PynPoint it was possible to create multiple image instances working on
        # separated data (in memory). Hence, every time a new ImageWrapper is created a new database
        # entry is required. (Using increasing identification numbers)
        self._m_image_data_tag = self._m_tag_root_image + str(ImageWrapper.class_counter).zfill(2)
        self._m_image_data_port = InputPort(self._m_image_data_tag)
        self._m_image_data_port.set_database_connection(working_pypeline.m_data_storage)

        self._m_image_data_masked_tag = self._m_tag_root_mask_image + \
                                        str(ImageWrapper.class_counter).zfill(2)
        self._m_image_data_masked_port = InputPort(self._m_image_data_masked_tag)
        self._m_image_data_masked_port.set_database_connection(working_pypeline.m_data_storage)

        self._m_mask_tag = self._m_tag_root_mask + str(ImageWrapper.class_counter).zfill(2)
        self._m_mask_port = InputPort(self._m_mask_tag)
        self._m_mask_port.set_database_connection(working_pypeline.m_data_storage)

        ImageWrapper.class_counter += 1

    def plt_im(self,ind):
        pass

    @classmethod
    def create_wfitsfiles(cls, *args,**kwargs):
        pass

    def mk_psf_realisation(self,ind,full=False):
        pass

    @classmethod
    def create_whdf5input(cls,
                          file_in,
                          **kwargs):
        pass

