# own modules
from PynPoint2.wrapper.WrapperUtils import BasePynpointWrapper
from PynPoint2.core.DataIO import InputPort


class ImageWrapper(BasePynpointWrapper):

    def __init__(self,
                 working_pypeline):

        super(ImageWrapper, self).__init__(working_pypeline)

        # needed for data export (we want to get rid of the identification numbers used for the
        # image instances
        self._m_tag_root_image = "im_arr"
        self._m_tag_root_mask_image = "im_mask_arr"
        self._m_tag_root_mask = "im_cent_mask"
        self._m_tag_root_psf_image_arr = "psf_im_arr"

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

        self._m_psf_image_arr_tag = self._m_tag_root_psf_image_arr + \
                                    str(ImageWrapper.class_counter).zfill(2)
        self._m_psf_image_arr_port = InputPort(self._m_psf_image_arr_tag)
        self._m_psf_image_arr_port.set_database_connection(working_pypeline.m_data_storage)

        self._m_restore_tag_dict = {self._m_tag_root_image: self._m_image_data_tag,
                                    self._m_tag_root_mask_image: self._m_image_data_masked_tag,
                                    self._m_tag_root_mask: self._m_mask_tag,
                                    self._m_tag_root_psf_image_arr: self._m_psf_image_arr_tag}

        self._m_save_tag_dict = {self._m_image_data_tag: self._m_tag_root_image,
                                 self._m_image_data_masked_tag: self._m_tag_root_mask_image,
                                 self._m_mask_tag: self._m_tag_root_mask,
                                 self._m_psf_image_arr_tag: self._m_tag_root_psf_image_arr}

        ImageWrapper.class_counter += 1

    def mk_psf_realisation(self,
                           ind,
                           full=False):
        """
        Function for making a realisation of the PSF using the data stored in the object

        :param ind: index of the image to be modelled
        :param full: if set to True then the masked region will be included
        :return: an image of the PSF model
        """
        im_temp = self.psf_im_arr[ind, ]
        if self.cent_remove is True:
            if full is True:
                im_temp = im_temp + self.im_arr_mask[ind,]
            elif full is False:
                im_temp = im_temp * self.cent_mask
        return im_temp

