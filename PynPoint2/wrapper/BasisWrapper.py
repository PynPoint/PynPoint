from PynPoint2.wrapper.WrapperUtils import BasePynpointWrapper
from PynPoint2.core.DataIO import InputPort
from PynPoint2.processing_modules.PSFSubtraction import MakePCABasisModule


class BasisWrapper(BasePynpointWrapper):

    def __init__(self,
                 working_pypeline):

        super(BasisWrapper, self).__init__(working_pypeline)

        # needed for data export (we want to get rid of the identification numbers used for the
        # image instances
        self._m_tag_root_image = "basis_arr"
        self._m_tag_root_mask_image = "basis_mask_arr"
        self._m_tag_root_mask = "basis_cent_mask"
        self._m_tag_root_basis = "psf_basis"
        self._m_tag_root_im_average = "basis_im_ave"
        self._m_tag_root_psf_image_arr = "basis_psf_im_arr"

        # In the old PynPoint it was possible to create multiple image instances working on
        # separated data (in memory). Hence, every time a new ImageWrapper is created a new database
        # entry is required. (Using increasing identification numbers)
        self._m_image_data_tag = self._m_tag_root_image + str(BasisWrapper.class_counter).zfill(2)
        self._m_image_data_port = InputPort(self._m_image_data_tag)
        self._m_image_data_port.set_database_connection(working_pypeline.m_data_storage)

        self._m_image_data_masked_tag = self._m_tag_root_mask_image + \
                                        str(BasisWrapper.class_counter).zfill(2)
        self._m_image_data_masked_port = InputPort(self._m_image_data_masked_tag)
        self._m_image_data_masked_port.set_database_connection(working_pypeline.m_data_storage)

        self._m_mask_tag = self._m_tag_root_mask + str(BasisWrapper.class_counter).zfill(2)
        self._m_mask_port = InputPort(self._m_mask_tag)
        self._m_mask_port.set_database_connection(working_pypeline.m_data_storage)

        self._m_psf_image_arr_tag = self._m_tag_root_psf_image_arr + \
                                    str(BasisWrapper.class_counter).zfill(2)
        self._m_psf_image_arr_port = InputPort(self._m_psf_image_arr_tag)
        self._m_psf_image_arr_port.set_database_connection(working_pypeline.m_data_storage)

        # ONLY for Basis not for Image
        self._m_basis_tag = self._m_tag_root_basis \
                            + str(BasisWrapper.class_counter).zfill(2)
        self._m_basis_port = InputPort(self._m_basis_tag)
        self._m_basis_port.set_database_connection(working_pypeline.m_data_storage)

        self._m_im_average_tag = self._m_tag_root_im_average \
                                 + str(BasisWrapper.class_counter).zfill(2)
        self._m_im_average_port = InputPort(self._m_im_average_tag)
        self._m_im_average_port.set_database_connection(working_pypeline.m_data_storage)

        self._m_restore_tag_dict = {self._m_tag_root_image: self._m_image_data_tag,
                                    self._m_tag_root_mask_image: self._m_image_data_masked_tag,
                                    self._m_tag_root_mask: self._m_mask_tag,
                                    self._m_tag_root_basis: self._m_basis_tag,
                                    self._m_tag_root_im_average: self._m_im_average_tag,
                                    self._m_tag_root_psf_image_arr: self._m_psf_image_arr_tag}

        self._m_save_tag_dict = {self._m_image_data_tag: self._m_tag_root_image,
                                 self._m_image_data_masked_tag: self._m_tag_root_mask_image,
                                 self._m_mask_tag: self._m_tag_root_mask,
                                 self._m_basis_tag: self._m_tag_root_basis,
                                 self._m_im_average_tag: self._m_tag_root_im_average,
                                 self._m_psf_image_arr_tag: self._m_tag_root_psf_image_arr}

        BasisWrapper.class_counter += 1

    def __getattr__(self, item):
        res = super(BasisWrapper, self).__getattr__(item)

        if res is not None:
            return res

        data_bases = {"im_ave": self._m_im_average_port,
                      "psf_basis": self._m_basis_port}

        if item in data_bases:
            return data_bases[item].get_all()

        elif item == "psf_basis_type":
            return self._m_basis_port.get_attribute("basis_type")

    @classmethod
    def create_wdir(cls,
                    dir_in,
                    **kwargs):

        obj = super(BasisWrapper, cls).create_wdir(dir_in,
                                                   **kwargs)
        obj.mk_basis_set()
        return obj

    @classmethod
    def create_whdf5input(cls,
                          file_in,
                          pypline_working_place=None,
                          **kwargs):

        obj = super(BasisWrapper, cls).create_whdf5input(file_in,
                                                         pypline_working_place,
                                                         **kwargs)
        obj.mk_basis_set()

        return obj

    def mk_basis_set(self):

        basis_creation = MakePCABasisModule("basis_creation",
                                            im_arr_in_tag=self._m_image_data_tag,
                                            im_arr_out_tag=self._m_image_data_tag,
                                            im_average_out_tag=self._m_im_average_tag,
                                            basis_out_tag=self._m_basis_tag)

        self._pypeline.add_module(basis_creation)
        self._pypeline.run_module("basis_creation")

    def mk_orig(self,ind):
        pass

    def mk_psfmodel(self, num):

        # call the super function with own attributes (basis is a basis)
        super(BasisWrapper, self).mk_psfmodel(self, num)
