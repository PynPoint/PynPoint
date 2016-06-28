from PynPoint.WrapperUtils import BasePynpointWrapper
from PynPoint.DataIO import InputPort


class BasisWrapper(BasePynpointWrapper):

    def __init__(self,
                 working_pypeline):

        super(BasisWrapper, self).__init__(working_pypeline)

        # needed for data export (we want to get rid of the identification numbers used for the
        # image instances
        self._m_tag_root_image = "basis_arr"
        self._m_tag_root_mask_image = "basis_mask_arr"
        self._m_tag_root_mask = "basis_cent_mask"

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

        BasisWrapper.class_counter += 1

    @classmethod
    def create_wdir(cls,
                    dir_in,
                    **kwargs):

        obj = super(BasisWrapper, cls).create_wdir(dir_in,
                                                   **kwargs)
        obj.mk_basis_set()
        return obj

    @classmethod
    def create_restore(cls,
                       filename,
                       pypline_working_place=None):

        obj = super(BasisWrapper, cls).create_restore(filename,
                                                      pypline_working_place)
        obj.mk_basis_set()
        return obj

    @classmethod
    def create_wfitsfiles(cls, files,**kwargs):
        pass

    @classmethod
    def create_whdf5input(cls, file_in,**kwargs):
        pass

    def mk_basis_set(self):
        pass

    def mk_orig(self,ind):
        pass

    def mk_psfmodel(self, num):
        pass