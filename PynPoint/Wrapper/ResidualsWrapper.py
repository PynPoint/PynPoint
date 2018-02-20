import numpy as np

from PynPoint.Wrapper.BasisWrapper import BasisWrapper
from PynPoint.Wrapper.ImageWrapper import ImageWrapper
from PynPoint.Core.DataIO import InputPort
from PynPoint.ProcessingModules.PSFSubtractionPCA import CreateResidualsModule


class ResidualsWrapper(object):
    class_counter = 1

    def __init__(self,
                 working_pypeline):
        self._pypeline = working_pypeline

        self._m_res_arr_root = "res_arr"
        self._m_res_rot_root = "res_rot"
        self._m_res_mean_root = "res_mean"
        self._m_res_median_root = "res_median"
        self._m_res_var_root = "res_var"
        self._m_res_rot_mean_clip_root = "res_rot_mean_clip"

        self._m_res_arr = self._m_res_arr_root + str(ResidualsWrapper.class_counter).zfill(2)
        self._m_res_arr_port = InputPort(self._m_res_arr)
        self._m_res_arr_port.set_database_connection(self._pypeline.m_data_storage)

        self._m_res_rot = self._m_res_rot_root + str(ResidualsWrapper.class_counter).zfill(2)
        self._m_res_rot_port = InputPort(self._m_res_rot)
        self._m_res_rot_port.set_database_connection(self._pypeline.m_data_storage)

        self._m_res_mean = self._m_res_mean_root + str(ResidualsWrapper.class_counter).zfill(2)
        self._m_res_mean_port = InputPort(self._m_res_mean)
        self._m_res_mean_port.set_database_connection(self._pypeline.m_data_storage)

        self._m_res_median = self._m_res_median_root + str(ResidualsWrapper.class_counter).zfill(2)
        self._m_res_median_port = InputPort(self._m_res_median)
        self._m_res_median_port.set_database_connection(self._pypeline.m_data_storage)

        self._m_res_var = self._m_res_var_root + str(ResidualsWrapper.class_counter).zfill(2)
        self._m_res_var_port = InputPort(self._m_res_var)
        self._m_res_var_port.set_database_connection(self._pypeline.m_data_storage)

        self._m_res_rot_mean_clip = self._m_res_rot_mean_clip_root \
                                    + str(ResidualsWrapper.class_counter).zfill(2)
        self._m_res_rot_mean_clip_port = InputPort(self._m_res_rot_mean_clip)
        self._m_res_rot_mean_clip_port.set_database_connection(self._pypeline.m_data_storage)

        ResidualsWrapper.class_counter += 1

        self._m_basis = None
        self._m_images = None

    def __getattr__(self, item):
        data_bases = {"im_arr": self._m_images.im_arr,
                      "cent_mask": self._m_images.cent_mask,
                      "im_arr_mask": self._m_images.im_arr_mask,
                      "psf_im_arr": self._m_images.psf_im_arr}

        if item in data_bases:
            return data_bases[item]

    @classmethod
    def create_restore(cls,
                       filename):

        image = ImageWrapper.create_restore(filename)
        basis = BasisWrapper.create_restore(filename)

        tmp_pypeline = image._pypeline

        obj = cls(tmp_pypeline)

        obj._m_basis = basis
        obj._m_images = image

        return obj

    def save(self,
             filename):
        # save image
        self._m_images.save(filename)

        # save basis
        self._m_basis.save(filename)


    @classmethod
    def create_winstances(cls,
                          images,
                          basis):

        tmp_pypeline = images._pypeline

        obj = cls(tmp_pypeline)

        obj._m_basis = basis
        obj._m_images = images

        # Input Ports to return results

        assert np.array_equal(basis.cent_mask,
                              images.cent_mask)
        assert np.array_equal(basis.psf_basis[0, ].shape,
                              images.im_arr[0, ].shape)

        return obj

    def _mk_result(self,
                   extra_rot_in=0.0):

        if "res_module" in self._pypeline._m_modules:
            return

        res_module = CreateResidualsModule(name_in="res_module",
                                           im_arr_in_tag=self._m_images._m_image_data_tag,
                                           psf_im_in_tag=self._m_images._m_psf_image_arr_tag,
                                           mask_in_tag=self._m_images._m_mask_tag,
                                           res_arr_out_tag=self._m_res_arr,
                                           res_arr_rot_out_tag=self._m_res_rot,
                                           res_mean_tag=self._m_res_mean,
                                           res_median_tag=self._m_res_median,
                                           res_var_tag=self._m_res_var,
                                           res_rot_mean_clip_tag=self._m_res_rot_mean_clip,
                                           extra_rot=extra_rot_in)
        self._pypeline.add_module(res_module)
        self._pypeline.run_module("res_module")

    def res_arr(self,
                num_coeff):

        # check if psf image array was calculated
        if self._m_images.psf_im_arr is None:
            self.mk_psfmodel(num_coeff)

        self._mk_result()

        return self._m_res_arr_port.get_all()

    def res_rot(self,
                num_coeff,
                extra_rot=0.0):
        # check if psf image array was calculated
        if self._m_images.psf_im_arr is None:
            self.mk_psfmodel(num_coeff)

        self._mk_result(extra_rot)

        return self._m_res_rot_port.get_all()

    def res_rot_mean(self,
                     num_coeff,
                     extra_rot=0.0):
        self.res_rot(num_coeff=num_coeff,
                     extra_rot=extra_rot)

        return self._m_res_mean_port.get_all()

    def res_rot_median(self,
                       num_coeff,
                       extra_rot=0.0):
        self.res_rot(num_coeff=num_coeff,
                     extra_rot=extra_rot)

        return self._m_res_median_port.get_all()

    def res_rot_mean_clip(self,
                          num_coeff,
                          extra_rot=0.0):
        self.res_rot(num_coeff=num_coeff,
                     extra_rot=extra_rot)

        return self._m_res_rot_mean_clip_port.get_all()

    def res_rot_var(self,
                    num_coeff,
                    extra_rot=0.0):
        self.res_rot(num_coeff=num_coeff,
                     extra_rot=extra_rot)

        return self._m_res_var_port.get_all()

    def _psf_im(self,
                num_coeff):

        if self._m_images.psf_im_arr is None:
            self.mk_psfmodel(num_coeff)

        return self._m_images.psf_im_arr

    def mk_psfmodel(self, num):
        self._m_images.mk_psfmodel(self._m_basis,
                                   num)
