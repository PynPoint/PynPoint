"""
Modules for preparing the PSF subtraction.
"""

from __future__ import division

import numpy as np
from scipy import ndimage

from PynPoint.core.Processing import ProcessingModule


class PSFdataPreparation(ProcessingModule):
    """
    Module to prepare the data for PSF subtraction with PCA. The preparation steps include
    resizing, masking, and image normalization.
    """

    def __init__(self,
                 name_in=None,
                 image_in_tag="im_arr",
                 image_out_tag="im_arr",
                 image_mask_out_tag="im_mask_arr",
                 mask_out_tag="mask_arr",
                 resize=False,
                 cent_remove=True,
                 F_final=2.0,
                 cent_size=0.05,
                 edge_size=1.0):

        """
        Constructor of PSFdataPreparation.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry with images that is written as output.
        :type image_out_tag: str
        :param image_mask_out_tag: Tag of the database entry with the mask that is written as
                                   output.
        :type image_mask_out_tag: str
        :param mask_out_tag: Tag of the database entry with the mask that is written as output.
        :type mask_out_tag: str
        :param resize: Resize the data by a factor F_final.
        :type resize: bool
        :param cent_remove: Mask the central region of the data with a fractional mask radius of
                            cent_size.
        :type cent_remove: bool
        :param F_final: Factor by which the data is resized. F_final=2. will upsample the data by a
                        factor of two.
        :type F_final: float
        :param para_sort: Dummy argument.
        :type para_sort: bool
        :param cent_size: Fractional radius of the central mask relative to the image size.
        :type cent_size: float
        :param edge_size: Fractional outer radius relative to the image size. The images are
                          masked beyond this radius. Currently this parameter is not used.
        :type edge_size: float
        :return: None
        """

        super(PSFdataPreparation, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_mask_out_port = self.add_output_port(image_mask_out_tag)
        self.m_mask_out_port = self.add_output_port(mask_out_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        # Note recentering is not longer supported
        self.m_resize = resize
        self.m_cent_remove = cent_remove
        self.m_f_final = F_final
        self.m_cent_size = cent_size
        self.m_edge_size = edge_size

    @staticmethod
    def _im_norm(im_data_in):
        """
        Static method which normalizes the input data with its Frobenius norm.
        """

        im_norm = np.linalg.norm(im_data_in, ord="fro", axis=(1, 2))

        for i in range(0, len(im_data_in[:, 0, 0])):
            im_data_in[i, ] /= im_norm[i]

        return im_norm

    def _im_resizing(self,
                     im_data_in):
        """
        Internal method which resamples the data with a factor F_final, using a spline
        interpolation of the fifth order.
        """

        x_num_final, y_num_final = int(im_data_in.shape[1] * self.m_f_final), \
                                   int(im_data_in.shape[2] * self.m_f_final)

        im_arr_res = np.zeros([im_data_in.shape[0], x_num_final, y_num_final])

        for i in range(0, im_data_in.shape[0]):
            im_tmp = im_data_in[i]
            im_tmp = ndimage.interpolation.zoom(im_tmp, \
                                                [self.m_f_final, \
                                                 self.m_f_final], \
                                                 order=5)
            im_arr_res[i,] = im_tmp

        return im_arr_res

    def _im_masking(self,
                    im_data_in):
        """
        Internal method which masks the central and outer parts of the images.
        """

        def mk_circle_func(center_x, center_y):
            """sets up a function for calculating the radius to x,y (after having been initialised
            with x_cent and y_cent) """
            return lambda x, y: np.sqrt((center_x-x)**2 +(center_y-y)**2)

        def mk_circle(x_num, y_num, x_cent, y_cent, rad_lim):
            """function for making a circular aperture"""
            y_val, x_val = np.indices([x_num, y_num])
            rad = mk_circle_func(x_cent, y_cent)(x_val, y_val)
            i, j = np.where(rad <= rad_lim)
            mask_base = np.ones((x_num, y_num), float)
            mask_base[i, j] = 0.0
            return mask_base

        im_size = im_data_in[0, ].shape

        # TODO add edge_size as outer radius for the mask
        if self.m_cent_remove:

            mask_c = mk_circle(im_size[0],
                               im_size[1],
                               im_size[0]/2.,
                               im_size[1]/2.,
                               self.m_cent_size * im_size[0])

            mask_outside = mk_circle(im_size[0],
                                     im_size[1],
                                     im_size[0]/2.,
                                     im_size[1]/2.,
                                     0.5 * im_size[0])

            cent_mask = mask_c * (1.0 - mask_outside)
            res_cent_mask = (1.0 - cent_mask)
            im_arr_i_mask = im_data_in * res_cent_mask
            self.m_image_mask_out_port.set_all(im_arr_i_mask)

            im_arr_o_mask = im_data_in * cent_mask

            self.m_mask_out_port.set_all(cent_mask)
            return im_arr_o_mask

        else:
            cent_mask = np.ones(im_size)

        self.m_mask_out_port.set_all(cent_mask)
        return im_data_in

    def run(self):
        """
        Run method of the module. Normalizes, resizes and masks the images.

        :return: None
        """

        im_data = self.m_image_in_port.get_all()

        # image normalization
        im_norm = self._im_norm(im_data)

        # image resizing
        if self.m_resize:
            im_data = self._im_resizing(im_data)

        # image masking
        im_data = self._im_masking(im_data)

        self.m_image_out_port.set_all(im_data,
                                      keep_attributes=True)

        self.m_image_out_port.add_attribute("im_norm",
                                            im_norm,
                                            static=False)

        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        # save attributes
        attributes = {"cent_remove": self.m_cent_remove,
                      "resize": self.m_resize,
                      "F_final": float(self.m_f_final),
                      "cent_size": float(self.m_cent_size),
                      "edge_size": float(self.m_edge_size)}

        for key, value in attributes.iteritems():
            self.m_image_out_port.add_attribute(key,
                                                value,
                                                static=True)
