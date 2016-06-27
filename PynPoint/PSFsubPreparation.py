"""
Module for data Preparation before PSF Subtraction. Includes:
    - Resizing
    - Masking
    - Image normalization
"""

from __future__ import division

import numpy as np
from scipy import ndimage


from PynPoint.Processing import ProcessingModule


class PSFdataPreparation(ProcessingModule):

    def __init__(self,
                 name_in=None,
                 image_in_tag="im_arr",
                 image_out_tag="im_arr",
                 image_mask_out_tag="im_mask_arr",
                 mask_out_tag="mask_arr",
                 resize=False,
                 cent_remove=True,
                 F_final=2.0,
                 para_sort=True,
                 cent_size=0.05,
                 edge_size=1.0):

        super(PSFdataPreparation, self).__init__(name_in)

        # Note recentering is not longer supported
        self.m_resize = resize
        self.m_cent_remove = cent_remove
        self.m_f_final = F_final
        self.m_para_sort = para_sort
        self.m_cent_size = cent_size
        self.m_edge_size = edge_size

        self.m_image_in_tag = image_in_tag
        self.m_image_out_tag = image_out_tag
        self.m_image_mask_out_tag = image_mask_out_tag
        self.m_mask_out_tag = mask_out_tag

        # create Ports
        self.add_input_port(image_in_tag)
        self.add_output_port(image_mask_out_tag)
        self.add_output_port(mask_out_tag)
        self.add_output_port(image_out_tag)

    @staticmethod
    def _im_norm(im_data_in):

        im_norm = (im_data_in.sum(axis=1)).sum(axis=1)

        for i in range(0, len(im_data_in[:, 0, 0])):
            im_data_in[i, ] /= im_norm[i]

        return im_norm

    def _im_resizing(self,
                     im_data_in):
        x_num_final, y_num_final = int(im_data_in.shape[1] * self.m_f_final),\
                                        int(im_data_in.shape[2] * self.m_f_final)
        im_arr_res = np.zeros([im_data_in.shape[0], x_num_final, y_num_final])

        for i in range(0, im_data_in.shape[0]):
            im_tmp = im_data_in[i]
            im_tmp = ndimage.interpolation.zoom(im_tmp,
                                                [self.m_f_final,
                                                 self.m_f_final],
                                                order=5)
            im_arr_res[i,] = im_tmp

        return im_arr_res

    def _im_masking(self,
                    im_data_in):

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
            self._m_output_ports[self.m_image_mask_out_tag].set_all(im_arr_i_mask)

            im_arr_o_mask = im_data_in * cent_mask

            self._m_output_ports[self.m_mask_out_tag].set_all(cent_mask)
            return im_arr_o_mask

        else:
            cent_mask = np.ones(im_size)

        self._m_output_ports[self.m_mask_out_tag].set_all(cent_mask)
        return im_data_in

    def run(self):

        im_data = self._m_input_ports[self.m_image_in_tag].get_all()

        # image normalization
        im_norm = self._im_norm(im_data)
        self._m_output_ports[self.m_image_out_tag].append_attribute_data("im_norm",
                                                                         im_norm)

        # TODO para_sort

        # image resizing
        if self.m_resize:
            im_data = self._im_resizing(im_data)

        # image masking
        im_data = self._im_masking(im_data)

        self._m_output_ports[self.m_image_out_tag].set_all(im_data,
                                                           keep_attributes=True)

        # save attributes
        attributes = {"cent_remove": self.m_cent_remove,
                      "resize": self.m_resize,
                      "para_sort": self.m_para_sort,
                      "F_final": float(self.m_f_final),
                      "cent_size": float(self.m_cent_size),
                      "edge_size": float(self.m_edge_size)}

        for key, value in attributes.iteritems():
            self._m_output_ports[self.m_image_out_tag].add_attribute(key,
                                                                     value,
                                                                     static=True)
