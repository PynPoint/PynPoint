"""
Modules to prepare the data for the PSF subtraction.
"""

from __future__ import division

import sys

import numpy as np

from scipy import ndimage

from PynPoint.Util.Progress import progress
from PynPoint.Core.Processing import ProcessingModule


class PSFpreparationModule(ProcessingModule):
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
                 norm=True,
                 resize=-1,
                 cent_remove=True,
                 cent_size=0.05,
                 edge_size=1.0,
                 **kwargs):
        """
        Constructor of PSFpreparationModule.

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
        :param norm: Normalization of each image by its Frobenius norm.
        :type norm: bool
        :param resize: Factor by which the data is resized. For example, if *resize* is 2 then
                       the data will be upsampled by a factor of two. No resizing is applied
                       with a negative value.
        :type resize: float
        :param cent_remove: Mask the central region of the data with a fractional mask radius of
                            cent_size.
        :type cent_remove: bool
        :param cent_size: Fractional radius of the central mask relative to the image size.
        :type cent_size: float
        :param edge_size: Fractional outer radius relative to the image size. The images are
                          masked beyond this radius. Currently this parameter is not used.
        :type edge_size: float
        :param \**kwargs:
            See below.

        :Keyword arguments:
             * **verbose** (*bool*) -- Print progress to the standard output.

        :return: None
        """

        if "verbose" in kwargs:
            self.m_verbose = kwargs["verbose"]
        else:
            self.m_verbose = True

        super(PSFpreparationModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        if image_mask_out_tag is not None:
            self.m_image_mask_out_port = self.add_output_port(image_mask_out_tag)
        if mask_out_tag is not None:
            self.m_mask_out_port = self.add_output_port(mask_out_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_image_mask_out_tag = image_mask_out_tag
        self.m_mask_out_tag = mask_out_tag

        self.m_resize = resize
        self.m_cent_remove = cent_remove
        self.m_cent_size = cent_size
        self.m_edge_size = edge_size
        self.m_norm = norm

    def _im_norm(self,
                 im_data_in):
        """
        Internal method which normalizes the input data by its Frobenius norm.
        """

        if self.m_norm:
            im_norm = np.linalg.norm(im_data_in, ord="fro", axis=(1, 2))
            for i in range(im_data_in.shape[0]):
                im_data_in[i, ] /= im_norm[i]

        else:
            im_norm = np.ones(im_data_in.shape)

        return im_norm

    def _im_resizing(self,
                     im_data_in):
        """
        Internal method which resamples the data with a factor *resize*, using a spline
        interpolation of the fifth order.
        """

        x_num_final, y_num_final = int(im_data_in.shape[1] * self.m_resize), \
                                   int(im_data_in.shape[2] * self.m_resize)

        im_arr_res = np.zeros([im_data_in.shape[0], x_num_final, y_num_final])

        for i in range(im_data_in.shape[0]):
            im_tmp = im_data_in[i]
            im_tmp = ndimage.interpolation.zoom(im_tmp,
                                                [self.m_resize, self.m_resize],
                                                order=5)
            im_arr_res[i,] = im_tmp

        return im_arr_res

    def _im_masking(self,
                    im_data_in):
        """
        Internal method which masks the central and outer parts of the images.
        """

        def mk_circle_func(x_center, y_center):
            """
            Function for calculating the radius to (x, y) after having been initialized with
            x_center and y_center.
            """

            return lambda x, y: np.sqrt((x_center-x)**2 +(y_center-y)**2)

        def mk_circle(x_num, y_num, x_cent, y_cent, rad_lim):
            """
            Function for making a circular aperture.
            """

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
            im_arr_o_mask = im_data_in * cent_mask

            if self.m_image_mask_out_tag is not None:
                self.m_image_mask_out_port.set_all(im_arr_i_mask)

            im_data_out = im_arr_o_mask

        else:
            cent_mask = np.ones(im_size)
            im_data_out = im_data_in

        if self.m_mask_out_tag is not None:
            self.m_mask_out_port.set_all(cent_mask)

        return im_data_out

    def run(self):
        """
        Run method of the module. Normalizes, resizes, and masks the images.

        :return: None
        """

        if self.m_verbose:
            sys.stdout.write("Running PSFpreparationModule...")
            sys.stdout.flush()

        im_data = self.m_image_in_port.get_all()
        im_norm = self._im_norm(im_data)

        if self.m_resize > 0.:
            im_data = self._im_resizing(im_data)

        im_data = self._im_masking(im_data)

        self.m_image_out_port.set_all(im_data, keep_attributes=True)
        self.m_image_out_port.add_attribute("im_norm", im_norm, static=False)
        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)

        attributes = {"resize": float(self.m_resize),
                      "cent_remove": self.m_cent_remove,
                      "cent_size": float(self.m_cent_size),
                      "edge_size": float(self.m_edge_size)}

        for key, value in attributes.iteritems():
            self.m_image_out_port.add_attribute(key, value, static=True)

        if self.m_verbose:
            sys.stdout.write(" [DONE]\n")
            sys.stdout.flush()


class AngleCalculationModule(ProcessingModule):
    """
    Module for calculating the parallactic angle values by interpolating between the begin and end
    value of a data cube.
    """

    def __init__(self,
                 name_in="angle_calculation",
                 data_tag="im_arr"):
        """
        Constructor of AngleCalculationModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param data_tag: Tag of the database entry for which the parallactic angles are written as
                         attributes.
        :type data_tag: str

        :return: None
        """

        super(AngleCalculationModule, self).__init__(name_in)

        self.m_data_in_port = self.add_input_port(data_tag)
        self.m_data_out_port = self.add_output_port(data_tag)

    def run(self):
        """
        Run method of the module. Calculates the parallactic angles of each frame by linearly
        interpolating between the start and end values of the data cubes. The values are written
        as attributes to *data_tag*.

        :return: None
        """

        parang_start = self.m_data_in_port.get_attribute("PARANG_START")
        parang_end = self.m_data_in_port.get_attribute("PARANG_END")

        steps = self.m_data_in_port.get_attribute("NFRAMES")
        ndit = self.m_data_in_port.get_attribute("NDIT")

        if False in ndit == steps:
            warnings.warn("There is a mismatch between the NDIT and NAXIS3 values. The parallactic"
                          "angles are calculated with a linear interpolation by using NAXIS3 "
                          "steps. A frame selection should be applied after the parallactic "
                          "angles are calculated.")

        new_angles = []

        for i in range(len(parang_start)):
            progress(i, len(parang_start), "Running AngleCalculationModule...")

            new_angles = np.append(new_angles,
                                   np.linspace(parang_start[i],
                                               parang_end[i],
                                               num=steps[i]))

        sys.stdout.write("Running AngleCalculationModule... [DONE]\n")
        sys.stdout.flush()

        self.m_data_out_port.add_attribute("PARANG",
                                           new_angles,
                                           static=False)
