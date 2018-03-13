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
                 mask_out_tag="mask_arr",
                 norm=True,
                 resize=None,
                 cent_size=None,
                 edge_size=None,
                 **kwargs):
        """
        Constructor of PSFpreparationModule.

        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry with images that is written as output.
        :type image_out_tag: str
        :param mask_out_tag: Tag of the database entry with the mask that is written as output.
        :type mask_out_tag: str
        :param norm: Normalization of each image by its Frobenius norm.
        :type norm: bool
        :param resize: Factor by which the data is resized. For example, if *resize* is 2 then
                       the data will be upsampled by a factor of two. No resizing is applied
                       when set to None.
        :type resize: float
        :param cent_size: Radius of the central mask (arcsec). No mask is used when set to None.
        :type cent_size: float
        :param edge_size: Outer radius (arcsec) beyond which pixels are masked. No outer mask is
                          used when set to None. If the value is larger than half the image size
                          then it will be set to half the image size.
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
        if mask_out_tag is not None:
            self.m_mask_out_port = self.add_output_port(mask_out_tag)
        else:
            self.m_mask_out_port is None
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_resize = resize
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
                    im_data):
        """
        Internal method which masks the central and outer parts of the images.
        """

        im_shape = (im_data.shape[1], im_data.shape[2])

        mask = np.ones(im_shape)

        if self.m_cent_size is not None or self.m_edge_size is not None:
            npix = im_shape[0]

            if npix%2 == 0:
                x_grid = y_grid = np.linspace(-npix/2+0.5, npix/2-0.5, npix)
            elif npix%2 == 1:
                x_grid = y_grid = np.linspace(-(npix-1)/2, (npix-1)/2, npix)

            xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)
            rr_grid = np.sqrt(xx_grid**2+yy_grid**2)

        if self.m_cent_size is not None:
            mask[rr_grid < self.m_cent_size] = 0.

        if self.m_edge_size is not None:
            if self.m_edge_size > npix/2.:
                self.m_edge_size = npix/2.
            mask[rr_grid > self.m_edge_size] = 0.

        if self.m_mask_out_port is not None:
            self.m_mask_out_port.set_all(mask)

        return im_data * mask

    def run(self):
        """
        Run method of the module. Normalizes, resizes, and masks the images.

        :return: None
        """

        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")

        if self.m_cent_size is not None:
            self.m_cent_size /= pixscale
        if self.m_edge_size is not None:
            self.m_edge_size /= pixscale

        if self.m_verbose:
            sys.stdout.write("Running PSFpreparationModule...")
            sys.stdout.flush()

        im_data = self.m_image_in_port.get_all()
        im_norm = self._im_norm(im_data)

        if self.m_resize is not None:
            im_data = self._im_resizing(im_data)

        im_data = self._im_masking(im_data)

        self.m_image_out_port.set_all(im_data, keep_attributes=True)
        self.m_image_out_port.add_attribute("im_norm", im_norm, static=False)
        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)
        if self.m_resize is not None:
            self.m_image_out_port.add_attribute("PIXSCALE", pixscale/self.m_resize)

        if self.m_resize is None:
            self.m_resize = -1

        if self.m_cent_size is None:
            self.m_cent_size = -1
        else:
            self.m_cent_size *= pixscale

        if self.m_edge_size is None:
            self.m_edge_size = -1
        else:
            self.m_edge_size *= pixscale

        attributes = {"resize": float(self.m_resize),
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
