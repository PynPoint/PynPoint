"""
Pipeline modules for resizing of images.
"""

from __future__ import absolute_import

import math
import warnings

import numpy as np

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.image import crop_image, scale_image


class CropImagesModule(ProcessingModule):
    """
    Module for cropping of images around a given position.
    """

    def __init__(self,
                 size,
                 center=None,
                 name_in="crop_image",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_cropped"):
        """
        Constructor of CropImagesModule.

        Parameters
        ----------
        size : float
            New image size (arcsec). The same size will be used for both image dimensions.
        center : tuple(int, int)
            Tuple (x0, y0) with the new image center. Python indexing starts at 0. The center of
            the input images will be used when *center* is set to *None*.
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        image_out_tag : str
            Tag of the database entry that is written as output. Should be different from
            *image_in_tag*.

        Returns
        -------
        NoneType
            None
        """

        super(CropImagesModule, self).__init__(name_in=name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_size = size
        self.m_center = center

        if self.m_center is not None:
            self.m_center = (self.m_center[1], self.m_center[0]) # (y, x)

    def run(self):
        """
        Run method of the module. Decreases the image size by cropping around an given position.
        The module always returns odd-sized images.

        Returns
        -------
        NoneType
            None
        """

        self.m_image_out_port.del_all_attributes()
        self.m_image_out_port.del_all_data()

        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")

        self.m_size = int(math.ceil(self.m_size/pixscale))

        def _image_cutting(image_in,
                           size,
                           center):

            return crop_image(image_in, center, size)

        self.apply_function_to_images(_image_cutting,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      "Running CropImagesModule...",
                                      func_args=(self.m_size, self.m_center))

        history = "image size [pix] = "+str(self.m_size)
        self.m_image_out_port.add_history("CropImagesModule", history)
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.close_port()


class ScaleImagesModule(ProcessingModule):
    """
    Module for rescaling of an image.
    """

    def __init__(self,
                 scaling=(None, None, None),
                 pixscale=False,
                 name_in="scaling",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_scaled"):
        """
        Constructor of ScaleImagesModule.

        Parameters
        ----------
        scaling : tuple(float, float, float)
            Tuple with the scaling factors for the image size and flux, (scaling_x, scaling_y,
            scaling_flux). Upsampling and downsampling of the image corresponds to
            *scaling_x/y* > 1 and 0 < *scaling_x/y* < 1, respectively.
        pixscale : bool
            Adjust the pixel scale by the average scaling in x and y direction.
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        image_out_tag : str
            Tag of the database entry that is written as output. Should be different from
            *image_in_tag*.

        Returns
        -------
        NoneType
            None
        """

        super(ScaleImagesModule, self).__init__(name_in=name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        if len(scaling) == 2:
            warnings.warn("The 'scaling' parameter requires three values: (scaling_x, scaling_y, "
                          "scaling_flux). Using the same scaling in x and y direction...")

            scaling = (scaling[0], scaling[0], scaling[1])

        elif len(scaling) < 2 or len(scaling) > 3:
            raise ValueError("The 'scaling' parameter requires three values: (scaling_x, "
                             "scaling_y, scaling_flux).")

        if scaling[0] is None:
            self.m_scaling_x = 1.
        else:
            self.m_scaling_x = scaling[0]

        if scaling[1] is None:
            self.m_scaling_y = 1.
        else:
            self.m_scaling_y = scaling[1]

        if scaling[2] is None:
            self.m_scaling_flux = 1.
        else:
            self.m_scaling_flux = scaling[2]

        self.m_pixscale = pixscale

    def run(self):
        """
        Run method of the module. Rescales an image with a fifth order spline interpolation and a
        reflecting boundary condition.

        Returns
        -------
        NoneType
            None
        """

        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")

        def _image_scaling(image_in,
                           scaling_x,
                           scaling_y,
                           scaling_flux):

            tmp_image = scale_image(image_in, scaling_x, scaling_y)

            return scaling_flux * tmp_image

        self.apply_function_to_images(_image_scaling,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      "Running ScaleImagesModule...",
                                      func_args=(self.m_scaling_x,
                                                 self.m_scaling_y,
                                                 self.m_scaling_flux,))

        history = "scaling = ("+str("{:.2f}".format(self.m_scaling_x)) + ", " + \
                  str("{:.2f}".format(self.m_scaling_y)) + ", " + \
                  str("{:.2f}".format(self.m_scaling_flux)) + ")"

        self.m_image_out_port.add_history("ScaleImagesModule", history)
        self.m_image_out_port.copy_attributes(self.m_image_in_port)

        if self.m_pixscale:
            mean_scaling = (self.m_scaling_x+self.m_scaling_y)/2.
            self.m_image_out_port.add_attribute("PIXSCALE", pixscale/mean_scaling)

        self.m_image_out_port.close_port()


class AddLinesModule(ProcessingModule):
    """
    Module to add lines of pixels to increase the size of an image.
    """

    def __init__(self,
                 lines,
                 name_in="add_lines",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_add"):
        """
        Constructor of AddLinesModule.

        Parameters
        ----------
        lines : tuple(int, int, int, int)
            The number of lines that are added in left, right, bottom, and top direction.
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        image_out_tag : str
            Tag of the database entry that is written as output, including the images with
            increased size. Should be different from *image_in_tag*.

        Returns
        -------
        NoneType
            None
        """

        super(AddLinesModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_lines = np.asarray(lines)

    def run(self):
        """
        Run method of the module. Adds lines of zero-value pixels to increase the size of an image.

        Returns
        -------
        NoneType
            None
        """

        shape_in = self.m_image_in_port.get_shape()

        shape_out = (shape_in[-2]+int(self.m_lines[2])+int(self.m_lines[3]),
                     shape_in[-1]+int(self.m_lines[0])+int(self.m_lines[1]))

        if shape_out[0] != shape_out[1]:
            warnings.warn("The dimensions of the output images %s are not equal. PynPoint only "
                          "supports square images." % str(shape_out))

        def _add_lines(image_in):
            image_out = np.zeros(shape_out)

            image_out[int(self.m_lines[2]):int(self.m_lines[3]),
                      int(self.m_lines[0]):int(self.m_lines[1])] = image_in

            return image_out

        self.m_lines[1] = shape_out[1] - self.m_lines[1] # right side of image
        self.m_lines[3] = shape_out[0] - self.m_lines[3] # top side of image

        self.apply_function_to_images(_add_lines,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      "Running AddLinesModule...")

        history = "number of lines = "+str(self.m_lines)
        self.m_image_out_port.add_history("AddLinesModule", history)
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.close_port()


class RemoveLinesModule(ProcessingModule):
    """
    Module to decrease the dimensions of an image by removing lines of pixels.
    """

    def __init__(self,
                 lines,
                 name_in="cut_top",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_cut"):
        """
        Constructor of RemoveLinesModule.

        Parameters
        ----------
        lines : tuple(int, int, int, int)
            The number of lines that are removed in left, right, bottom, and top direction.
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        image_out_tag : str
            Tag of the database entry that is written as output, including the images with
            decreased size. Should be different from *image_in_tag*.

        Returns
        -------
        NoneType
            None
        """

        super(RemoveLinesModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_lines = lines

    def run(self):
        """
        Run method of the module. Removes the lines given by *lines* from each frame.

        Returns
        -------
        NoneType
            None
        """

        def _remove_lines(image_in):
            shape_in = image_in.shape

            return image_in[int(self.m_lines[2]):shape_in[0]-int(self.m_lines[3]),
                            int(self.m_lines[0]):shape_in[1]-int(self.m_lines[1])]

        self.apply_function_to_images(_remove_lines,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      "Running RemoveLinesModule...")

        history = "number of lines = "+str(self.m_lines)
        self.m_image_out_port.add_history("RemoveLinesModule", history)
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.close_port()
