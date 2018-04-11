"""
Modules with simple pre-processing tools.
"""

import warnings

import numpy as np

from skimage.transform import rescale

from PynPoint.Core.Processing import ProcessingModule


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

        :param size: New image size (arcsec). The same size will be used for both image dimensions.
        :type size: float
        :param center: Tuple (x0, y0) with the new image center. Python indexing starts at 0. The
                       center of the input images will be used when *center* is set to *None*.
        :type center: tuple, int
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output. Should be
                              different from *image_in_tag*.
        :type image_out_tag: str

        :return: None
        """

        super(CropImagesModule, self).__init__(name_in=name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_size = size
        self.m_center = center

    def run(self):
        """
        Run method of the module. Reduces the image size by cropping around an given position.

        :return: None
        """

        self.m_image_out_port.del_all_attributes()
        self.m_image_out_port.del_all_data()

        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")

        self.m_size = int(self.m_size/pixscale)

        def image_cutting(image_in,
                          size,
                          center):

            if center is None:
                x_off = (image_in.shape[0] - size) / 2
                y_off = (image_in.shape[1] - size) / 2

                if size > image_in.shape[0] or size > image_in.shape[1]:
                    raise ValueError("Input frame resolution smaller than target image resolution.")

                image_out = image_in[x_off:x_off+size, y_off:y_off+size]

            else:
                x_in = int(center[0] - size/2)
                y_in = int(center[1] - size/2)

                x_out = int(center[0] + size/2)
                y_out = int(center[1] + size/2)

                if x_in < 0 or y_in < 0 or x_out > image_in.shape[1] or y_out > image_in.shape[0]:
                    raise ValueError("Target image resolution does not fit inside the input frame "
                                     "resolution.")

                image_out = image_in[y_in:y_out, x_in:x_out]

            return image_out

        self.apply_function_to_images(image_cutting,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      "Running CropImagesModule...",
                                      func_args=(self.m_size, self.m_center))

        self.m_image_out_port.add_history_information("Image cropped", str(self.m_size))
        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)
        self.m_image_out_port.close_port()


class ScaleImagesModule(ProcessingModule):
    """
    Module for rescaling of an image.
    """

    def __init__(self,
                 scaling=(None, None),
                 name_in="scaling",
                 image_in_tag="im_arr",
                 image_out_tag="im_arr_scaled"):
        """
        Constructor of ScaleImagesModule.

        :param scaling: Tuple with the scaling factors for the image shape and pixel values,
                        (scaling_size, scaling_flux). Upsampling and downsampling of the image
                        corresponds to *scaling_size* > 1 and 0 < *scaling_size* < 1, respectively.
        :type scaling: tuple, float
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output. Should be
                              different from *image_in_tag*.
        :type image_out_tag: str

        :return: None
        """

        super(ScaleImagesModule, self).__init__(name_in=name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        if scaling[0] is None:
            self.m_scaling_size = 1.
        else:
            self.m_scaling_size = scaling[0]

        if scaling[1] is None:
            self.m_scaling_flux = 1.
        else:
            self.m_scaling_flux = scaling[1]

    def run(self):
        """
        Run method of the module. Rescales an image with a fifth order spline interpolation and a
        reflecting boundary condition.

        :return: None
        """

        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")

        def image_scaling(image_in,
                          scaling_size,
                          scaling_flux):

            sum_before = np.sum(image_in)

            tmp_image = rescale(image=np.asarray(image_in, dtype=np.float64),
                                scale=(scaling_size, scaling_size),
                                order=5,
                                mode="reflect")

            sum_after = np.sum(tmp_image)

            return tmp_image * (sum_before / sum_after) * scaling_flux

        self.apply_function_to_images(image_scaling,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      "Running ScaleImagesModule...",
                                      func_args=(self.m_scaling_size, self.m_scaling_flux,))

        history = "size  = "+str(self.m_scaling_size)+", flux = "+str(self.m_scaling_flux)
        self.m_image_out_port.add_history_information("Images scaled", history)
        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)
        self.m_image_out_port.add_attribute("PIXSCALE", pixscale/self.m_scaling_size)
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

        :param lines: Tuple with the number of additional lines in left, right, bottom, and top
                      direction.
        :type lines: tuple, int
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output. Should be
                              different from *image_in_tag*.
        :type image_out_tag: str

        :return: None
        """

        super(AddLinesModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_lines = lines

    def run(self):
        """
        Run method of the module. Adds lines of zero-value pixels to increase the size of an image.

        :return: None
        """

        shape_in = self.m_image_in_port.get_shape()

        if any(np.asarray(self.m_lines) < 0.):
            raise ValueError("The lines argument should contain values equal to or larger than "
                             "zero.")

        if shape_in.ndim == 3:
            shape_out = (shape_in[1]+int(self.m_lines[2])+int(self.m_lines[3]),
                         shape_in[2]+int(self.m_lines[0])+int(self.m_lines[1]))

        elif shape_in.ndim == 2:
            shape_out = (shape_in[0]+int(self.m_lines[2])+int(self.m_lines[3]),
                         shape_in[1]+int(self.m_lines[0])+int(self.m_lines[1]))

        if shape_out[0] != shape_out[1]:
            warnings.warn("The dimensions of the output images %s are not equal. PynPoint only "
                          "supports square images." % str(shape_out))

        def add_lines(image_in):
            image_out = np.zeros(shape_out)
            image_out[int(self.m_lines[2]):-int(self.m_lines[3]),
                      int(self.m_lines[0]):-int(self.m_lines[1])] = image_in

            return image_out

        self.apply_function_to_images(add_lines,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      "Running AddLinesModule...")

        self.m_image_out_port.add_history_information("Lines added", str(self.m_lines))
        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)
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

        :param lines: Tuple with the number of lines to be removed in left, right, bottom,
                      and top direction.
        :type lines: tuple, int
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output. Should be
                              different from *image_in_tag*.
        :type image_out_tag: str

        :return: None
        """

        super(RemoveLinesModule, self).__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)

        self.m_lines = lines

    def run(self):
        """
        Run method of the module. Removes the lines given by *lines* from each frame.

        :return: None
        """

        if self.m_image_in_port.tag == self.m_image_out_port.tag:
            raise ValueError("Input and output tags should be different.")

        def remove_lines(image_in):
            shape_in = image_in.shape
            return image_in[int(self.m_lines[2]):shape_in[0]-int(self.m_lines[3]),
                            int(self.m_lines[0]):shape_in[1]-int(self.m_lines[1])]

        self.apply_function_to_images(remove_lines,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      "Running RemoveLinesModule...")

        self.m_image_out_port.add_history_information("Lines removed", str(self.m_lines))
        self.m_image_out_port.copy_attributes_from_input_port(self.m_image_in_port)
        self.m_image_out_port.close_port()
